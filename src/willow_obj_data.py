import os
import os.path as osp
import numpy as np
import random
import torch
from pathlib import Path
import scipy.io as sio
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import itertools

from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip, extract_tar)
from torch_geometric.utils import to_undirected, dense_to_sparse, to_dense_adj

'''
Important Notice: Face image 160 contains only 8 labeled keypoints (should be 10)
'''

class GEDDataset(InMemoryDataset):
    url = 'http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip'

    willow_classes = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']

    norm_value = 300

    def __init__(self, root, name, set='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        assert self.name in ['Willow']
        super(GEDDataset, self).__init__(root, transform, pre_transform,
                                         pre_filter)
        class_name = 'Car'

        path = self.processed_paths[self.willow_classes.index(class_name)]
        self.data, self.slices = torch.load(path)
        path = osp.join(self.processed_dir, '{}_ged.pt'.format(class_name))
        self.ged = torch.load(path)
        path = osp.join(self.processed_dir, '{}_norm_ged.pt'.format(class_name))
        self.norm_ged = torch.load(path)

    @property
    def raw_file_names(self):
        return ['WILLOW-ObjectClass']

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(x) for x in self.willow_classes]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        willow_dataset = WillowObject(self.raw_paths[0])
        for cls in willow_dataset.classes:
            data_list = []
            Ns = []
            for i, mat_file in enumerate(willow_dataset.mat_list[cls]):
                anno_dict = willow_dataset.get_anno_dict(mat_file, cls)
                kpts = np.array([(kp['x'], kp['y']) for kp in anno_dict['keypoints']]) / self.norm_value
                Ns.append(kpts.shape[0])
                adj = torch.from_numpy(delaunay_triangulate(kpts)).to(dtype=torch.float)
                kpts = torch.from_numpy(kpts).to(dtype=torch.float)
                distance = torch.sqrt(torch.sum((kpts.unsqueeze(0) - kpts.unsqueeze(1)) ** 2, dim=-1))
                edge_index, edge_attr = dense_to_sparse(distance * adj)

                data = Data(edge_index=edge_index, edge_attr=edge_attr, i=i)
                data.num_nodes = Ns[-1]

                #data.x = kpts

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[self.willow_classes.index(cls)])

            mat = torch.full((len(data_list), len(data_list)), float('inf'))
            for (i, mat_i), (j, mat_j) in itertools.combinations(enumerate(willow_dataset.mat_list[cls]), 2):
                anno_dict_i = willow_dataset.get_anno_dict(mat_i, cls)
                anno_dict_j = willow_dataset.get_anno_dict(mat_j, cls)
                perm_mat = torch.from_numpy(willow_dataset.get_pair((anno_dict_i, anno_dict_j)))

                adj_i = to_dense_adj(data_list[i].edge_index, edge_attr=data_list[i].edge_attr).squeeze(0)
                adj_j = to_dense_adj(data_list[j].edge_index, edge_attr=data_list[j].edge_attr).squeeze(0)

                ged = torch.sum(torch.abs(torch.chain_matmul(perm_mat.t(), adj_i, perm_mat) - adj_j))
                mat[i, j] = ged
                mat[j, i] = ged
            torch.diagonal(mat)[:] = 0

            path = osp.join(self.processed_dir, '{}_ged.pt'.format(cls))
            torch.save(mat, path)

            N = torch.tensor(Ns, dtype=torch.float)
            norm_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

            path = osp.join(self.processed_dir, '{}_norm_ged.pt'.format(cls))
            torch.save(norm_mat, path)

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class WillowObject:
    def __init__(self, dataset_path):
        self.classes = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
        self.kpt_len = [10, 10, 10, 10, 10]

        self.root_path = Path(dataset_path)

        self.rand_outlier = 0

        self.mat_list = dict()
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob('*.mat')]
            if cls_name == 'Face':
                cls_mat_list.remove(self.root_path / cls_name / 'image_0160.mat')
                assert not self.root_path / cls_name / 'image_0160.mat' in cls_mat_list
            self.mat_list[cls_name] = sorted(cls_mat_list)

    def get_pair(self, anno_pair):
        """
        Randomly get a pair of objects from WILLOW-object dataset
        :return: (pair of data, groundtruth permutation matrix)
        """
        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    if keypoint['name'] != 'outlier':
                        perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return perm_mat

    def get_anno_dict(self, mat_file, cls):
        """
        Get an annotation dict from .mat annotation
        """
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        img_name = mat_file.stem + '.png'
        img_file = mat_file.parent / img_name

        struct = sio.loadmat(mat_file.open('rb'))
        kpts = struct['pts_coord']

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]
            img = np.array(img)

        keypoint_list = []
        for idx, keypoint in enumerate(np.split(kpts, kpts.shape[1], axis=1)):
            attr = {
                'name': idx,
                'x': float(keypoint[0]),
                'y': float(keypoint[1])
            }
            keypoint_list.append(attr)

        for idx in range(self.rand_outlier):
            attr = {
                'name': 'outlier',
                'x': random.uniform(0, w),
                'y': random.uniform(0, h)
            }
            keypoint_list.append(attr)

        random.shuffle(keypoint_list)

        anno_dict = dict()
        anno_dict['image'] = img
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = cls

        return anno_dict

    def len(self, cls):
        if type(cls) == int:
            cls = self.classes[cls]
        assert cls in self.classes
        return len(self.mat_list[self.classes.index(cls)])


def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
    """
    n = P.shape[0]
    if n < 3:
        A = fully_connect(P)
    else:
        try:
            d = Delaunay(P)
            #assert d.coplanar.size == 0, 'Delaunay triangulation omits points.'
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print('Delaunay triangulation error detected. Return fully-connected graph.')
            print('Traceback:')
            print(err)
            A = fully_connect(P)
    return A


def fully_connect(P: np.ndarray, thre=None):
    """
    Fully connect a graph.
    :param P: point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix A
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    if thre is not None:
        for i in range(n):
            for j in range(i):
                if np.linalg.norm(P[i] - P[j]) > thre:
                    A[i, j] = 0
                    A[j, i] = 0
    return A