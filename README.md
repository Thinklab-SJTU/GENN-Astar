# GENN-A*
This repository contains the implementation of GENN-A*, which aims to accelerate the A* solver for graph edit distance problem based on Graph Neural Network.

GENN-A* is presented in the following CVPR 2021 paper:
    
* Runzhong Wang, Tianqi Zhang, Tianshu Yu, Junchi Yan, Xiaokang Yang. 
_Combinatorial Learning of Graph Edit Distance via Dynamic Embedding._
CVPR 2021. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.pdf)

Graph Edit Neural Network (GENN) aided A* algorithm works by replacing the heuristic prediction module in A* by GNN. Since the accuracy of heuristic prediction is crucial for the performance of A*, our approach can significantly improve the efficiency of A*.

And it is worth noting that: GENN-A* no longer guarantees to find the optimal GED because the optimality guarantee in A* (heuristic prediction must be the lower bound of optimal GED) is broken by the unbounded graph neural network module. However, the performance loss is not significant in our experiment compared to the cut of inference time.

## Get Started
### Install Requirements
The codebase is built and tested with Python 3.8.8. Firstly install required packages:
```
pip install -r requrements.txt
```

Secondly, install Pytorch 1.6 with optionally GPU support (you may run this code repository with only CPU).

Finally, install torch-geometric. First specify the torch and CPU/CUDA version to be used:
```
export CUDA=cpu
export TORCH=1.6.0
```
If you are using the CUDA version of pytorch, please modify ``export CUDA=CPU`` into ``export CUDA=cuxxx`` (e.g. ``export CUDA=cu101`` for CUDA 10.1). Please refer to [the official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for more details.
```
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==1.6.3
```

### Build Cython extension
The A* algorithm is implemented by Cython for better efficiency. Follow these instructions to build the Cython code:
```
cd src && python3 setup.py build_ext --inplace
```

### Datasets
The datasets are handled with the help of [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GEDDataset),
and will be automatically downloaded up on requirement.

## Pretrained Models
We provide the pretrained models to reproduce the results in our paper on [google drive](https://drive.google.com/drive/folders/1mUpwHeW1RbMHaNxX_PZvD5HrWvyCQG8y?usp=sharing).

## Run the Experiment
Our method supports two work modes: 1) fast neural network regression model, denoted as **GENN** and 2) a neural network guided A* solver, denoted as **GENN-A***.

For more detailed configurations, please refer to ``main.py --help``

### AIDS700nef
Train the neural network module and test on validation set:
```
python main.py --dataset AIDS700nef --epochs 50000 --weight-decay 5.0e-5 --batch-size 128 --learning-rate 0.001
```
the trained model weights are stored in ``best_genn_AIDS700nef_gcn.pt``

Test GENN on test set:
```
python main.py --test --dataset AIDS700nef
```
the model weights are loaded from ``best_genn_AIDS700nef_gcn.pt``

Finetune with ground-truth edit paths:
```
python main.py --enable-astar --dataset AIDS700nef --epochs 1000 --weight-decay 5.0e-5 --batch-size 1 --learning-rate 0.001
```
the trained model weights are stored in ``best_genn_AIDS700nef_gcn_astar.pt``

Test GENN-A* on test set:
```
python main.py --test --dataset AIDS700nef --enable-astar --astar-use-net --batch-size 1
```
the model weights are loaded from ``best_genn_AIDS700nef_gcn_astar.pt``

### LINUX
Train the neural network module and test on validation set:
```
python main.py --dataset LINUX --epochs 50000 --weight-decay 5.0e-5 --batch-size 128 --learning-rate 0.001
```
the trained model weights are stored in ``best_genn_LINUX_gcn.pt``

Test GENN on test set:
```
python main.py --test --dataset LINUX
```
the model weights are loaded from ``best_genn_LINUX_gcn.pt``

Finetune with ground-truth edit paths:
```
python main.py --enable-astar --dataset LINUX --epochs 1000 --weight-decay 5.0e-5 --batch-size 1 --learning-rate 0.001
```
the trained model weights are stored in ``best_genn_LINUX_gcn_astar.pt``

Test GENN-A* on test set:
```
python main.py --test --dataset LINUX --enable-astar --astar-use-net --batch-size 1
```
the model weights are loaded from ``best_genn_LINUX_gcn_astar.pt``

### Willow-Cars
Train the neural network module:
```
python main.py --dataset Willow --gnn-operator spline --epochs 50000 --weight-decay 5.0e-5 --batch-size 16 --learning-rate 0.001
```
the trained model weights are stored in ``best_genn_Willow_spline.pt``

Finetune with ground truth edit paths:
```
python main.py --dataset Willow --enable-astar --gnn-operator spline --epochs 1000 --weight-decay 5.0e-5 --batch-size 1 --learning-rate 0.001
```
the trained model weights are stored in ``best_genn_Willow_spline_astar.pt``

Test GENN-A* on test set:
```
python main.py --test --dataset Willow --enable-astar --astar-use-net --gnn-operator spline --batch-size 1
```
the model weights are loaded from ``best_genn_Willow_spline_astar.pt``

Willow-Cars dataset does not support testing with GENN because it is trained with surrogate labels instead of ground truth GED labels.

Besides, this repository is originally developed for problems where the ground truth GED is available. The metrics printed for Willow dataset are computed with surrogate labels which are incorrect, and users should compute the correct metric by the result files stored in ``results/``.

## A Note about the Evaluation Time
GENN-A* purely runs on CPU, and the evaluation of GENN-A* is faster than the classic A* algorithm. However, evaluating GENN-A* may be still time-consuming, and we provide some possible reasons and clues:
1. We follow the testing protocol by existing GED learning papers that all pairs of graphs are compared. If there are N graphs in the test dataset, N(N-1) NP-hard GED problems need to be solved.
    * Set a beam search width for A* so that it will predict a result more efficiently (at the cost of decreased accuracy);
    * Sample a subset of pairs of graphs during evaluation (and please note that the result is no longer a fair comparison with existing GED learning papers);
2. Since GENN-A* purely runs on CPU, the CPU speed is very important. Our experiments are run on i7-9700K CPU @ 3.60GHz, but I know many deep learning servers are equipped with Xeon CPUs @ 2.xGHz that can make the algorithm slower. 
    * You may try running our code on faster CPUs (say, on your gaming computer whose CPU is usually faster).
3. The nature of A* algorithm is doing exaustive tree search on all possible mathcings to find the optimal. 
    * If you care more about faster feasible solutions, you may also try our PPO-BiHyb ([[paper]](https://arxiv.org/abs/2106.04927), [[code]](https://github.com/Thinklab-SJTU/ppo-bihyb#graph-edit-distance-ged))

## Some Remarks on Using GENN-A* on Your Own Dataset
We provide some remarks on applying GENN-A* to your own dataset. 
* Like other machine learning tasks, you should split your data into three subsets: training set, validation set and testing set.
* You should implement a dataloader following the [torch_geometric API](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).
* You should implement a ``node_metric`` function in ``astar_genn.py`` to reflect the node edit cost of your problem. It is worth noting that we do not implement customized edge edit cost, and we currently support ``abs(e_ij - e_ab)``. You may modify the source code in ``astar_genn.py`` to fit your settings.
* The training process may be different depending on whether you have ground truth labels of the edit cost:
    * If you have ground truth GED values (see our examples AIDS700nef and LINUX), you may simply train the network with ground truth GED, and then run GENN with A* on test data. 
    * If you have no ground truth GED values but only some surrogate labels (see our example Willow-Cars), you should firstly run the training code, then finetune with ground truth edit paths (which is time-consuming), and finally test the learned GENN-A* on test data.

## Credits and Citation

If you find this repository useful in your research, please cite:
```
@inproceedings{WangCVPR21,
  author = {Wang, Runzhong and Zhang, Tianqi and Yu, Tianshu and Yan, Junchi and Yang, Xiaokang},
  title = {Combinatorial Learning of Graph Edit Distance via Dynamic Embedding},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

We would also like to give credits to the following repositories and thank their great wok:
* [Extended-SimGNN](https://github.com/gospodima/Extended-SimGNN)
* [SimGNN](https://github.com/yunshengb/SimGNN)
* [GEDLIB](https://github.com/dbblumenthal/gedlib)
