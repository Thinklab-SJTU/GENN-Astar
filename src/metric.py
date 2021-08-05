import torch

default_mapping = torch.sparse.FloatTensor(
    torch.LongTensor([[0, 1]]),
    torch.FloatTensor([0, 1])
)


def binary_to_int(binary):
    assert torch.all(torch.logical_or(binary == 0, binary == 1))
    mask = 2**torch.arange(binary.shape[-1]).to(binary.device, binary.dtype)
    return torch.sum(mask * binary, dim=-1).to(torch.long)


def node_metric(node1, node2, mapping=default_mapping):
    """
    :param node1: one-hot features of graph 1
    :param node2: one-hot features of graph 2
    :param mapping:
    :return:
    """
    encoding = binary_to_int(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)))

    assert type(mapping) is torch.Tensor

    if mapping.is_sparse:
        mapping = mapping.to_dense()

    return mapping[encoding].to(node1.device)


def generate_mapping(map_dic: dict):
    """
    :param map_dic: An example of map_dic:
    :return:
    """

