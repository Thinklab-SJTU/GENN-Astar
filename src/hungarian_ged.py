import torch
from src.hungarian import hungarian

def hungarian_ged(node_cost_mat, n1, n2):
    assert node_cost_mat.shape[-2] == n1+1
    assert node_cost_mat.shape[-1] == n2+1
    device = node_cost_mat.device
    upper_left = node_cost_mat[:n1, :n2]
    upper_right = torch.full((n1, n1), float('inf'), device=device)
    torch.diagonal(upper_right)[:] = node_cost_mat[:-1, -1]
    lower_left = torch.full((n2, n2), float('inf'), device=device)
    torch.diagonal(lower_left)[:] = node_cost_mat[-1, :-1]
    lower_right = torch.zeros((n2, n1), device=device)

    large_cost_mat = torch.cat((torch.cat((upper_left, upper_right), dim=1),
                                torch.cat((lower_left, lower_right), dim=1)), dim=0)

    large_pred_x = hungarian(-large_cost_mat)
    pred_x = torch.zeros_like(node_cost_mat)
    pred_x[:n1, :n2] = large_pred_x[:n1, :n2]
    pred_x[:-1, -1] = torch.sum(large_pred_x[:n1, n2:], dim=1)
    pred_x[-1, :-1] = torch.sum(large_pred_x[n1:, :n2], dim=0)

    ged_lower_bound = torch.sum(pred_x * node_cost_mat)

    return pred_x, ged_lower_bound