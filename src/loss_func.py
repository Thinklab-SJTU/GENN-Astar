import torch
import torch.nn as nn
import torch.nn.functional as F
from src.hungarian import hungarian


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    Proposed by Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, *gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        try:
            assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_perm)
            raise err

        loss = torch.zeros(batch_num, device=pred_perm.device)
        for b in range(batch_num):
            batch_slice = [b, slice(pred_ns[b])]
            for gtn in gt_ns:
                batch_slice.append(slice(gtn[b]))
            loss[b] = F.binary_cross_entropy(
                pred_perm[batch_slice],
                gt_perm[batch_slice],
                reduction='sum') / pred_ns[b].to(loss.dtype).to(pred_perm.device)

        return loss


class CrossEntropyLossHung(nn.Module):
    """
    Cross entropy loss between two permutations with Hungarian attention.
    Proposed by Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention. ICLR 2020.
    """
    def __init__(self):
        super(CrossEntropyLossHung, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        lap_solver = hungarian

        dis_pred = lap_solver(pred_perm, pred_ns, gt_ns)
        # dis_pred = dis_pred.detach()

        # pdb.set_trace()
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm > 1.0] = 1.0 # Hung
        pred_perm = torch.mul(ali_perm, pred_perm)
        gt_perm = torch.mul(ali_perm, gt_perm)
        # pdb.set_trace()
        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        # pdb.set_trace()
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)
        # pdb.set_trace()
        return loss / n_sum


class RobustLoss(nn.Module):
    """
    RobustLoss Criterion computes a robust loss function.
    Proposed by Zanfir et al. Deep Learning of Graph Matching. CVPR 2018.
    L = Sum(Phi(d_i - d_i^gt)),
        where Phi(x) = sqrt(x^T * x + epsilon)
    Parameter: a small number for numerical stability epsilon
               (optional) division taken to normalize the loss norm
    Input: displacement matrix d1
           displacement matrix d2
           (optional)dummy node mask mask
    Output: loss value
    """
    def __init__(self, epsilon=1e-5, norm=None):
        super(RobustLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1, d2, mask=None):
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class FocalLoss(nn.Module):
    """
    Focal loss between two permutations.
    """
    def __init__(self, alpha=1., gamma=0., eps=1e-15):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_perm[b, :pred_ns[b], :gt_ns[b]]
            y = gt_perm[b, :pred_ns[b], :gt_ns[b]]
            loss += torch.sum(
                #- self.alpha * (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                #- (1 - self.alpha) * x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
                - (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum


class InnerProductLoss(nn.Module):
    """
    Inner product loss for self-supervised problems.
    """
    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            raise err

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(pred_ns[b]), slice(gt_ns[b])]
            loss -= torch.sum(pred_perm[batch_slice] * gt_perm[batch_slice])
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum

