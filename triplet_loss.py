import torch
import torch.nn as nn
import torch.nn.functional as F

import logger as log
import numpy as np

choices = ["BatchHard", "BatchSoft", "BatchHardWithSoftmaxLoss"]

def calc_cdist(a, b, metric='euclidean'):
    diff = a[:, None, :] - b[None, :, :]
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(diff*diff, dim=2) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(diff*diff, dim=2)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=2)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)


def _apply_margin(x, m):
    if isinstance(m, float):
        return (x + m).clamp(min=0)
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % self.m)


def batch_hard(cdist, pids, margin):
    """Computes the batch hard loss as in arxiv.org/abs/1703.07737.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
    """
    mask_pos = (pids[None, :] == pids[:, None]).float()

    ALMOST_INF = 9999.9
    furthest_positive = torch.max(cdist * mask_pos, dim=0)[0]
    furthest_negative = torch.min(cdist + ALMOST_INF*mask_pos, dim=0)[0]
    #furthest_negative = torch.stack([
    #    torch.min(row_d[row_m]) for row_d, row_m in zip(cdist, mask_neg)
    #]).squeeze() # stacking adds an extra dimension

    return _apply_margin(furthest_positive - furthest_negative, margin)


class BatchHard(nn.Module):
    def __init__(self, m):
        super(BatchHard, self).__init__()
        self.name = "BatchHard(m={})".format(m)
        self.m = m

    def forward(self, cdist, pids):
        return batch_hard(cdist, pids, self.m)


def batch_soft(cdist, pids, margin, T=1.0):
    """Calculates the batch soft.
    Instead of picking the hardest example through argmax or argmin,
    a softmax (softmin) is used to sample and use less difficult examples as well.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
        T (float): The temperature of the softmax operation.
    """
    # mask where all positivies are set to true
    mask_pos = pids[None, :] == pids[:, None]
    mask_neg = 1 - mask_pos.data

    # only one copy
    cdist_max = cdist.clone()
    cdist_max[mask_neg] = -float('inf')
    cdist_min = cdist.clone()
    cdist_min[mask_pos] = float('inf')

    # NOTE: We could even take multiple ones by increasing num_samples,
    #       the following `gather` call does the right thing!
    idx_pos = torch.multinomial(F.softmax(cdist_max/T, dim=1), num_samples=1)
    idx_neg = torch.multinomial(F.softmin(cdist_min/T, dim=1), num_samples=1)
    positive = cdist.gather(dim=1, index=idx_pos)[:,0]  # Drop the extra (samples) dim
    negative = cdist.gather(dim=1, index=idx_neg)[:,0]

    return _apply_margin(positive - negative, margin)


class BatchSoft(nn.Module):
    """BatchSoft implementation using softmax."""

    def __init__(self, m, T=1.0):
        """
        Args:
            m: margin
            T: Softmax temperature
        """
        super(BatchSoft, self).__init__()
        self.name = "BatchSoft(m={}, T={})".format(m, T)
        self.m = m
        self.T = T

    def forward(self, cdist, pids):
        return batch_soft(cdist, pids, self.m, self.T)


class BatchHardWithSoftmaxLoss(nn.Module):
    """SoftmaxLoss or Softmax Classifier uses the NegativeLogLikelyLoss
    and the softmax function.
    The torch implementation of CrossEntropy includes the softmax.

    """

    def __init__(self, m):
        super().__init__()
        self.batch_hard = BatchHard(m)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.name = "BatchHardWithSofmtax(m={})".format(m)

    def forward(self, cdist, pids, features):
        batch_hard_loss = self.batch_hard(cdist, pids)
        cross_entropy_loss = self.cross_entropy(features, pids)
        ce_loss = float(var2num(cross_entropy_loss))
        bh_loss = float(var2num(torch.mean(batch_hard_loss)))
        print("bh loss {:.3f} ce loss: {:.3f}".format(bh_loss, ce_loss))
        log.write("loss", (bh_loss, ce_loss), dtype=np.float32)
        return batch_hard_loss + cross_entropy_loss

def var2num(x):
    return x.data.cpu().numpy()

