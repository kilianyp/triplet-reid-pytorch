import torch.nn as nn
import torch


choices = ["BatchHard", "BatchSoft"]

def calc_cdist(a, b, metric='euclidean'):
    if metric == 'euclidean':
        diff = a[:, None, :] - b[None, :, :]
        # derivative of sqrt(0) is not defined!
        return torch.sqrt(torch.sum(torch.mul(diff, diff), dim=2) + 1e-12)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)


class BatchHard(nn.Module):

    def __init__(self, m):
        super(BatchHard, self).__init__()
        self.m = m
        self.name = "BatchHard"

    def forward(self, cdist, pids):
        """Caluclates the batch hard loss as in in arxiv.org/abs/1703.07737.
        
        Args:
            cdist (3D Tensor): Cross-distance between two 2D Vectors.
            pids: 1D tensor of the identities of shape [batch_size].
        """
        mask_max = pids[None, :] == pids[:, None]

        mask_min = 1 - mask_max.data
        furthest_positive = torch.max(cdist * mask_max.float(), dim=0)[0] #TODO dimension?
        furthest_negative = [torch.min(cdists[mask_min[id]]) for id, cdists in enumerate(cdist)]
        furthest_negative = torch.stack(furthest_negative).squeeze() # stacking adds another dimension
        
        diff = furthest_positive - furthest_negative
        if isinstance(self.m, float):
            diff = (diff + self.m).clamp(min=0)
        elif self.m.lower() == "soft":
            soft = torch.nn.Softplus()
            diff = soft(diff)
        elif self.m.lower() == "none":
            pass
        else:
            raise NotImplementedError("The margin %s is not implemented in BatchHard!" % self.m)

        return diff

import pyro
import numpy as np
class BatchSoft(nn.Module):
    """BatchSoft implementation using softmax."""

    def __init__(self, m, T=1.0):
        """
        Args:
            m: margin
            T: Softmax temperature
        """
        super(BatchSoft, self).__init__()
        self.m = m
        self.T = T
        self.name = "BatchSoft-%s-%f" % (str(m), T)
        self.softmax = torch.nn.Softmax()
        self.softmin = torch.nn.Softmin()

    def forward(self, cdist, pids):
        """Calculates the batch soft.
        Instead of picking the hardest example through argmax or argmin
        a softmax (softmin) is used to sample and use less difficult examples as well.
        """
        # mask where all positivies are set to true
        mask_pos = pids[None, :] == pids[:, None]
        mask_neg = 1 - mask_pos.data
        
        # only one copy
        cdist_max = cdist.clone()
        cdist_max[mask_neg] = -np.inf
        
        cdist_min = cdist.clone()
        cdist_min[mask_pos] = np.inf
        idx_pos = pyro.distributions.categorical(self.softmax(cdist_max/self.T))
        idx_neg = pyro.distributions.categorical(self.softmin(cdist_min/self.T))
        max_pos = cdist.masked_select(idx_pos.byte())
        min_neg = cdist.masked_select(idx_neg.byte())
        
        diff = max_pos - min_neg

        if isinstance(self.m, float):
            diff = (diff + self.m).clamp(min=0)
        elif self.m.lower() == "soft":
            soft = torch.nn.Softplus()
            diff = soft(diff)
        elif self.m.lower() == "none":
            pass
        else:
            raise NotImplementedError("The margin %s is not implemented in BatchHard!" % self.m)


        return diff
