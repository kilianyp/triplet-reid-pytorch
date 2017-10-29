import torch.nn as nn
import torch



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
