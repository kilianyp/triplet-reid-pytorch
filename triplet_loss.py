import torch.nn as nn
import torch



def calc_cdist(a, b, metric='euclidean'):
    if metric == 'euclidean':
        diff = a[None, :, :] - b[:, None, :]
        return torch.sqrt(torch.sum(torch.mul(diff, diff), dim=2))
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)


class BatchHard(nn.Module):

    def __init__(self, m):
        super(BatchHard, self).__init__()
        self.m = m

    def forward(self, embd, pids):
        """Caluclates the batch hard loss as in in arxiv.org/abs/1703.07737.
        
        Args:
            embd: 2D tensor of the embedding vector shape [batch_size, Features]
            pids: 1D tensor of the identities of shape [batch_size]. Tensor has to be of 
        """

        cdist = calc_cdist(embd, embd)
        mask_max = pids[None, :] == pids[:, None]

        mask_min = 1 - mask_max.data

        #furthest_positive = torch.max(
        #        torch.mask_select(cdist, mask_max).view(self.K, self.batch_size),
        #        dim=0)[0]  # max returns tuple, also indexes
        #closest_negative = torch.min(
        #        torch.mask_select(cdist, mask_min).view(batch_size - K,batch_size),
        #        dim=0)[0]
        furthest_positive = torch.max(cdist * mask_max.float(), dim=0)[0]
        furthest_positive -= torch.mean(furthest_positive)
        
        furthest_negative = [torch.max(cdists[mask_min[id]])[0] for id, cdists in enumerate(cdist)]
        furthest_negative = torch.stack(furthest_negative)
        print(furthest_negative)
        furthest_negative = torch.min(furthest_negative, dim=0)[0]

        diff = furthest_positive - furthest_negative

        if isinstance(self.m, float):
            diff = (diff + self.m).clamp(min=0)
        elif self.m.lower() == "soft":
            soft = torch.SoftPlus()
            diff = soft(diff)
        elif self.m.lower() == "none":
            pass
        else:
            raise NotImplementedError("The margin %s is not implemented in BatchHard!" % self.m)

        return diff
