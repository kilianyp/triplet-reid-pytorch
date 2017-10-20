import torch
class TripletBatchSampler(object):
    """Sampler to create batches with P x K.
        
        Only returns indices.
        
    """
    def __init__(self, P, K, data_source, drop_last=True):
        self.P = P
        self.K = K
        self.batch_size = self.P * self.K
        self.data_source = data_source
        self.drop_last = drop_last

        # TODO create target2idx here?
        self.target2idx = self.data_source.target2idx


    def __iter__(self):
        batch = []

        P_perm = torch.randperm(len(self.target2idx))
        for p in P_perm:
            person = self.target2idx[p]
            K_perm = torch.randperm(len(person))
            if len(person) < self.K:
                print("Warning: not enough images for person id %d (%d/%d)"
                      % (p, len(person), self.K))
                continue
            for k in range(self.K):
                batch.append(person[K_perm[k]])
        
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 1 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.P
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.P
