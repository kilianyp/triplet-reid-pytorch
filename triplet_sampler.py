import numpy as np
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

        P_perm = np.random.permutation(len(self.target2idx))
        for p in P_perm:
            images = self.target2idx[p]
            K_perm = np.random.permutation(len(images))
            # fill up by repeating the permutation
            if len(images) < self.K:
                K_perm = np.tile(K_perm, self.K//len(images))
                left = self.K - len(K_perm)
                K_perm = np.concatenate((K_perm, K_perm[:left]))
            for k in range(self.K):
                batch.append(images[K_perm[k]])
        
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
