import numpy as np


def create_pids2idxs(data_source):
    """Creates a mapping between pids and indexes of images for that pid.
    Returns:
        2D List with pids => idx
    """
    pid2imgs = {}
    for idx, (img, target) in enumerate(data_source.imgs):
        if target not in pid2imgs:
            pid2imgs[target] = [idx]
        else:
            pid2imgs[target].append(idx)
    return list(pid2imgs.values())


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

        self.pid2imgs = create_pids2idxs(self.data_source)

    def __iter__(self):
        batch = []
        P_perm = np.random.permutation(len(self.pid2imgs))
        for p in P_perm:
            images = self.pid2imgs[p]
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
