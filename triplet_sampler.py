import numpy as np


def create_pids2idxs(data_source):
    """Creates a mapping between pids and indexes of images for that pid.
    Returns:
        2D List with pids => idx
    """
    pid2imgs = {}
    for idx, (img, target, _) in enumerate(data_source.imgs):
        if target not in pid2imgs:
            pid2imgs[target] = [idx]
        else:
            pid2imgs[target].append(idx)
    return pid2imgs


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
        """Iterator over all images in dataset.

        Picks images accoding to Batch Hard.

        P: #pids in batch
        K: #images per pid
        
        Sorts PIDs randomly and iterates over each pid once.
        Fills batch by selecting K images for each PID. If expected size
        is reach, batch is yielded.
        """

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

class TripletBatchWithJunkSampler(object):
    """Sampler to create batches with P x K.
        
       Only returns indices.
        
    """
    JUNK_LABEL = -1
    def __init__(self, P, K, J, data_source, drop_last=True):
        self.P = P
        self.K = K
        self.J = J
        self.batch_size = self.P * self.K
        self.data_source = data_source
        self.drop_last = drop_last

        self.pid2imgs = create_pids2idxs(self.data_source)
        if not self.JUNK_LABEL in self.pid2imgs:
            raise RuntimeError("No Junk images found in datasource!")

        print("Found {} junk images in data source!".format(len(self.pid2imgs[self.JUNK_LABEL])))

    def __iter__(self):
        """Iterator over all images in dataset.

        Picks images accoding to Batch Hard.

        P: #pids in batch
        K: #images per pid
        
        Sorts PIDs randomly and iterates over each pid once.
        Fills batch by selecting K images for each PID. If expected size
        is reach, batch is yielded.
        """

        batch = []
        # -1 for junk id
        P_perm = np.random.permutation(len(self.pid2imgs) - 1)
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
                #now sample junk
                junk_images = self.pid2imgs[self.JUNK_LABEL]
                J_perm = np.random.permutation(len(junk_images))
                for j in range(self.J):
                    batch.append(junk_images[J_perm[j]])
                yield batch
                batch = []
        if len(batch) > 1 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.P
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.P
