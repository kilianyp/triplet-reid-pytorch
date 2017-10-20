import torch
from torch.utils.data import Dataset

import os
from PIL import Image


def pil_loader(path):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def extract_class(file_name):
    """Extracts the class according to the Market-1501 default naming.

        0001_c1s1_001051_00.jpg

        0001:   identity
        c1:     camera 1
        s1:     seqence1
        001051: 1051th frame (frame rate 25)
        00:     DPM bounding box id
    """

    target = file_name.split('_')[0]
    return int(target)

def make_dataset(dir, limit):
    dir = os.path.expanduser(dir)
    prev_target = -2
    limit_counter = 0
    target2idx = []
    imgs = []
    for idx, file_name in enumerate(sorted(os.listdir(dir))):
        if not file_name.endswith('.jpg'):
            continue
        target = extract_class(file_name)
        file_name = os.path.join(dir, file_name)
        if limit_counter >= limit:
            break
        
        if target != prev_target:
            limit_counter += 1
            prev_target = target
            target2idx.append([])
        target2idx[-1].append(idx)
        imgs.append((file_name, target))
    return imgs, target2idx

class MarketDataset(Dataset):
    """Market-1501 Dataset.

    Assumes unchanged folder names.

    Similar to ImageFolder.
    """

    test_dir = "bounding_box_test"
    train_dir = "bounding_box_train"
    dataset_dir = "Market-1501"

    def __init__(self, root_dir, transform=None, test=False, limit=None):
        
        self.root_dir = root_dir
        
        if not os.path.exists(self.root_dir):
            raise RuntimeError("Folder not found %s" % (self.root_dir))

        self.data_dir = os.path.join(self.root_dir, MarketDataset.dataset_dir)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please download from \
                https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing \
                and extract into Market-1501!"
            )
            
        self.limit = limit

        self.loader = pil_loader
        self.transform = transform

        if test:
            target_len = 750
            folder = MarketDataset.test_dir
        else:
            target_len = 751
            folder = MarketDataset.train_dir
            
        if limit is not None:
            target_len = limit
        else:
            limit = target_len

        self.data_dir = os.path.join(self.data_dir, folder)

        self.imgs, self.target2idx = make_dataset(self.data_dir, limit) 

        if len(self.target2idx) != target_len:
            print("Warning! The dataset seems to be corrupted. The expected number of classes \
                    could not be found! %d instead of %d." % (len(self.target2idx), target_len))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.data_dir, MarketDataset.test_dir)) and \
                os.path.exists(os.path.join(self.data_dir, MarketDataset.train_dir))

    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
