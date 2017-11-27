import torch
from torch.utils.data import Dataset

import os
from PIL import Image

import csv
import warnings

def pil_loader(path):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_dataset(csv_file, data_dir, limit):
    """Reads in a csv file according to the scheme "target, path".
    Args:
        limit: Number of images that are read in.
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for id, row in enumerate(reader):
            if limit is not None and id >= limit:
                break
            target = row[0]
            file_name = row[1]
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir)
                continue
            imgs.append((file_dir, int(target)))
            
    return imgs

class CsvDataset(Dataset):
    """Loads data from a csv file."""

    def __init__(self, csv_file, data_dir, transform=None, limit=None):
        """
        Args:
            csv_file: The path to the csv file.
            data_dir: The path where the data is stored relative to the paths 
                given in the csv file.
            transform: Transformations that are executed on each image.
        """

        self.data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(self.data_dir):
            raise RuntimeError("Data directory was not found %s" % (self.data_dir))

        self.csv_file = csv_file
        #if not os.path.isfile(self.csv_file):
        #    raise RuntimeError("CSV file was not found in %s." % self.csv_file)
            

        self.loader = pil_loader
        self.transform = transform

        self.imgs = make_dataset(self.csv_file, self.data_dir, limit)

    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, path

    def __len__(self):
        return len(self.imgs)
