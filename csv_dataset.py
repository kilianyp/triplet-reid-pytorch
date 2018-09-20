import torch
from torch.utils.data import Dataset

import os
from PIL import Image

import csv
import warnings
import numpy as np

def pil_loader_with_crop(path, row):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            bbox = (row["left"], row["top"], row["width"], row["height"])
            bbox = tuple(map(float, bbox))
            left, top, width, height = bbox
            right = left + width
            lower = top + height
            bbox = (left, top, right, lower)
            img = img.crop(bbox)
            return img.convert('RGB')

def pil_loader(path, row):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def make_dataset_default(csv_file, data_dir, limit):
    """Reads in a csv file according to the scheme "target, path".
    Args:
        limit: Number of images that are read in.
    """
    header = ["pid", "path"]
    def default_name_fn(row):
        return row['path']
    return make_dataset_unamed(csv_file, data_dir, default_name_fn, header, limit)

def make_dataset_mot_old(csv_file, data_dir, limit):
    """Reads in a csv file according to the scheme of mot files.
    Args:
        limit: Number of images that are read in.
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for id, row in enumerate(reader):
            if limit is not None and id >= limit:
                break
            # TODO why has this changed
            #target = row[7]
            #file_name = row[8]
            target = row[1]
            file_name = row[-1]
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir)
                continue
            imgs.append([file_dir, target])
    return imgs

def mot_name_fn(row):
    return "{:06}.jpg".format(int(row["frame"]))

def make_dataset_named_mot(csv_file, data_dir, limit):
    return make_dataset_named(csv_file, data_dir, mot_name_fn, limit)


def parse(row, header):
    from collections import OrderedDict
    dic = OrderedDict()
    for idx, col in enumerate(header):
        dic[col] = row[idx]
    return dic


        
def make_dataset_mot(csv_file, data_dir, limit):

    header = ["frame", "pid", "left", "top", 
            "width", "height", "confidence"]
    return make_dataset_unamed(csv_file, data_dir, mot_name_fn, header, limit)

def make_dataset_unamed(csv_file, data_dir, image_name_fn, header, limit):
    """Reads in a csv file according to the scheme of mot files.
    Args:
        limit: Number of images that are read in.
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for id, row in enumerate(reader):
            if limit is not None and id >= limit:
                break
            row = parse(row, header)
            file_name = image_name_fn(row)
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir)
                continue
            target = row["pid"]
            # list so that targets can be rewritten later
            imgs.append([file_dir, target, row])
    return imgs, header

def make_dataset_named(csv_file, data_dir, image_name_fn, limit):
    """Reads in a csv file with named columns.
    Args:
        csv_file, data_dir where the image is stored
        limit: Number of images that are read in.
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for id, row in enumerate(reader):
            if limit is not None and id >= limit:
                break
            file_name = image_name_fn(row)
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir) 
                continue
            target = row["pid"]
            # list so that targets can be rewritten later
            imgs.append([file_dir, target, row])
    return imgs, reader.fieldnames

import time
class CsvDataset(Dataset):
    """Loads data from a csv file."""
    JUNK_LABEL = "-1"
    def __init__(self, csv_file, data_dir, loader_fn=pil_loader,
                 transform=None, limit=None, make_dataset_func=make_dataset_default, rewrite=True):
        """
        Args:
            csv_file: The path to the csv file.
            data_dir: The path where the data is stored relative to the paths 
                given in the csv file.
            crop: should be the image be extracted
            transform: Transformations that are executed on each image.
            rewrite (bool): Rewrite labels to numerical order.
        """
        self.data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(self.data_dir):
            raise RuntimeError("Data directory was not found %s" % (self.data_dir))

        self.csv_file = csv_file
        #if not os.path.isfile(self.csv_file):
        #    raise RuntimeError("CSV file was not found in %s." % self.csv_file)
            
        self.loader = loader_fn
        self.transform = transform

        self.imgs, self.header = make_dataset_func(self.csv_file, self.data_dir, limit)
        # because of path in csv, everything is converted to string
        labels = np.unique(np.asarray(self.imgs, dtype=str)[:, 1])
        if rewrite:
            label_dic = {}
            new_label = 0
            # rewrite pids starting from 0
            for label in labels:
                if label == self.JUNK_LABEL:
                    label_dic[label] = int(label)
                else:
                    label_dic[label] = new_label
                    new_label += 1
            for img in self.imgs:
                img[1] = label_dic[str(img[1])]
        self.num_labels = len(labels)
        print("Dataset with {} labels.".format(self.num_labels))

    def __getitem__(self, index):
        path, target, row = self.imgs[index]
        img = self.loader(path, row)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, row

    def __len__(self):
           return len(self.imgs)

