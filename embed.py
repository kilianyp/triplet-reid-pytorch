import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from csv_dataset import CsvDataset
from trinet import mgn
from trinet import trinet
from trinet import model_parameters

import os
import h5py
import sys
from argparse import ArgumentParser
import json
import logger as log

def clean_dict(dic):
    """Removes module from keys. For some reason those are added when saving."""
    fresh_dict = {}
    for key, value in dic.items():
        prefix = "module."
        if key.startswith(prefix):
            key = key[len(prefix):]
        fresh_dict[key] = value
    return fresh_dict


def extract_csv_name(csv_file):
    filename = os.path.basename(csv_file)
    if filename.endswith(".csv"):
        return filename[:-4]
    else:
        return filename


def write_to_h5(csv_file, data_dir, model_file, batch_size, filename=None, output_dir="embed"):

    experiment = os.path.realpath(model_file).split('/')[-2]
    if filename == None:
        model_name = os.path.basename(model_file)
        csv_name = extract_csv_name(csv_file)
        output_file = "%s_%s.h5" % (csv_name, model_name)
    else:
        output_file = filename

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, experiment)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    output_file = os.path.join(os.path.abspath(output_dir), output_file)
    print(output_file)

    if os.path.isfile(output_file):
        #TODO create numerated filename
        print("File %s already exists! Please choose a different name." % output_file)
        return output_file
    else:
        print("Creating file in %s" % output_file)

    args = load_args(model_file)
    
    transform_comp  = restore_transform(args)
    model           = restore_model(args, model_file)

    dataset = CsvDataset(csv_file, data_dir, transform=transform_comp)

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size
            )
    
    print("Model dimension is {}".format(model.module.dim))
    with h5py.File(output_file) as f_out:
        # Dataparallel class!
        emb_dataset = f_out.create_dataset('emb', shape=(len(dataset), model.module.dim), dtype=np.float32)
        start_idx = 0
        for result in create_embeddings(dataloader, model):
            end_idx = start_idx + len(result)
            emb_dataset[start_idx:end_idx] = result
            start_idx = end_idx
    return output_file

def get_args_path(model_path):
    model_path = os.path.dirname(model_path)
    return os.path.join(model_path, "args.json")


class InferenceModel(object):
    def __init__(self, model_path, transform, cuda=True):
        self.cuda = cuda

        self.transform = transform
        args = load_args(model_path)
        model = restore_model(args, model_path)
        if self.cuda:
            model = model.cuda()
        
        self.model = model
        self.endpoints = {}

    def __call__(self, data):
        """Forward pass on an image.
        Args:
            data: A PIL image
        """

        data = Variable(self.transform(data))
        print(data.size())
        if self.cuda:
            data = data.cuda()
        self.endpoints = self.model(data, self.endpoints)
        result = self.endpoints["emb"]
        # mean over crops
        # TODO this depends on the data augmentation
        result = result.mean(0)
        return result


def restore_transform(args):
    # TODO unify with training routine
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    H = args["image_height"]
    W = args["image_width"]
    scale = args["scale"]

    print(args)
    to_tensor = transforms.ToTensor()

    def to_normalized_tensor(crop):
        return normalize(to_tensor(crop))

    transform_comp = transforms.Compose([
            transforms.Resize((int(H*scale), int(W*scale))),
            transforms.TenCrop((H, W)),
            transforms.Lambda(lambda crops: torch.stack([to_normalized_tensor(crop) for crop in crops]))
        ])
    return transform_comp

def restore_model(args, model_path):
       
    model_parameters.update(args)
    model_module = __import__('trinet')
    model = getattr(model_module, args["model"])
    model = model(**model_parameters)
    #restore trained model
    state_dict = torch.load(model_path)
    state_dict = clean_dict(state_dict)
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    return model

def load_args(model_path):
    args_path = get_args_path(model_path)

    with open(args_path, 'r') as f_handle:
        return json.load(f_handle)

def create_embeddings(dataloader, model):
    """Create ten fold embeddings."""

    endpoints = {}
    # this is important, otherwise there might be a race condition
    # which gpu sets emb first => will lead to batch contains only values from one gpu twice
    endpoints["emb"] = None
    for idx, (data, _, path) in enumerate(dataloader):
        data = Variable(data).cuda()
        # with cropping there is an additional dimension
        bs, ncrops, c, h, w = data.size()
        endpoints = model(data.view(-1, c, h, w), endpoints)
        result = endpoints["emb"]
        #restore batch and crops dimension and use mean over all crops
        result = result.view(bs, ncrops, -1).mean(1)
        print("\rDone (%d/%d)" % (idx, len(dataloader)), flush=True, end='')
        yield  result.data.cpu().numpy()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
            '--output_dir', default="embed",
            help="Output directory for embedding hd5 file."
            )
    parser.add_argument(
            '--filename', default=None, 
            help="Output filename")

    parser.add_argument(
            '--csv_file', required=True,
            help="CSV file containing relative paths.")

    parser.add_argument(
            '--data_dir', required=True,
            help="Root dir where the data is stored. This and the paths in the\
            csv file have to result in the correct file path."
            )

    parser.add_argument(
            '--model', required=True,
            help="Path to state dict of model."
            )
    args = parser.parse_args()

    csv_file = os.path.expanduser(args.csv_file)
    data_dir = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model)
    write_to_h5(csv_file, data_dir, model_dir, 6, args.filename, args.output_dir)

