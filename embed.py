import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from csv_dataset import CsvDataset
from trinet import mgn
from trinet import trinet
from trinet import model_parameters
from csv_dataset import make_dataset_mot
from csv_dataset import make_dataset_default
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


def write_to_h5(output_file, model, dataloader, dataset):
    print(len(dataloader), len(dataset))

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

from PIL import Image
class Hflip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        return (img, flipped)


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def restore_transform(args, augmentation):
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
            augmentation((H, W)),
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
    
    model.eval()

    for idx, (data, _, path) in enumerate(dataloader):
        data = Variable(data).cuda()
        # with cropping there is an additional dimension
        bs, ncrops, c, h, w = data.size()
        endpoints = model(data.view(-1, c, h, w), endpoints)
        #endpoints = model(data, endpoints)
        result = endpoints["emb"]
        #restore batch and crops dimension and use mean over all crops
        result = result.view(bs, ncrops, -1).mean(1)
        print("\rDone (%d/%d)" % (idx, len(dataloader)), flush=True, end='')
        yield  result.data.cpu().numpy()


def augment_function_builder(augmentation):
    def no_augmentation(img):
        """Returns an iterable to be compatible with cropping augmentations."""
        return (img, )
    if augmentation == "TenCrop":
        return lambda args: transforms.TenCrop(args)
    elif augmentation == "HorizontalFlipping":
        return lambda args: Hflip()
    else:
        return lambda args: no_augmentation

def run(csv_file, data_dir, model_file, batch_size, make_dataset_func, 
        augmentation, prefix=None, output_dir="embed"):

    augment_func = augment_function_builder(augmentation)

    experiment = os.path.realpath(model_file).split('/')[-2]
    model_name = os.path.basename(model_file)
    csv_name = extract_csv_name(csv_file)
    output_file = "%s_%s.h5" % (csv_name, model_name)
    if prefix is not None:
        output_file = "{}_{}".format(prefix, output_file)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, experiment)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(os.path.abspath(output_dir), output_file)

    if os.path.isfile(output_file):
        #TODO create numerated filename
        print("File %s already exists! Please choose a different name." % output_file)
        return output_file
    else:
        print("Creating file in %s" % output_file)

    args = load_args(model_file)
    transform_comp  = restore_transform(args, augment_func)

    dataset = CsvDataset(csv_file, data_dir, transform=transform_comp, make_dataset_func=make_dataset_func)

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size
            )
    model = restore_model(args, model_file)
    if model == None:
        if False:
            model.train()
            endpoints = {}
            endpoints["emb"] = None
            print("Starting Warmup...", end='', flush=True)
            for idx, (data, _, path) in enumerate(dataloader):
                if idx > 65:
                    break

                bs, ncrops, c, h, w = data.size()
                with torch.no_grad():
                    endpoints = model(data.view(-1, c, h, w), endpoints)
                #endpoints = model(data, endpoints)
            print(" Done!")

    return write_to_h5(output_file, model, dataloader, dataset)

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
           '--mot', action='store_true')

    parser.add_argument(
            '--model', required=True,
            help="Path to state dict of model."
            )
    parser.add_argument(
            '--prefix', required=False)
    parser.add_argument("--augmentation", default=None, choices=["TenCrop", "HorizontalFlipping"])
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    csv_file = os.path.expanduser(args.csv_file)
    data_dir = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model)
    if args.mot == True:
        make_dataset_func = make_dataset_mot
    else:
        make_dataset_func = make_dataset_default
    run(csv_file, data_dir, model_dir, args.batch_size, make_dataset_func, 
        args.augmentation, args.filename, args.output_dir, args.prefix)


