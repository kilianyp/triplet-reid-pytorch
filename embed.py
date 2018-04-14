import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from csv_dataset import CsvDataset
from trinet import mgn

import os
import h5py
import sys

from argparse import ArgumentParser


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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

H = 256
W = 128
scale = 1.125

embedding_dim = 128
batch_size = 30 

transforms = transforms.Compose([
        transforms.Resize((int(H*scale), int(W*scale))),
#         transforms.TenCrop((H, W)),
        transforms.ToTensor(),
        normalize
      ])


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

if __name__ == "__main__":
    args = parser.parse_args()

    csv_file = os.path.expanduser(args.csv_file)
    data_dir = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model)

    model_path = os.path.realpath(args.model).split('/')[-2]
    if args.filename == None:
        model_name = os.path.basename(args.model)
        csv_name = extract_csv_name(csv_file)
        output_file = "%s_%s_embeddings.h5" % (csv_name, model_name)
    else:
        output_file = args.filename

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, model_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    output_file = os.path.join(os.path.abspath(output_dir), output_file)
    print(output_file)

    if os.path.isfile(output_file):
        #TODO create numerated filename
        print(output_file, file=sys.stderr)
        print("File %s already exists! Please choose a different name." % output_file)
        sys.exit(0)
    else:
        print("Creating file in %s" % output_file)

    dataset = CsvDataset(csv_file, data_dir, transform=transforms)

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size
            )

    model = mgn(dim=128, num_classes=751)
    #restore trained model
    state_dict = torch.load(model_dir)
    state_dict = clean_dict(state_dict)
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()
#    model = model.cuda()
    model.eval()

    endpoints = {}
    import gc
    with h5py.File(output_file) as f_out:
        emb_dataset = f_out.create_dataset('emb', shape=(len(dataset), embedding_dim), dtype=np.float32)
        start_idx = 0
        for idx, (data, _, _) in enumerate(dataloader):
            data = data
            data = Variable(data)
            model(data, endpoints)
            result = endpoints["emb"]
            end_idx = start_idx + len(result)
            emb_dataset[start_idx:end_idx] = result.data.cpu().numpy()
            start_idx = end_idx
            print("Done (%d/%d)" % (idx, len(dataloader)))
            gc.collect()
    print(output_file, file=sys.stderr)
