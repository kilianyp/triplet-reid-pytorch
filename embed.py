import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from csv_dataset import CsvDataset
from trinet import trinet

import os
import h5py

from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument(
        '--filename', default=None)

dataset = "market"
data_config = {
        "market": (
            "~/Projects/cupsizes/data/market1501_query.csv",
            "~/Projects/triplet-reid-pytorch/datasets/Market-1501/")}

csv_file = os.path.expanduser(data_config[dataset][0])
data_dir = os.path.expanduser(data_config[dataset][1])


model = trinet()
state_dict_dir = "~/Projects/triplet-reid-pytorch/training/BatchHard-1.0_18-4_0.000300_25000/model_15000"
state_dict_dir = os.path.expanduser(state_dict_dir)
#restore trained model
state_dict = torch.load(state_dict_dir)
#print(state_dict.keys())

#print(model.state_dict().keys())
fresh_dict = {}
for key, value in state_dict.items():
    prefix = "module."
    if key.startswith(prefix):
        key = key[len(prefix):]
    fresh_dict[key] = value
model.load_state_dict(fresh_dict)

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


def extract_csv_name(csv_file):
    filename = os.path.basename(csv_file)
    if filename.endswith(".csv"):
        return filename[:-4]
    else:
        return filename

args = parser.parse_args()

if args.filename == None:
    output_file = "%s_embeddings.h5" % extract_csv_name(csv_file)
else:
    output_file = args.filename

if os.path.isfile(output_file):
    #TODO create numerated filename
    raise RuntimeError("File %s already exists! Please choose a different name." % output_file)

dataset = CsvDataset(csv_file, data_dir, transform=transforms)

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size
        )

model = torch.nn.DataParallel(model).cuda()
model.eval()
import gc
with h5py.File(output_file) as f_out:
    emb_dataset = f_out.create_dataset('emb', shape=(len(dataset), embedding_dim), dtype=np.float32)
    start_idx = 0
    for idx, (data, _) in enumerate(dataloader):
        data = data.cuda()
        data = Variable(data)
        result = model(data)
        end_idx = start_idx + len(result)
        emb_dataset[start_idx:end_idx] = result.data.cpu().numpy()
        start_idx = end_idx
        print("Done (%d/%d)" % (idx, len(dataloader)))
        gc.collect()
