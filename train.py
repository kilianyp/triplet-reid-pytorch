import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from csv_dataset import CsvDataset

# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True

from triplet_sampler import TripletBatchSampler
from trinet import choices as model_choices

from triplet_loss import choices as loss_choices
from triplet_loss import calc_cdist

import os
import h5py

from argparse import ArgumentParser

import logger as log
from logger import save_pytorch_model
import time

parser = ArgumentParser()

parser.add_argument('experiment',
        help="Name of the experiment")

parser.add_argument('--output_path', default="./experiments",
        help="Path where logging files are stored.")

parser.add_argument(
        '--csv_file', required=True,
        help="CSV file containing relative paths.")

parser.add_argument(
        '--data_dir', required=True,
        help="Root dir where the data is stored. This and the paths in the\
        csv file have to result in the correct file path."
        )

parser.add_argument(
        '--log_level', default=1, type=int,
        help="logging level"
        )

parser.add_argument(
        '--limit', default=None, type=int,
        help="The maximum number of (Images) that are loaded from the dataset")

parser.add_argument(
        '--P', default=18, type=int,
        help="Number of persons (pids) per batch.")

parser.add_argument(
        '--K', default=4, type=int,
        help="Number of images per pid.")

parser.add_argument(
        '--train_iterations', default=25000, type=int,
        help="Number of training iterations.")

parser.add_argument(
        '--embedding_dim', default=128, type=int,
        help="Size of the embedding vector."
        )
parser.add_argument(
        '--decay_start_iteration', default=15000, type=int,
        help="Learningg decay starts at this iteration")

parser.add_argument(
        '--checkpoint_frequency', default=1000, type=int,
        help="After how many iterations a new checkpoint is created.")

parser.add_argument('--margin', default='soft',
        help="What margin to use: a float value, 'soft' for "
        "soft-margin, or no margin if 'none'")

parser.add_argument('--alpha', default=1.0, type=float,
        help="Weight of the softmax loss.")

parser.add_argument('--temp', default=1.0,
        help="Temperature of BatchSoft")

parser.add_argument('--scale', default=1.125, type=float,
        help="Scaling of images before crop [scale * (image_height, image_width)]")

parser.add_argument('--image_height', default=256, type=int,
        help="Height of image that is fed to network.")

parser.add_argument('--image_width', default=128, type=int,
        help="Width of image that is fed to network.")

parser.add_argument('--lr', default=3e-4, type=float,
        help="Learning rate.")

parser.add_argument('--model', required=True, choices=model_choices)
parser.add_argument('--loss', required=True, choices=loss_choices)


def extract_csv_name(csv_file):
    filename = os.path.basename(csv_file)
    if filename.endswith(".csv"):
        return filename[:-4]
    else:
        return filename

def adjust_learning_rate(optimizer, t):
    global t0, t1, eps0
    if t <= t0:
        return eps0
    lr = eps0 * pow(0.001, (t - t0) / (t1 - t0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_alpha_rate(loss, t):
    a1 = 5000
    a2 = 10000
    if t <= a1:
        alpha = 1.0
    elif t < a2:
        alpha = 1.0 - ((t - a1) / (a2 - a1))
    else:
        alpha = 0.0

    loss.a = alpha
    return alpha

def topk(cdist, pids, k):
    """Calculates the top-k accuracy.
    
    Args:
        k: k smallest value
        
    """ 
    batch_size = cdist.size()[0]
    index = torch.topk(cdist, k+1, largest=False, dim=1)[1] #topk returns value and index
    index = index[:, 1:] # drop diagonal

    topk = torch.zeros(cdist.size()[0]).byte()
    topk = topk.cuda()
    topks = []
    for c in index.split(1, dim=1):
        c = c.squeeze() # c is batch_size x 1
        topk = topk | (pids.data == pids[c].data)
        acc = torch.sum(topk) / batch_size
        topks.append(acc)
    return topks

def var2num(x):
    return x.data.cpu().numpy()

args = parser.parse_args()

csv_file = os.path.expanduser(args.csv_file)
data_dir = os.path.expanduser(args.data_dir)

mod = __import__('triplet_loss')
loss = getattr(mod, args.loss)

# TODO allow arbitrary number of arguments


eps0 = args.lr
t0 = args.decay_start_iteration
t1 = args.train_iterations


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

H = args.image_height
W = args.image_width
scale = args.scale
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((int(H*scale), int(W*scale))),
        transforms.RandomCrop((H, W)),
        transforms.ToTensor(),
        normalize
    ])
dataset = CsvDataset(csv_file, data_dir, transform=transform, limit=args.limit)

print("Loaded %d images" % len(dataset))

dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=TripletBatchSampler(args.P, args.K, dataset),
        num_workers=4, pin_memory=True
        )

#also save num_labels
args.num_classes = dataset.num_labels
model_parameters = {"dim": args.embedding_dim, "num_classes": dataset.num_labels}
model_module = __import__('trinet')
model = getattr(model_module, args.model)
model = model(**model_parameters)

model = torch.nn.DataParallel(model).cuda()

loss_param = {"m": args.margin, "T": args.temp, "a": args.alpha}

loss_fn = loss(**loss_param)
optimizer = torch.optim.Adam(model.parameters(), lr=eps0, betas=(0.9, 0.999))

t = 1


training_name = args.experiment + "%s_%s-%s_%d-%d_%f_%d" % (
    extract_csv_name(csv_file), loss_fn.name,
    str(args.margin), args.P,
    args.K, eps0, args.train_iterations)

log = log.create_logger("h5", args.experiment, args.output_path, args.log_level)

#TODO restoring
# new experiment
log.save_args(args)



# save
# logging
    #emb_dataset = fout.create_dataset("emb", shape=(t1, batch_size,emb_dim), dtype=np.float32)
    #pids_dataset = fout.create_dataset("pids", shape=(t1, batch_size), dtype=np.int)
    #file_dataset = fout.create_dataset("file", shape=(t1, batch_size), dtype=h5py.special_dtype(vlen=str))
    #log_dataset = fout.create_dataset("log", shape=(t1, 6))


print("Starting training: %s" % training_name)
loss_data = {}
endpoints = {}
overall_time = time.time()
while t <= t1:
    for batch_id, (data, target, path) in enumerate(dataloader):
        start_time = time.time()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)
        endpoints = model(data, endpoints)
#        result.register_hook(lambda x: print("Gradient", x))
        loss_data["dist"] = calc_cdist(endpoints["emb"], endpoints["emb"])
        loss_data["pids"] = target
        loss_data["endpoints"] = endpoints
        #alpha = adjust_alpha_rate(loss_fn, t)
        losses = loss_fn(**loss_data)
        loss_mean = torch.mean(losses)
        lr = adjust_learning_rate(optimizer, t)
        topks = topk(loss_data["dist"], target, 5)
        min_loss = float(var2num(torch.min(losses)))
        max_loss =  float(var2num(torch.max(losses)))
        mean_loss = float(var2num(loss_mean))

        log.write("emb", var2num(endpoints["emb"]), dtype=np.float32)
        log.write("pids", var2num(target), dtype=np.int)
        log.write("file", path, dtype=h5py.special_dtype(vlen=str))
        log.write("log", [min_loss, mean_loss, max_loss, lr, topks[0], topks[4]], np.float32)
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()
        
        took = time.time() - start_time
        print("batch {} loss: {:.3f}|{:.3f}|{:.3f} lr: {:.6f} "
              "top1: {:.3f} top5: {:.3f} | took {:.3f}s".format(
            t, min_loss, mean_loss, max_loss, lr,
            topks[0], topks[4], took
            ))

        t += 1
        if t % 1000 == 0:
            save_pytorch_model(model, t)
        if t >= t1:
            break

        #if t % 10 == 0:
log.close()

print("Finished Training! Took: {:.3f}".format(time.time() - overall_time))
