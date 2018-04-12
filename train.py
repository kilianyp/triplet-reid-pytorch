import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from csv_dataset import CsvDataset

# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True

from triplet_sampler import TripletBatchSampler
from trinet import trinet

from triplet_loss import choices as loss_choices
from triplet_loss import calc_cdist

import os
import h5py
import json

from argparse import ArgumentParser

import logger as log

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
        '--log_dir', default="training",
        help="Training logs are stored in this directory."
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
        '--decay_start_iteration', default=15000, type=int,
        help="Learningg decay starts at this iteration")

parser.add_argument(
        '--checkpoint_frequency', default=1000, type=int,
        help="After how many iterations a new checkpoint is created.")

parser.add_argument('--margin', default='soft',
        help="What margin to use: a float value, 'soft' for "
        "soft-margin, or no margin if 'none'")

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

parser.add_argument('--model')
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
log_dir = os.path.expanduser(args.log_dir)

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
        batch_sampler=TripletBatchSampler(args.P, args.K, dataset))

model = trinet(dim=128, num_classes=dataset.num_labels)

model = torch.nn.DataParallel(model).cuda()

loss_param = {"m": args.margin, "T": args.temp}

loss_fn = loss(**loss_param)
optimizer = torch.optim.Adam(model.parameters(), lr=eps0, betas=(0.9, 0.999))

t = 1


training_name = args.experiment + "%s_%s-%s_%d-%d_%f_%d" % (
    extract_csv_name(csv_file), loss_fn.name,
    str(args.margin), args.P,
    args.K, eps0, args.train_iterations)

if args.log_level > 0:
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, training_name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
        print("Created new directory in %s" % log_dir)
    else:
        if os.listdir(log_dir):
            raise RuntimeError("Experiment seems to be have been already run in %s!"
                    "You can add a manual prefix with --prefix." % log_dir)

    args_file = os.path.join(log_dir, "args.json")

    with open(args_file, 'w') as file:
            json.dump(vars(args), file, ensure_ascii=False,
                      indent=2, sort_keys=True)

# save
# logging
log.create_logger(os.path.join(log_dir, "log.h5"), "h5", args.log_level)
    #emb_dataset = fout.create_dataset("emb", shape=(t1, batch_size,emb_dim), dtype=np.float32)
    #pids_dataset = fout.create_dataset("pids", shape=(t1, batch_size), dtype=np.int)
    #file_dataset = fout.create_dataset("file", shape=(t1, batch_size), dtype=h5py.special_dtype(vlen=str))
    #log_dataset = fout.create_dataset("log", shape=(t1, 6))


print("Starting training: %s" % training_name)
loss_data = {}
endpoints = {}
while t <= t1:
    for batch_id, (data, target, path) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)
        model(data, endpoints)
#        result.register_hook(lambda x: print("Gradient", x))
        loss_data["dist"] = calc_cdist(endpoints["emb"], endpoints["emb"])
        loss_data["pids"] = target
        loss_data["endpoints"] = endpoints
        losses = loss_fn(**loss_data)
        loss_mean = torch.mean(losses)
        lr = adjust_learning_rate(optimizer, t)
        topks = topk(loss_data["dist"], target, 5)
        min_loss = float(var2num(torch.min(losses)))
        max_loss =  float(var2num(torch.max(losses)))
        mean_loss = float(var2num(loss_mean))
        print("batch {} loss: {:.3f}|{:.3f}|{:.3f} lr: {:.6f} "
              "top1: {:.3f} top5: {:.3f}".format(
            t, min_loss, mean_loss, max_loss, lr,
            topks[0], topks[4]
            ))


        if args.log_level > 0:
            log.write("emb", var2num(endpoints["emb"]), dtype=np.float32)
            log.write("pids", var2num(target), dtype=np.int)
            log.write("file", path, dtype=h5py.special_dtype(vlen=str))
            log.write("log", [min_loss, mean_loss, max_loss, lr, topks[0], topks[4]], np.float32)
            #emb_dataset[t-1] = var2num(result)
            #pids_dataset[t-1] = var2num(target)
            #file_dataset[t-1] = path
            #log_dataset[t-1] = [min_loss, mean_loss, max_loss, lr, topks[0], topks[4]]
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()
        t += 1
        if t % 1000 == 0:
            print("Iteration %d: Saved model" % t)
            torch.save(model.state_dict(), os.path.join(log_dir, "model_" + str(t)))
        if t >= t1:
            break

        #if t % 10 == 0:
log.close()

