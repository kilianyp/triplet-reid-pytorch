import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from csv_dataset import CsvDataset

# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True

from triplet_sampler import TripletBatchSampler
from trinet import trinet
from triplet_loss import *

import os
import csv

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
        '--csv_file', required=True,
        help="CSV file containing relative paths.")

parser.add_argument(
        '--data_dir', required=True,
        help="Root dir where the data is stored. This and the paths in the\
        csv file have to result in the correct file path."
        )

parser.add_argument(
        '--log_dir', default=".",
        help="Training logs are stored in this directory."
        )

parser.add_argument(
        '--limit', default=None, type=int,
        help="The maximum number of (Images) that are loaded from the dataset")

parser.add_argument(
        '--P', default=18,
        help="Number of persons (pids) per batch.")

parser.add_argument(
        '--K', default=4,
        help="Number of images per pid.")

parser.add_argument(
        '--train_iterations', default=25000,
        help="Number of training iterations.")

parser.add_argument(
        '--decay_start_iteration', default=15000,
        help="Learningg decay starts at this iteration")

parser.add_argument(
        '--checkpoint_frequency', default=1000,
        help="After how many iterations a new checkpoint is created.")

parser.add_argument('--margin', default='soft',
        help="What margin to use: a float value, 'soft' for "
        "soft-margin, or no margin if 'none'")

parser.add_argument('--prefix', default="",
        help="Prefix of the training folder.")

parser.add_argument('--scale', default=1.125, type=float,
        help="Scaling of images before crop [scale * (image_height, image_width)]")

parser.add_argument('--image_height', default=256, type=int,
        help="Height of image that is fed to network.")

parser.add_argument('--image_width', default=128, type=int,
        help="Width of image that is fed to network.")

parser.add_argument('--lr', default=3e-4, type=float,
        help="Learning rate.")

parser.add_argument('--model')


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


def var2f(x):
    return float(x.data.cpu().numpy())


args = parser.parse_args()

csv_file = os.path.expanduser(args.csv_file)
data_dir = os.path.expanduser(args.data_dir)
log_dir = os.path.expanduser(args.log_dir)


loss_fn = BatchHard(args.margin)
model = trinet()
model = torch.nn.DataParallel(model).cuda()

eps0 = args.lr
# save
log_dir = os.path.join(log_dir, "training")
training_name = args.prefix + "%s_%s-%s_%d-%d_%f_%d" % (
    extract_csv_name(csv_file), loss_fn.name,
    str(args.margin), args.P,
    args.K, eps0, args.train_iterations)

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
print(args.limit)
dataset = CsvDataset(csv_file, data_dir, transform=transform, limit=args.limit)


dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=TripletBatchSampler(args.P, args.K, dataset))


optimizer = torch.optim.Adam(model.parameters(), lr=eps0, betas=(0.9, 0.999))

t0 = args.decay_start_iteration
t1 = args.train_iterations

t = 1
 
print("Starting training: %s" % training_name)
log_h = open(os.path.join(log_dir, "log.csv"), 'w')
log_writer = csv.writer(log_h)


while t <= t1:
    for batch_id, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)
        result = model(data)
#        result.register_hook(lambda x: print("Gradient", x))
        losses = loss_fn(result, target)
        loss_mean = torch.mean(losses)
        loss_f = var2f(loss_mean)
        lr = adjust_learning_rate(optimizer, t)
        print("batch {} loss: {:.3f}|{:.3f}|{:.3f} lr: {:.6f}".format(
            t,
            var2f(torch.min(losses)), var2f(torch.max(losses)), loss_f, lr))
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()
        log_writer.writerow([loss_f])
        t += 1
        if t % 1000 == 0:
            print("Iteration %d: Saved model" % t)
            torch.save(model.state_dict(), os.path.join(log_dir, "model_" + str(t)))

        #if t % 10 == 0:
log_h.close()


