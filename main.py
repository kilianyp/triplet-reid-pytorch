import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets.market import MarketDataset

from triplet_sampler import TripletBatchSampler
from trinet import trinet
from triplet_loss import *

import os
import csv

# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True


root_dir = "."
data_dir = "datasets"
market_dir = os.path.join(root_dir, data_dir)

# hyperparameters
pretrained = True,
P = 18   # Persons
K = 4    # images
batch_size = P * K
t0 = 15000
t1 = 25000
margin = 1.0

loss_fn = BatchHard(margin)
scale = 1.125

if pretrained:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    H = 256
    W = 128
    eps0 = 3e-4
    model = trinet()
else:
    raise NotImplementedError
    normalize = transforms.Normalize()
    H = 128
    W = 64
    eps0 = 1e-3

# save

save_dir = os.path.join(root_dir, "training")
training_name = "%s-%s_%d-%d_%f_%d" % (loss_fn.name, str(margin), P, K, eps0, t1)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
log_dir = os.path.join(save_dir, training_name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
    print("Created new directory in %s" % log_dir)

train_transform = transforms.Compose([
    transforms.Scale((int(W*scale), int(H*scale))), # TODO old version?
        transforms.RandomCrop((H, W)),
        transforms.ToTensor(),
        normalize
    ])

test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        normalize
    ])

train_dataset = MarketDataset(market_dir, transform=train_transform, test=False)


train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=TripletBatchSampler(P, K, train_dataset))



model = torch.nn.DataParallel(model).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=eps0, betas=(0.9, 0.999))

def adjust_learning_rate(optimizer, t):
    if t <= t0:
        return
    lr = eps0 * 0.001((t - t0) / (t1 - t0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def var2f(x):
    return float(x.data.cpu().numpy())

t = 1
 
print("Starting training: %s" % training_name)
log_h = open(os.path.join(log_dir, "log.csv"), 'w')
log_writer = csv.writer(log_h)


while t <= t1:
    for batch_id,(data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)
        result = model(data)
#        result.register_hook(lambda x: print("Gradient", x))
        losses = loss_fn(result, target)
        loss_mean = torch.mean(losses)
        loss_f = var2f(loss_mean)
        print("batch {} loss: {:.3f}|{:.3f}|{:.3f}".format(
            t,
            var2f(torch.min(losses)), var2f(torch.max(losses)), loss_f))
        adjust_learning_rate(optimizer, t)
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

        


