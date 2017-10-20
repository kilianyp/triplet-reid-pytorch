import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets.market import MarketDataset

from triplet_sampler import TripletBatchSampler
from trinet import trinet

import os

# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True

root_dir = "."
data_dir = "datasets"
market_dir = os.path.join(root_dir, data_dir)

# hyperparameters

pretrained = True,
P = 8   # Persons
K = 4    # images
batch_size = P * K
t0 = 15000
t1 = 25000

if pretrained:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    H = 256
    W = 128
    eps0 = 3e10-4
else:
    normalize = transforms.Normalize()
    H = 128
    W = 64
    eps0 = 3e10-3


train_transform = transforms.Compose([
        transforms.Scale(256),
#        transforms.RandomCrop((H, W)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ])

test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        normalize
    ])

train_dataset = MarketDataset(market_dir, transform=train_transform, test=False, limit=23)


train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=TripletBatchSampler(P, K, train_dataset))


model = trinet().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=eps0, betas=(0.9, 0.999))

def adjust_learning_rate(optimizer, t):
    if t <= t0:
        return
    lr = eps0 * 0.001((t - t0) / (t1 - t0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for x in range(1):
#while (True):
    for id, (data, target) in enumerate(train_loader):
        print("data shape:", data.shape)
        print("target shape:", target.shape)
        print(target)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        result = model(data)
        print(result.data.shape)
        silly_loss = torch.mean(result)
        silly_loss.backward()
        optimizer.step()
        


