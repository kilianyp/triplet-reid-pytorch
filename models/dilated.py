import torch.nn as nn
import torch
import torch.nn.functional as f
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo

model_parameters = {"dim": None, "num_classes": None, "mgn_branches": None}

class DilatedBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1, dilation=1):
        super(DilatedBottleneck, self).__init__(inplanes, planes, stride=1, downsample=downsample)
        print("Dilation before conv1: {}".format(dilation))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        # This uses stride, after this layer all conv layers need to be dilated.
        print("Dilation before conv2: {}".format(dilation))
        
        # for layer 4 this is normally stride in the first block 2
        # after this layer all convs need to be dilated
        # also changed padding
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, bias=False, dilation=dilation)
        dilation = dilation * rate
        print("Dilation before conv3: {}".format(dilation))
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)

class StrideTest(ResNet):
    """Stride test

    Final layer has only stride one.
    """
    
    def _make_dilated_layer4(self, block, planes, blocks):
        """Copied from torchvision/models/resnet.py
        Adapted to always be follow after layer3
        """

        # layer3 has 256 * block.expansion output channels
        inplanes = 256 * block.expansion #here
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        print(downsample)
        layers = []
        # first layer stride changes
        # after this all have to be dilated
        layers.append(block(inplanes, planes, stride=1, downsample=downsample,
                      rate=2, dilation=1))
        for i in range(1, blocks):
            #block default is stride=1
            layers.append(block(planes * block.expansion, planes, dilation=2)) #here

        return nn.Sequential(*layers)

    def __init__(self, block, layers, num_classes, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1) # 0 classes thows an error

        #overwrite self.inplanes which is set by make_layer
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_dilated_layer4(DilatedBottleneck, 512, layers[3])
        self.avgpool = nn.AvgPool2d((16, 8))
        self.fc1 = nn.Linear(512 * block.expansion, 1024)
        self.batch_norm = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc_emb = nn.Linear(1024, dim)
        self.fc_soft = nn.Linear(1024, num_classes)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()
        self.dim = dim

    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        endpoints["soft"] = [self.fc_soft(x)]
        endpoints["emb"] = self.fc_emb(x)
        return endpoints

def stride_test(**kwargs):


    model = StrideTest(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                       if  not (k.startswith("layer4") and "downsample" in k)}

    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("layer4.0")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    endpoints = {}
    endpoints["emb"] = None
    return model


def make_dilated_layer4(block, planes, blocks):
    """Copied from torchvision/models/resnet.py
    Adapted to always be follow after layer3
    """

    # layer3 has 256 * block.expansion output channels
    inplanes = 256 * block.expansion #here
    downsample = None
    if inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    print(downsample)
    layers = []
    # first layer stride changes
    # after this all have to be dilated
    layers.append(block(inplanes, planes, stride=1, downsample=downsample,
                    rate=2, dilation=1))
    for i in range(1, blocks):
        #block default is stride=1
        layers.append(block(planes * block.expansion, planes, dilation=2)) #here

    return nn.Sequential(*layers)
