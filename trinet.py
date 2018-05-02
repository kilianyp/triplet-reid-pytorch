import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo

choices = ["trinet", "mgn", "softmax", "mgn_advanced", "stride_test"]

class TriNet(ResNet):
    """TriNet implementation.

    Replaces the last layer of ResNet50 with two fully connected layers.

    First: 1024 units with batch normalization and ReLU
    Second: 128 units, final embedding.
    """
    
    def __init__(self, block, layers, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super(TriNet, self).__init__(block, layers, 1) # 0 classes thows an error

        batch_norm = nn.BatchNorm1d(1024)
        self.avgpool = nn.AvgPool2d((8,4))
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 1024),
            batch_norm,
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

        batch_norm.weight.data.fill_(1)
        batch_norm.bias.data.zero_()

    def forward(self, x, endpoints):
        x = super().forward(x)
        endpoints["emb"] = x
        return endpoints


def trinet(**kwargs):
    """Creates a TriNet network and loads the pretrained ResNet50 weights.
    
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """


    model = TriNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


class MGN(ResNet):
    """MGN implementaion

    Still based on trinet with two fully connected.
    Additional fully connected for softmax.
    -> avg pool -> fc 1024 -> batch_norm -> relu -> fc #num_classes
                                                 -> fc #dim

    This is not like the original mgn. Check mgn advanced.
    Returns two heads, one for the TripletLoss, the other for the Softmax Loss.
    """
    
    def __init__(self, block, layers, num_classes, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1) # 0 classes thows an error

        self.avgpool = nn.AvgPool2d((8,4))

        self.fc1 = nn.Linear(512 * block.expansion, 1024)
        self.batch_norm = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc_emb = nn.Linear(1024, dim)
        self.fc_soft = nn.Linear(1024, num_classes)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

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
        endpoints["soft"] = self.fc_soft(x)
        endpoints["emb"] = self.fc_emb(x)
        return endpoints

def mgn(**kwargs):
    """Creates a MGN network and loads the pretrained ResNet50 weights.
    
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """


    model = MGN(Bottleneck, [3, 4, 6, 3], **kwargs)

    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


class StrideTest(ResNet):
    """Stride test

    Final layer has only stride one.
    """
    
    def __init__(self, block, layers, num_classes, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1) # 0 classes thows an error

        #overwrite self.inplanes which is set by make_layer
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d((16, 8))
        self.fc1 = nn.Linear(512 * block.expansion, 1024)
        self.batch_norm = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc_emb = nn.Linear(1024, dim)
        self.fc_soft = nn.Linear(1024, num_classes)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

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
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("layer4.0")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model



class MGNBranch(nn.Module):
    def __init__(self, parts, num_classes, dim, block, blocks):
        super().__init__()
        """Creates a new branch. 

        input: layer3 of resnet.
        
        if global branch:
        conv -> avg pool -> 1x1 conv -> batch norm -> relu -> global_emb
                         -> fc 2048 -> softmax

        if part branch:
        conv -> avg pool global -> 1x1 conv -> batch norm -> relu -> global embed
                                -> fc 2048 -> softmax
             -> avg pool branch -> 1x1 conv -> batch norm -> relu -> fc 256 -> softmax
        Args:
            parts: Number of parts the image should be split into.
            num_classes: number of classes for the fc of softmax.
            dim: Reduction dimension, also used for triplet loss as embedding dim.
            downsample: Should the last layer4 downsample (global or part)
            block: Bulding block for resnet
            layers: number of layers for resnet. Layer4 has usually 3.
        """
        self.parts = parts
        # output is (H, W)
        # this is layer4 of resnet, the 5th convolutional layer
        if parts == 1: # global branch => downsample
            self.final_conv = self._make_layer(block, 512, blocks, stride=2)
            # TODO dynamic
            output = (8, 4)
        else:
            self.final_conv = self._make_layer(block, 512, blocks, stride=1)
            output = (16, 8)
        

        # always do global avg pooling 
        # each global and part branch after pooling  has a fc
        # global is 
        self.g_avg = nn.AvgPool2d(output)

        self.g_softmax = nn.Linear(512 * block.expansion, num_classes)
        
        self.g_1x1 = nn.Linear(512 * block.expansion, dim) #1x1 conv
        self.g_batch_norm = nn.BatchNorm1d(dim)
        self.g_batch_norm.weight.data.fill_(1)
        self.g_batch_norm.bias.data.zero_()

        self.ReLU = nn.ReLU()

        # for the part branch

        if parts > 1:
            if output[0] % parts != 0:
                raise RuntimeError("Output feature map height {} has to be dividable by parts (={})!"\
                        .format(output, parts))
            self.b_avg = nn.AvgPool2d((output[0]//parts, output[1]))
            print(self.b_avg)
            self.b_1x1 = nn.ModuleList()
            self.b_batch_norm = nn.ModuleList()
            # batch norm learns parameter to estimate during inference
            self.b_softmax = nn.ModuleList()
            for part in range(parts):
                self.b_1x1.append(nn.Linear(512 * block.expansion, dim, 1))
                b_batch_norm = nn.BatchNorm1d(dim)
                b_batch_norm.weight.data.fill_(1)
                b_batch_norm.bias.data.zero_()
                self.b_batch_norm.append(b_batch_norm)
                self.b_softmax.append(nn.Linear(dim, num_classes)) # replace fc again with 1x1 conv

    
    def forward(self, x):
        # each branch returns one embedding and a number of softmaxe
        #print(x.shape)
        x = self.final_conv(x)
        g = self.g_avg(x)
        # TODO this will most likely be overwritten in an parallel enviroment
        softmax = []
        g = g.view(g.size(0), -1)
        g_softmax = self.g_softmax(g)
        softmax.append(g_softmax)
        
        g = self.g_1x1(g)
        g = self.g_batch_norm(g)
        emb = [self.ReLU(g)]
        
        if self.parts == 1:
            return emb, softmax

        b_avg = self.b_avg(x)
        #print(b_avg.shape)
        for p in range(self.parts):
            b = b_avg[:, :, p, :].contiguous().view(b_avg.size(0), -1)
        #    print(b.shape)
            b = self.b_1x1[p](b)
            b = self.b_batch_norm[p](b)
            b = self.ReLU(b)
            b_softmax = self.b_softmax[p](b)
            softmax.append(b_softmax)
        
        return emb, softmax


    def _make_layer(self, block, planes, blocks, stride=1):
        """Copied from torchvision/models/resnet.py
        Adapted to always be follow after layer3
        """
        # layer3 has 256 * block.expansion output channels
        inplanes = 256 * block.expansion #here
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes)) #here

        return nn.Sequential(*layers)


class MGNAdvanced(ResNet):
    """mgn_1N
    """

    
    def __init__(self, block, layers, num_classes, dim=256, **kwargs):

        super().__init__(block, layers, 1) # 0 classes thows an error
        print(dim)
        self.part1 = MGNBranch(1, num_classes, dim, block, layers[3]) 
        self.part2 = MGNBranch(2, num_classes, dim, block, layers[3])
        #self.part3 = MGNBranch(4, num_classes, dim, block, layers[3])


    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        emb1, emb2, emb3 = [], [], []
        softmax1, softmax2, softmax3 = [], [], []

        emb1, softmax1 = self.part1(x)
        emb2, softmax2 = self.part2(x)
        #emb3, softmax3 = self.part3(x)

        emb = emb1 + emb2 + emb3
        softmax = softmax1 + softmax2 + softmax3
        emb = torch.cat(emb, dim=1)
        #print(emb.shape)
        endpoints["emb"] = emb
        endpoints["soft"] = softmax
        return endpoints

def mgn_advanced(**kwargs):
    """
    
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """


    model = MGNAdvanced(Bottleneck, [3, 4, 6, 3], **kwargs)

    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    #for key, value in pretrained_dict.items():
    #    print(key)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


