import torch.nn as nn
import torch
import torch.nn.functional as f
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo

choices = ["trinet", "mgn", "softmax", "mgn_advanced", "stride_test", "trinetv2", "trinetv3"]
model_parameters = {"dim": None, "num_classes": None, "mgn_branches": None}

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
        self.dim = dim

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

class TrinetV2(ResNet):
    def __init__(self, block, layers, num_classes, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1) # 0 classes thows an error

        self.dim = 256 
        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.reduce_1x1 = nn.Linear(2048, 256, bias=False) # embedding for trinet
        self.soft_fc = nn.Linear(2048, num_classes) # for softmax
        self.batch_norm = nn.BatchNorm1d(2048)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = f.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.batch_norm(x)
        x = self.relu(x)
        soft_emb = self.soft_fc(x)
        triplet = self.reduce_1x1(x)
        endpoints["triplet"] = [triplet]
        emb = [triplet]
        endpoints["emb"] = torch.cat(emb, dim=1)
        endpoints["soft"] = [soft_emb]
        return endpoints

def trinetv2(**kwargs):
    model = TrinetV2(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model

class TrinetV3(ResNet):
    """
    Replace average pooling with conv layer.
    """
    def __init__(self, block, layers, num_classes, dim=128, **kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1) # 0 classes thows an error

        self.dim = 256 
        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.conv = nn.Conv2d(2048, 2048, (8,4))
        self.batch_norm = nn.BatchNorm1d(2048)
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()
        self.soft_fc = nn.Linear(2048, num_classes) # for softmax
        self.reduce_1x1 = nn.Linear(2048, 256, bias=False) # embedding for trinet
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.batch_norm(x)
        x = self.relu(x)
        soft_emb = self.soft_fc(x)
        triplet = self.reduce_1x1(x)
        endpoints["triplet"] = [triplet]
        emb = [triplet]
        endpoints["emb"] = torch.cat(emb, dim=1)
        endpoints["soft"] = [soft_emb]
        return endpoints

def trinetv3(**kwargs):
    model = TrinetV3(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()

    # filter out fully connected keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}

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

class MGNv2(ResNet):
    """With softmax but no fc after averagepooling.

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

        self.fc_soft = nn.Linear(2048, num_classes)
        self.dim = 2048

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
        endpoints["soft"] = [self.fc_soft(x)]
        endpoints["emb"] = x
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
    return model



class MGNBranch(nn.Module):
    def __init__(self, parts, b3_x, num_classes, dim, block, blocks):
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
            b3_x: Resnet layer3 after block 1
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
        else:
            # TODO stride 1 or 2
            self.final_conv = self._make_layer(block, 512, blocks, stride=1)

        self.g_batch_norm = nn.BatchNorm1d(2048)
        self.g_batch_norm.weight.data.fill_(1)
        self.g_batch_norm.bias.data.zero_()

        self.g_fc = nn.Linear(2048, num_classes) # for softmax
        self.g_1x1 = nn.Linear(2048, dim) # for triplet
        self.relu = nn.ReLU(inplace=True)
        self.layer3_x = b3_x

        # for the part branch

        if parts > 1:
            #if output[0] % parts != 0:
            #    raise RuntimeError("Output feature map height {} has to be dividable by parts (={})!"\
            #            .format(output, parts))
            #self.b_avg = nn.AvgPool2d((output[0]//parts, output[1]))
            self.b_1x1 = nn.Conv2d(512 * block.expansion, dim, 1)
            # TODO 1 or 2d batchnorm. I think it should not matter as one dimension is 1
            self.b_batch_norm = nn.BatchNorm2d(dim)
            self.b_batch_norm.weight.data.fill_(1)
            self.b_batch_norm.bias.data.zero_()
            # batch norm learns parameter to estimate during inference
            self.b_softmax = nn.ModuleList()
            for part in range(parts):
                self.b_softmax.append(nn.Linear(dim, num_classes)) # replace fc again with 1x1 conv
    
    def forward(self, x):
        # each branch returns one embedding and a number of softmaxe
        #print(x.shape)
        x = self.layer3_x(x)
        x = self.final_conv(x)
        output_shape = x.shape[-2:]
        g = f.avg_pool2d(x, output_shape) # functional
        g = g.view(g.size(0), -1)
        g = self.g_batch_norm(g)
        g = self.relu(g)
        softmax = [self.g_fc(g)]
        # This seems to be fine in parallel enviroments
        triplet = self.g_1x1(g)
        emb = [triplet]
        if self.parts == 1:
            return emb, [triplet], softmax
        
        # TODO does this return to cpu?
        if output_shape[0] % self.parts != 0:
            raise RuntimeError("Outputshape not dividable by parts")
        b_avg = f.avg_pool2d(x, (output_shape[0]//self.parts, output_shape[1]))
        b = self.b_1x1(b_avg)
        b = self.b_batch_norm(b)
        b = self.relu(b)
        for p in range(self.parts):
            b_part = b[:, :, p, :].contiguous().view(b.size(0), -1)
            b_softmax = self.b_softmax[p](b_part)
            softmax.append(b_softmax)
            # all the reduced features are concatenated together as the final feature 
            emb.append(b_part)
        
        return emb, [triplet], softmax


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

import copy
class MGNAdvanced(ResNet):
    """mgn_1N
    """

    @property
    def dim(self):
        """Warning errors in here are not shown in trace!
        https://github.com/encode/django-rest-framework/issues/2108"""
        dim = 0
        for branch in self.num_branches:
            if branch == 1:
                dim += self._dim
            else:
                dim += self._dim * (1 + branch) #global branch + softmax
        return dim

    def __init__(self, block, layers, mgn_branches, num_classes, dim=256, **kwargs):
        """Initialize MGN network.

        Args:
            block: Building block for resnet.
            layers: Parameter for layer building of resnet.
            branches (list): Branch configuration. List of parts.
            num_classes: Number of classes used for softmax layer.
        """
        super().__init__(block, layers, 1) # 0 classes thows an error
        if len(mgn_branches) == 0:
            raise RuntimeError("MGN needs at least one branch.")

        # only parts of layer 3 are in the backbone
        self.layer3_0 = self.layer3[0]
        layer3_x = self.layer3[1:]

        # explicitly delete layer 4 to avoid confusion when restoring.
        self.layer3 = None
        self.layer4 = None
        self.fc = None
        self.branches = nn.ModuleList()
        for branch in mgn_branches:
            print("Adding branch part-{}.".format(branch))
            b3_x = copy.deepcopy(layer3_x)
            self.branches.append(MGNBranch(branch, b3_x, num_classes,
                                           dim, block, layers[3]))
        self.num_branches = mgn_branches
        self._dim = dim
        print("embedding dim is {}".format(self.dim))

    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_0(x)
        emb = []
        triplet = []
        softmax = []
        for branch in self.branches:
            e, t, s = branch(x)
            emb.extend(e)
            triplet.extend(t)
            softmax.extend(s)
        
        # concatenate embedding
        emb = torch.cat(emb, dim=1)
        #print(emb.shape)
        endpoints["emb"] = emb
        endpoints["triplet"] = triplet
        endpoints["soft"] = softmax
        return endpoints

def mgn_advanced(**kwargs):
    """
    
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """
    print("Creating mgn advanced network.")
    model = MGNAdvanced(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()
    print(model_dict.keys())
    print(pretrained_dict.keys())

    layer3_0_dict = {} 
    for key, value in pretrained_dict.items():
        if key.startswith("layer3.0"):
            new_key = "layer3_0.{}".format(key[len("layer3.0."):])
            print(new_key)
            layer3_0_dict[new_key] = value
    
    # restore full layer 3, get keys from layer3.1
    layer3_dict = {k: v for k, v in pretrained_dict.items() 
                   if (not k.startswith("layer3.0") and k.startswith("layer3"))}

    for idx, parts in enumerate(model.branches):
        for key, value in layer3_dict.items():
            new_key = "branches.{}.layer3_x.{}".format(idx, key[len("layer3."):])
            print("{} => {}".format(key, new_key))
            pretrained_dict[new_key] = value
    
    # delay because layer_3_x otherwise sees those keys TODO clean
    pretrained_dict.update(layer3_0_dict)
    # restore branch final conv layer to layer4
    # only for layer with stride 2 (as original)
    layer4_dict = {k: v for k, v in pretrained_dict.items() if k.startswith("layer4")}
    for idx, parts in enumerate(model.num_branches):
        if parts != 1:
            # only global branch has stride 2 and can be restored
            # restoring with wrong stride has shown to have worse performance.
            continue
        for key, value in layer4_dict.items():
            new_key = "branches.{}.final_conv.{}".format(idx, key[len("layer4."):])
            pretrained_dict[new_key] = value


    # filter out fully connected keys
    # TODO sometimes we need to skip them, sometimes we do not?
    skips = ["fc", "layer4", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5"]
    # just for informational purpose
    for skip in skips:
        skipped_values = [k for k in pretrained_dict.keys() if k.startswith(skip)]
        print("Skipping: {}".format(skipped_values))
    
    #acutally skipping the values
    for skip in skips:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith(skip)}
    
    
    print("Restoring: {}".format(pretrained_dict.keys()))
 #   print(model)
    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


