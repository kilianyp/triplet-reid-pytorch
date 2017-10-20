import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo

class TriNet(ResNet):
    """TriNet implementation.

    Replaces the last layer of ResNet50 with two fully connected layers.

    First: 1024 units with batch normalization and ReLU
    Second: 128 units, final embedding.
    """
    
    def __init__(self, block, layers):
        """Initializes original ResNet and overwrites fully connected layer."""

        super(TriNet, self).__init__(block, layers, 1) # 0 classes thows an error

        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )


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







