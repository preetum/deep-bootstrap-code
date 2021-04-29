# import sys
# sys.path.append("/home/jupyter/pytorch-training/trainer")

from common.models.small import *
from common.models.vit import vit4, vit8

import torch
from torchvision.transforms import transforms
import torch.nn as nn
from common import load_state_dict

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def _mlp(widths=[128,128,128], num_classes=10, indim=32*32*3, batch_norm=True):
    layers = [Flatten()]
    for w in widths:
        layers.append(nn.Linear(indim, w, bias=True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(w))
        layers.append(nn.ReLU(inplace=True))
        indim = w
    layers.append(nn.Linear(indim, num_classes, bias=True))
    return nn.Sequential(*layers)

def mlp(pretrained_path : str = None, **kwargs):
    model = _mlp(**kwargs)
    if pretrained_path is None:
        return model
    else:
        load_state_dict(model, pretrained_path)
        model[-1].reset_parameters() # reset classification head
        return model

def rmlp(**kwargs):
    ''' raw mlp, no BatchNorm '''
    return mlp(batch_norm=False, **kwargs)

def mlp5(**kwargs):
    return mlp(widths=[128,128,128,128,128])

def smallCNN(width=64, in_channels=3, num_classes=10):
    c = width
    init_filters = 64
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(in_channels, init_filters, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(init_filters),
        nn.ReLU(inplace=True),

        # Layer 1
        nn.Conv2d(init_filters, c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2),

        Flatten(),
        nn.Linear(c*16*16, c, bias=False),
        nn.BatchNorm1d(c),
        nn.ReLU(inplace=True),
        nn.Linear(c, num_classes, bias=False)
    )

def mCNN(c=64, in_channels=3, num_classes=10):
    return mcnn(width=c, in_channels=in_channels, num_classes=num_classes)

def mcnn_nobn(**kwargs):
    return mcnn(batch_norm=False, **kwargs)

def mcnn(width=64, in_channels=3, num_classes=10, batch_norm=True):
    c = width
    norm = nn.BatchNorm2d if batch_norm else nn.Identity
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(in_channels, c, kernel_size=3, stride=1,
                  padding=1, bias=True),
        norm(c),
        nn.ReLU(inplace=True),

        # Layer 1
        nn.Conv2d(c, c*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        norm(c*2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        norm(c*4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        norm(c*8),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(2),

        # Layer 4
        # nn.MaxPool2d(4),
        nn.AdaptiveMaxPool2d(1),
        Flatten(),
        nn.Linear(c*8, num_classes, bias=True)
    )