import torch
from torchvision.transforms import transforms
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def mlp(widths=[128,128,128], num_classes=10, indim=28*28, batch_norm=False):
    layers = [Flatten()]
    for w in widths:
        layers.append(nn.Linear(indim, w, bias=True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(w))
        layers.append(nn.ReLU(inplace=True))
        indim = w
        
    layers.append(nn.Linear(indim, num_classes, bias=True))
    return nn.Sequential(*layers)

def cnn28(in_channels=1):
    k = 64
    layers = [
        # Layer 0
        nn.Conv2d(in_channels, k, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.ReLU(inplace=True),

        # Layer 1
        nn.Conv2d(k, k*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2),

        # Layer 2
        nn.Conv2d(k*2, k*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2),

        # Layer 3
        nn.Conv2d(k*2, k*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(k*2, 10)
    ]
    return nn.Sequential(*layers)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y

def leNet5():
    return LeNet5()