import torch.nn as nn
import math
import torch

__all__ = ['dconv', 'dconv_fc', 'dconv_l', 'sconv', 'sconv9', 'sconv33']

class Network(nn.Module):
    def __init__(self, cfg, kind='conv', cin=3, nclasses=10, imsize=32, ksize=3, base=32):
        super(Network, self).__init__()

        self.kind = kind
        layers, clayers = [], []
        self.fclayers, self.masks = [], []

        # creating convolutional layers or their locally/fully connected counterparts
        padding = math.floor(ksize / 2)
        for pair in cfg[0]:
            cout, outsize = base * \
                pair[0], math.floor(
                    (imsize + 2 * padding - ksize) / pair[1] + 1)
            layers += self.make_block(cin, cout, imsize=imsize, outsize=outsize, stride=pair[1],
                                      ksize=ksize, padding=padding, dropout=False, kind=self.kind)
            cin, imsize = cout, outsize
        self.features = nn.Sequential(*layers)

        # creating fully connected layers
        cin *= imsize ** 2
        for coeff in cfg[1]:
            clayers += self.make_block(cin, base * coeff, dropout=True)
            cin = base * coeff
        clayers += self.make_block(cin, nclasses, dropout=False, last=True)
        self.classifier = nn.Sequential(*clayers)

    def forward(self, x):

        if self.kind != 'conv':
            x = x.permute([0, 2, 3, 1])
            x = x.reshape(x.size(0), -1)
            if self.kind == 'local':
                self.apply_mask()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # creating a convolutional block or its locally/fully connected counterpart
    def make_block(self, cin, cout, imsize=1, outsize=1, stride=1, ksize=3, padding=1, dropout=False, kind='fc', last=False):
        if kind == 'conv':
            block = [nn.Conv2d(cin, cout, kernel_size=ksize, padding=padding,
                               stride=stride, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        else:
            self.fclayers.append(
                nn.Linear(cin * (imsize ** 2), cout * (outsize ** 2), bias=last))

            if kind == 'local':
                self.masks.append(make_conv_mask(
                    cin, cout, imsize=imsize, outsize=outsize, stride=stride, ksize=ksize, padding=padding))
                with torch.no_grad():
                    self.fclayers[-1].weight.data.mul_(self.masks[-1])
                    self.masks[-1] = self.masks[-1].cuda()
            if last:
                block = [self.fclayers[-1]]
            else:
                block = [
                    self.fclayers[-1], nn.BatchNorm1d(cout * (outsize ** 2)), nn.ReLU(inplace=True)]

        if dropout and not last:
            block.append(nn.Dropout())
        return block

    # apply the mask to the weights (used for local connections)
    def apply_mask(self):
        for mask, module in zip(self.masks, self.fclayers):
            h = module.weight.size(0)
            num = 10
            for i in range(num):
                first = round(i * h / num)
                last = round((i+1) * h / num)
                module.weight.data[first:last].mul_(mask[first:last].half())


# creating the mask for a locally connected layer
def make_conv_mask(cin, cout, imsize=1, outsize=1, stride=1, ksize=1, padding=0):

    mask = torch.BoolTensor(cout * (outsize ** 2),
                            cin * (imsize ** 2)).fill_(False)
    for i in range(-padding, imsize - ksize + padding + 1, stride):
        for j in range(-padding, imsize - ksize + padding + 1, stride):
            for kx in range(max(0, -i), min(ksize, imsize - i)):
                for ky in range(max(0, -j), min(ksize, imsize - j)):
                    x_in, y_in = i + kx, j + ky
                    x_out, y_out = math.floor(
                        (i + padding) / stride), math.floor((j + padding) / stride)
                    mask[cout * (x_out * outsize + y_out): cout * (x_out * outsize + y_out + 1),
                         cin * (x_in * imsize + y_in): cin * (x_in * imsize + y_in + 1)].fill_(True)

    return mask


# returns shallow network of the desired "kind" that can represent D-CONV with the given "base" cannels
def shallow_conv(kind='conv', nchannels=3, nclasses=10, imsize=32, ksize=9, base=256):
    cfg = ([(1, 2)], [24])
    return Network(cfg, kind=kind, cin=nchannels, nclasses=nclasses, imsize=imsize, ksize=ksize, base=base)

# returns deep network of the desired "kind" that can represent D-CONV with the given "base" cannels


def deep_conv(kind='conv', nchannels=3, nclasses=10, imsize=32, ksize=3, base=32):
    cfg = ([(1, 1), (2, 2), (2, 1), (4, 2),
            (4, 1), (8, 2), (8, 1), (16, 2)], [64])
    return Network(cfg, kind=kind, cin=nchannels, nclasses=nclasses, imsize=imsize, ksize=ksize, base=base)

##
## Specific instances
##

def dconv_base(kind='conv', width=32, num_classes=10):
    return deep_conv(kind=kind, base=width, nclasses=num_classes)

def dconv(**kwargs):
    return dconv_base(kind='conv', **kwargs)
def dconv_fc(**kwargs):
    return dconv_base(kind='fc', **kwargs)
def dconv_l(**kwargs):
    return dconv_base(kind='local', **kwargs)


## For ImageNet
def sconv(num_classes=2):
    return shallow_conv(kind='conv', nchannels=3, nclasses=num_classes, imsize=224, ksize=9, base=48)

def sconv9(num_classes=2):
    return shallow_conv(kind='conv', nchannels=3, nclasses=num_classes, imsize=224, ksize=9, base=48)
def sconv33(num_classes=2):
    return shallow_conv(kind='conv', nchannels=3, nclasses=num_classes, imsize=224, ksize=33, base=48)

if __name__ == '__main__':
    base = 32
    kinds = ['conv', 'local', 'fc']

    print('Shallow Architectures')
    for kind in kinds:
        print(
            f'Shallow-{kind} that can repsrent S-CONV with {base} base channels:')
        print(shallow_conv(kind=kind, base=base))

    print('Deep Architectures')
    for kind in kinds:
        print(f'Deep-{kind} that can repsrent S-CONV with {base} base channels:')
        print(deep_conv(kind=kind, base=base))