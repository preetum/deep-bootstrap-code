from google.cloud import storage
import torch
import torch.backends.cudnn as cudnn
import subprocess
import os
import time
import shutil
from datetime import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from common.datasets import load_cifar, TransformingTensorDataset, get_cifar_data_aug, load_cifar500
import common.models32 as models


def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
    if optimizer_name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'nesterov_sgd':
        return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler(args, scheduler_name, optimizer, num_epochs, batches_per_epoch, **kwargs):
    if args.opt == 'adam':
        # constant LR sched
        print("Using adam with const LR.")
        return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
    if args.sched is not None:
        sched = list(map(int, args.sched.split(',')))
        print("Using step-wise LR schedule:", sched)
        return lr_scheduler.MultiStepLR(optimizer, milestones=np.cumsum(sched[:-1]), gamma=0.1)

    if scheduler_name == 'const':
        return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
    elif scheduler_name == '3step':
        return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
    elif scheduler_name == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)
    elif scheduler_name == 'onecycle':
        return lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=batches_per_epoch, epochs=num_epochs)

def get_rrand_model(args, nclasses=10):
    ''' Returns random model with random width and num_cells. '''
    from nas import get_random_model
    import sys
    import random
    if args.aseed is None:
        args.aseed = random.randrange(sys.maxsize)

    r = random.Random(args.aseed+1)
    width = r.choice([16, 64])
    num_cells = r.choice([1, 5])
    model, genotype = get_random_model(seed=args.aseed, num_classes=nclasses, width=width, num_cells=num_cells, max_nodes=4)
    print('genotype: ', genotype.tostr())
    print(f'width={width}, num_cells={num_cells}')
    args.genotype = genotype.tostr()
    args.width = width
    args.num_cells = num_cells
    return model

def get_model32(args, model_name, nchannels=3, nclasses=10, half=False, pretrained_path=None):
    ngpus = torch.cuda.device_count()
    print("=> creating model '{}'".format(model_name))
    if model_name.startswith('mlp'): # eg: mlp[512,512,512]
        widths = eval(model_name[3:])
        model = models.mlp(widths=widths, num_classes=nclasses, pretrained_path=pretrained_path)
    elif model_name == 'rand5':
        from nas import get_random_model
        num_cells = 5
        seed = args.aseed if 'aseed' in args else None
        model, genotype = get_random_model(seed=seed, num_classes=nclasses, width=64, num_cells=num_cells, max_nodes=4)
        print('genotype: ', genotype.tostr())
        args.genotype = genotype.tostr()
    elif model_name == 'rrand':
        model = get_rrand_model(args, nclasses=nclasses)
    elif model_name.startswith('vit'):
        model = models.__dict__[model_name](num_classes=nclasses, pretrained_path=pretrained_path)
    else:
        if args.width is not None:
            model = models.__dict__[model_name](num_classes=nclasses, width=args.width)
        else:
            model = models.__dict__[model_name](num_classes=nclasses)

    args.nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num model parameters:", args.nparams)
    if half:
        print('Using half precision except in Batch Normalization!')
        model = model.half()
        for module in model.modules():
            if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
                module.float()
    return model




class AverageMeter(object):
    def __init__(self, name=None):
        self.reset()
        self.name=name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'[{self.name}]:{self.avg}'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res