import os
import wandb

import argparse
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler

from .utils import AverageMeter
from common.datasets import load_cifar, TransformingTensorDataset, get_cifar_data_aug
from common.datasets import load_cifar550, load_svhn_all, load_svhn, load_cifar5m
import common.models32 as models
from .utils import get_model32, get_optimizer, get_scheduler

from common.logging import VanillaLogger

parser = argparse.ArgumentParser(description='vanilla training')
parser.add_argument('--proj', default='test-soft', type=str, help='project name')
parser.add_argument('--dataset', default='cifar5m', type=str)
parser.add_argument('--nsamps', default=50000, type=int, help='num. train samples')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--k', default=64, type=int, help="log every k batches", dest='k')
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')

# parser.add_argument('--arch', metavar='ARCH', default='mlp[16384,16384,512]')
parser.add_argument('--arch', metavar='ARCH', default='preresnet18')
parser.add_argument('--pretrained', type=str, default=None, help='gcs-path to pretrained model state dict (optional)')
parser.add_argument('--width', default=None, type=int, help="architecture width parameter (optional)")
parser.add_argument('--loss', default='xent', choices=['xent', 'mse'], type=str)

parser.add_argument('--opt', default="sgd", type=str)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default="cosine", type=str, help='lr scheduler')
parser.add_argument('--sched', default=None, type=str)
parser.add_argument('--aug', default=0, type=int, help='data-aug (0: none, 1: flips, 2: all)')

parser.add_argument('--epochs', default=100, type=int)
# for keeping the same LR sched across different samp sizes.
parser.add_argument('--nbatches', default=None, type=int, help='Total num. batches to train for. If specified, overrides EPOCHS.')
parser.add_argument('--batches_per_lr_step', default=390, type=int)

parser.add_argument('--noise', default=0.0, type=float, help='label noise probability (train & test).')

parser.add_argument('--momentum', default=0.0, type=float, help='momentum (0 or 0.9)')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')

parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--half', default=False, action='store_true', help='training with half precision')
parser.add_argument('--fast', default=False, action='store_true', help='do not log more frequently in early stages')
parser.add_argument('--earlystop', default=False, action='store_true', help='stop when train loss < 0.01')

parser.add_argument('--aseed', default=None, type=int, help="architecture-seed for rand NAS archs (optional)")
parser.add_argument('--comment', default=None)

args = parser.parse_args()


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i



def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    if args.half: images = images.half()
    return images, target

def mse_loss(output, y):
    y_true = F.one_hot(y, 10).float()
    return (output - y_true).pow(2).sum(-1).mean()

def predict(loader, model):
    # switch to evaluate mode
    model.eval()
    n = 0
    predsAll = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = cuda_transfer(images, target)
            output = model(images)

            preds = output.argmax(1).long().cpu()
            predsAll.append(preds)

    preds = torch.cat(predsAll)
    return preds


def test_all(loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    aloss = AverageMeter('Loss')
    aerr = AverageMeter('Error')
    asoft = AverageMeter('SoftError')
    mets = [aloss, aerr, asoft]

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            bs = len(images)
            images, target = cuda_transfer(images, target)
            output = model(images)
            loss = criterion(output, target)

            err = (output.argmax(1) != target).float().mean().item()
            p = F.softmax(output, dim=1) # [bs x 10] : softmax probabilties
            p_corr = p.gather(1, target.unsqueeze(1)).squeeze() # [bs]: prob on the correct label
            soft = (1-p_corr).mean().item()


            aloss.update(loss.item(), bs)
            aerr.update(err, bs)
            asoft.update(soft, bs)

    results = {m.name : m.avg for m in mets}
    return results



def get_dataset(dataset):
    '''
        Returns dataset and pre-transform (to process dataset into [-1, 1] torch tensor)
    '''

    noop = transforms.Compose([])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    uint_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize]) # numpy unit8 --> [-1, 1] tensor

    if dataset == 'cifar10':
        return load_cifar(), noop
    if dataset == 'cifar550':
        return load_cifar550(), noop
    if dataset == 'cifar5m':
        return load_cifar5m(), uint_transform

    
def add_noise(Y, p: float):
    ''' Adds noise to Y, s.t. the label is wrong w.p. p '''
    num_classes = torch.max(Y).item()+1
    print('num classes: ', num_classes)
    noise_dist = torch.distributions.categorical.Categorical(
        probs=torch.tensor([1.0 - p] + [p / (num_classes-1)] * (num_classes-1)))
    return (Y + noise_dist.sample(Y.shape)) % num_classes


def get_data_aug(aug : int):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    unnormalize = transforms.Compose([
        transforms.Normalize((0, 0, 0), (2, 2, 2)),
        transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
    ])

    if aug == 0:
        print('data-aug: NONE')
        return transforms.Compose([])
    elif aug == 1:
        print('data-aug: flips only')
        return transforms.Compose(
            [unnormalize,
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    elif aug == 2:
        print('data-aug: full')
        return transforms.Compose(
            [unnormalize,
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    elif aug == 3:
        print('data-aug: full (reflect-crop)')
        return transforms.Compose(
            [unnormalize,
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])


def make_loader(x, y, transform=None, batch_size=256):
    dataset = TransformingTensorDataset(x, y, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
    return loader

def main():
    ## argparsing hacks
    if args.sched is not None:
        sched = list(map(int, args.sched.split(',')))
        args.epochs = sum(sched)
        args.scheduler = 'steps'
    if args.pretrained == 'None':
        args.pretrained = None # hack for caliban

    wandb.init(project=args.proj)
    cudnn.benchmark = True

    #load the model
    model = get_model32(args, args.arch, half=args.half, nclasses=10, pretrained_path=args.pretrained)
    # model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    # init logging
    logger = VanillaLogger(args, wandb, hash=True)

    print('Loading datasets...')
    (X_tr, Y_tr, X_te, Y_te), preproc = get_dataset(args.dataset)

    # subsample
    if not args.iid:
        I = np.random.permutation(len(X_tr))[:args.nsamps]
        X_tr, Y_tr = X_tr[I], Y_tr[I]

    # Add noise (optionally)
    Y_tr = add_noise(Y_tr, args.noise)
    Y_te = add_noise(Y_te, args.noise)

    tr_set = TransformingTensorDataset(X_tr, Y_tr, transform=transforms.Compose([preproc, get_data_aug(args.aug)]))
    val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,
            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True) # drop the last batch if it's incomplete (< batch size)
    te_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
            shuffle=False, num_workers=args.workers, pin_memory=True)

    cifar_test = make_loader(*(load_cifar()[2:])) # original cifar-10 test set
    print('Done loading.')
    

    # batches / lr computations
    batches_per_epoch = int(np.floor(args.nsamps / args.batchsize))
    if args.nbatches is None:
        # set nbatches from EPOCHS
        args.nbatches = int((args.nsamps / args.batchsize) * args.epochs)
        args.batches_per_lr_step = batches_per_epoch
    num_lr_steps = (args.nbatches) // args.batches_per_lr_step # = epochs (unless --epochs is overridden)
    print(f'Num. total train batches: {args.nbatches}')


    # define loss function (criterion), optimizer and scheduler
    criterion = nn.CrossEntropyLoss().cuda() if args.loss == 'xent' else mse_loss
    optimizer = get_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args, args.scheduler, optimizer, num_epochs=num_lr_steps, batches_per_epoch=args.batches_per_lr_step)



    n_tot = 0
    for i, (images, target) in enumerate(recycle(tr_loader)):
        model.train()
        images, target = cuda_transfer(images, target)
        output = model(images)
        loss = criterion(output, target)

        n_tot += len(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## logging
        lr = optimizer.param_groups[0]['lr']

        # if i % args.k == 0: # first 512 batches, and every kth batch after that
        if i % args.k == 0 or (not args.fast and  ( \
            # (i < 128) or \
            # (i < 512 and i % 2 == 0) or \
            (i < 1024 and i % 4 == 0) or \
            (i < 2048 and i % 8 == 0))):
            ''' Every k batches (and more frequently in early stages): log train/test errors. '''

            d = {'batch_num': i,
                'lr': lr,
                'n' : n_tot}

            test_m = test_all(te_loader, model, criterion)
            testcf_m = test_all(cifar_test, model, criterion)
            d.update({ f'Test {k}' : v for k, v in test_m.items()})
            d.update({ f'CF10 {k}' : v for k, v in testcf_m.items()})

            if not args.iid:
                train_m = test_all(tr_loader, model, criterion)
                d.update({ f'Train {k}' : v for k, v in train_m.items()})

                print(f'Batch {i}.\t lr: {lr:.3f}\t Train Loss: {d["Train Loss"]:.4f}\t Train Error: {d["Train Error"]:.3f}\t Test Error: {d["Test Error"]:.3f}')
            else:
                print(f'Batch {i}.\t lr: {lr:.3f}\t Test Error: {d["Test Error"]:.3f}')

            
            logger.log_scalars(d)
            logger.flush()


        if (i+1) % args.batches_per_lr_step == 0:
            scheduler.step()

        if (i+1) % batches_per_epoch == 0:
            print(f'[ Epoch {i // batches_per_epoch} ]')

        if (i+1) == args.nbatches:
            break;

        if not args.iid and args.earlystop and d['Train Loss'] < 0.01:
            break; # break if small train loss

    ## Final logging
    logger.save_model(model)

    summary = {}
    summary.update({ f'Final Test {k}' : v for k, v in test_all(te_loader, model, criterion).items()})
    summary.update({ f'Final Train {k}' : v for k, v in test_all(tr_loader, model, criterion).items()})
    summary.update({ f'Final CF10 {k}' : v for k, v in test_all(cifar_test, model, criterion).items()})

    logger.log_summary(summary)
    logger.flush()


if __name__ == '__main__':
    main()

