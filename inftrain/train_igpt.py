import os

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
from torch.utils.data import TensorDataset, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler

from .utils import AverageMeter
from .utils import get_model32, get_optimizer, get_scheduler

from common.datasets import TransformingTensorDataset, get_cifar_data_aug
from common.datasets import load_cifar5m
from common.logging import VanillaLogger
from common import dload
from common.models.igpt import make_igpt, load_cifar_np, igptDataset

parser = argparse.ArgumentParser(description='vanilla training')
parser.add_argument('--proj', default='test-igpt', type=str, help='project name')
parser.add_argument('--dataset', default='cifar5m', type=str)
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')

parser.add_argument('--k', default=8, type=int, help="log every k full-batches", dest='k')
parser.add_argument('--ubatchsize', default=2, type=int, help="microbatch size (split over num gpus)")
parser.add_argument('--ubatches', default=64, type=int, help='number of gradient accumulation batches before gradient-stepping')
parser.add_argument('-vb', '--val-batchsize-mult', type=int, default=16, metavar='N', help='batchsize-multiplier for validation.')

parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default="const", type=str, help='lr scheduler')

parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta_1')
parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta_2')

parser.add_argument('--nsamps', default=50000, type=int, help='num. train samples')
parser.add_argument('--nte', type=int, default=10000, help='num. test samples')

parser.add_argument('--arch', metavar='ARCH', default='igpt-s')
parser.add_argument('--ckpt', type=int, default=1000000, choices=[131000,1000000],help="igpt checkpoint={262000,1000000}")
parser.add_argument('--no_pretrained', default=False, action='store_true')

parser.add_argument('--aug', default=0, type=int, help='data-aug (0: none, 1: flips, 2: all)')
parser.add_argument('--epochs', default=100, type=int)

parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

parser.add_argument('--comment', default=None)
parser.add_argument('--pub_wandb', default=False, action='store_true')

args = parser.parse_args()

if not args.pub_wandb:
    os.environ['WANDB_API_KEY']='local-3a7bef69991b670271c7bb739cbb47b26fa9af31'
    os.environ['WANDB_BASE_URL']='http://10.138.0.5:8888' # google-internal wandb server
import wandb


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i



def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
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

def get_scheduler(args, scheduler_name, optimizer, num_epochs, **kwargs):
    if scheduler_name == 'const':
        return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)


def get_dataset_np(dataset):
    # returns (X_tr, Y_tr, X_te, Y_te)
    if dataset == 'cifar10':
        return load_cifar_np() # X: unit8 
    elif dataset == 'cifar5m':
        return load_cifar5m()
    

def make_igpt_dataset_from_np(X, Y, aug=None) -> Dataset:
    '''
        Returns igpt dataset.
    '''
    clusters = np.load(dload('gs://gpreetum/igpt/content/clusters/kmeans_centers.npy'))
    dset = igptDataset(X, Y, clusters, data_aug=aug)
    return dset

    
def get_data_aug(aug : int):
    ## Note, this must output a tensor in [0, 1].

    if aug == 0:
        print('data-aug: NONE')
        return transforms.ToTensor()
    elif aug == 1:
        print('data-aug: flips only')
        return transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
    elif aug == 2:
        print('data-aug: full')
        return transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
    elif aug == 3:
        print('data-aug: full (reflect-crop)')
        return transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])


def make_loader(x, y, transform=None, batch_size=256):
    dataset = TransformingTensorDataset(x, y, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
    return loader

def main():
    wandb.init(project=args.proj)
    cudnn.benchmark = True

    ## load the model
    model_size = args.arch[-1] # arch='igpt-s'
    model = make_igpt(model_size=model_size, num_classes=10, pretrained=(not args.no_pretrained), ckpt=args.ckpt)
    args.nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num model parameters:", args.nparams)
    model = torch.nn.DataParallel(model).cuda()

    # init logging
    logger = VanillaLogger(args, wandb, hash=True)

    print('Loading datasets...')
    (X_tr, Y_tr, X_te, Y_te) = get_dataset_np(args.dataset)
    X_te, Y_te = X_te[:args.nte], Y_te[:args.nte] # less test samples (for debugging)

    # subsample
    if not args.iid:
        I = np.random.permutation(len(X_tr))[:args.nsamps]
        X_tr, Y_tr = X_tr[I], Y_tr[I]

    tr_set = make_igpt_dataset_from_np(X_tr, Y_tr, aug=get_data_aug(args.aug))
    val_set = make_igpt_dataset_from_np(X_te, Y_te, aug=None)

    # batchsize computations
    args.batchsize = args.ubatchsize * args.ubatches # effective batchsize
    ubatches_per_epoch = int(np.floor(args.nsamps / args.ubatchsize))
    args.num_ubatches = int((args.nsamps / args.ubatchsize) * args.epochs)
    print(f'Num. total train batches: {args.num_ubatches}')


    # data loaders
    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.ubatchsize,
            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True) # drop the last batch if it's incomplete (< batch size)
    te_loader = torch.utils.data.DataLoader(val_set, batch_size=args.ubatchsize * args.val_batchsize_mult,
            shuffle=False, num_workers=args.workers, pin_memory=True)

    tr_val_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.ubatchsize * args.val_batchsize_mult,
            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    print('Done loading.')
    
    # define loss function (criterion), optimizer and scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0)
    scheduler = get_scheduler(args, args.scheduler, optimizer, args.epochs)

    n_tot = 0
    optimizer.zero_grad()
    for i, (images, target) in enumerate(recycle(tr_loader)):
        model.train()
        images, target = cuda_transfer(images, target)
        output = model(images)
        loss = criterion(output, target)
        loss = loss / args.ubatches # rescale loss if accumulating gradients.
        loss.backward()
        n_tot += len(images)

        if (i+1) % args.ubatches == 0: # accumulate gradients
            optimizer.step()
            optimizer.zero_grad()

        ## logging
        if (i+1) % (args.k * args.ubatches) == 0: # every k full-batches
            ''' Every k batches: log train/test errors. '''
            lr = optimizer.param_groups[0]['lr']
            d = {
                'ubatch_num': (i+1),
                'batch_num': (i+1) // args.ubatches,
                'lr': lr,
                'n' : n_tot}

            print("Evaluating on test set...")
            test_m = test_all(te_loader, model, criterion)
            d.update({ f'Test {k}' : v for k, v in test_m.items()})
            print("Done.")

            if not args.iid:
                print("Evaluating on train set...")
                train_m = test_all(tr_val_loader, model, criterion)
                d.update({ f'Train {k}' : v for k, v in train_m.items()})
                print("Done.")

                print(f'uBatch {i}.\t lr: {lr:.3f}\t Train Loss: {d["Train Loss"]:.4f}\t Train Error: {d["Train Error"]:.3f}\t Test Error: {d["Test Error"]:.3f}')
            else:
                print(f'uBatch {i}.\t lr: {lr:.3f}\t Test Error: {d["Test Error"]:.3f}')

            
            logger.log_scalars(d)
            logger.flush()


        if (i+1) % ubatches_per_epoch == 0:
            print(f'[ Epoch {i // ubatches_per_epoch} ]')
            scheduler.step()

        if (i+1) == args.num_ubatches:
            break;


    ## Final logging
    summary = {}
    summary.update({ f'Final Test {k}' : v for k, v in test_all(te_loader, model, criterion).items()})
    summary.update({ f'Final Train {k}' : v for k, v in test_all(tr_loader, model, criterion).items()})

    logger.log_summary(summary)
    logger.flush()

    logger.save_model(model)


if __name__ == '__main__':
    main()

