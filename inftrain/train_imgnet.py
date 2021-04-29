import os
import wandb

import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import numpy as np
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler

from .utils import AverageMeter
from common.datasets.imagenet import ImageDataset, load_imgnet_xy, download_imgnet, load_imgnet10, download_dogbird
from common.datasets.imagenet import load_imgnet10_xy, load_superclasses_xy, FixedSeedDataset
import torchvision.models as tvmodels
from common.logging import VanillaLogger

parser = argparse.ArgumentParser(description='vanilla training')
parser.add_argument('--proj', default='test-imagenet', type=str, help='project name')
parser.add_argument('--dataset', default='dogbird', type=str, help='imagenet,imagenet10,dogbird')
parser.add_argument('--nsamps', default=10000, type=int, help='num. train samples')
parser.add_argument('--size', default=224, type=int, help="input dim to resize to (optional, defaults to 224)")

parser.add_argument('--augreps', default=0, type=int, help="number of augmented-repeates to use, to expand nsamps in the Ideal World.")

parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture')
parser.add_argument('--width', type=int, default=64, help='num. channels for resnet-k family')
parser.add_argument('--batchsize', default=128, type=int) 
parser.add_argument('--k', default=32, type=int, help="log every k batches", dest='k')
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')

parser.add_argument('--opt', default="sgd", type=str)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default="cosine", type=str, help='lr scheduler')
parser.add_argument('--loss', default='xent', choices=['xent', 'mse'], type=str)
parser.add_argument('--aug', default=0, type=int, help='data-aug (0: none, 1: flips, 2: flips+crop)')

# Adam Betas
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta_1')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta_2 (default: 0.999, slow: 0.95)')
parser.add_argument('--adameps', type=float, default=1e-8, help='Adam epsilon')

parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
# for keeping the same LR sched across different samp sizes.
parser.add_argument('--nbatches', default=None, type=int, help='Total num. batches to train for. If specified, overrides EPOCHS.')
parser.add_argument('--batches_per_lr_step', default=390, type=int)

parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', default=0.0, type=float)


parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--half', default=False, action='store_true', help='training with half precision')

parser.add_argument('--inmem', default=False, action='store_true', help='download datasets to /dev/shm (in-memory)')

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

def test_all(loader, model, criterion):
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
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=(args.beta1, args.beta2), eps=args.adameps)

def get_scheduler(args, scheduler_name, optimizer, num_epochs, **kwargs):
    if scheduler_name == 'const':
        return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
    elif scheduler_name == '3step':
        return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
    elif scheduler_name == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)



def get_model(args, model_name, nchannels=3, nclasses=1000, half=False):
    import common.models.large as cmodels # custom models
    ngpus = torch.cuda.device_count()
    print("=> creating model '{}'".format(model_name))

    if model_name.startswith('mlp'): # eg: mlp[512,512,512]
        from common.models32 import mlp
        widths = eval(model_name[3:])
        px = args.size
        model = mlp(widths=widths, indim=px*px*3, num_classes=nclasses)
    elif model_name.startswith('bagnet'):
        model = cmodels.__dict__[model_name](num_classes=nclasses, pretrained=False)
    elif model_name.startswith('sconv'): # sconv9, sconv33
        import common.models.small.behnam as bmodels
        # model = sconv(num_classes=nclasses)
        model = bmodels.__dict__[model_name](num_classes=nclasses)
    elif model_name in ['resnet18k', 'resnet34k', 'resnet50k']:
        model = cmodels.__dict__[model_name](width=args.width, num_classes=nclasses)
        # model = cmodels.resnet18k(width=args.width, num_classes=2)
    else:
        model = tvmodels.__dict__[model_name](num_classes=nclasses, pretrained=False)

    print("Num model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if half:
        print('Using half precision except in Batch Normalization!')
        model = model.half()
        for module in model.modules():
            if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
                module.float()
    return model

def get_transforms(aug: int, size : int):
    if size == 224:
        resize = transforms.Compose([]) # noop
    else:
        resize = transforms.Resize(size)

    if aug == 0:
        tr_aug = transforms.CenterCrop(224)
    elif aug == 1:
        tr_aug = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            ])
    elif aug == 2:
        # standard flip+crop
        tr_aug = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ])

    train_trans =   transforms.Compose([
                    tr_aug,
                    resize,
                    transforms.ToTensor()])

    val_trans = transforms.Compose([
                transforms.CenterCrop(224),
                resize,
                transforms.ToTensor()])

    return train_trans, val_trans

def get_augrep_transform(size: int):
    if size == 224:
        resize = transforms.Compose([]) # noop
    else:
        resize = transforms.Resize(size)
    tr_aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ])
    train_trans =   transforms.Compose([
                    tr_aug,
                    resize,
                    transforms.ToTensor()])
    return train_trans

def main():
    cudnn.benchmark = True
    wandb.init(project=args.proj)
    logger = VanillaLogger(args, wandb)

    datadir = '~/tmp/data' if not args.inmem else '/dev/shm'
    if args.dataset == 'dogbird':
        # dogbird is special, has its own tar (~3 GB)
        root = download_dogbird(dir=datadir, crc=False)
    else:
        root = download_imgnet(dir=datadir, crc=False) # (~20 GB)

    if args.dataset == 'imagenet':
        # tr_set, val_set = load_imgnet(root=root, size=args.size)
        (X_tr, Y_tr, X_te, Y_te) = load_imgnet_xy(root=root)
        num_classes = 1000
    elif args.dataset == 'imagenet10':
        (X_tr, Y_tr, X_te, Y_te) = load_imgnet10_xy(root=root)
        num_classes = 10
    elif args.dataset == 'dogbird':
        cats = ['hunting_dog.n.01', 'bird.n.01']
        num_classes = len(cats)
        (X_tr, Y_tr, X_te, Y_te) = load_superclasses_xy(root, cats = cats)

    print('Real train samples in dataset: ', len(X_tr))
    if args.iid and args.augreps > 0:
        # repeat samples for augmentation
        X_tr = np.tile(X_tr, args.augreps)
        Y_tr = np.tile(Y_tr, args.augreps)
    print('Effective train samples in dataset (with aug-reps): ', len(X_tr))

    if args.iid:
        I = np.random.permutation(len(X_tr)) # just permute
    else:
        I = np.random.permutation(len(X_tr))[:args.nsamps] # permute and subsample
    X_tr, Y_tr = X_tr[I], Y_tr[I]

    print('Num. train samples used: ', len(X_tr))
    print('Num. test samples used: ', len(X_te))
    logger.log_final({'yTe' : Y_te})

    train_trans, val_trans = get_transforms(args.aug, args.size)
    val_set = ImageDataset((X_te, Y_te), transform=val_trans)
    if args.augreps == 0:
        # standard training
        tr_set = ImageDataset((X_tr, Y_tr), transform=train_trans)
    else:
        # expand the train set using augmentations (fixed seed).
        # NO extra data-aug.
        tr_set_base = ImageDataset((X_tr, Y_tr), transform=None)
        # override the train transf with augmentation-transform
        tr_set = FixedSeedDataset(tr_set_base, seed=np.random.randint(2147483647), transform=get_augrep_transform(args.size))
        print("warning: NOT using additional data-aug (beyond aug-rep)")


    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize, drop_last=True,
            shuffle=True, num_workers=args.workers, pin_memory=True)
    te_loader = torch.utils.data.DataLoader(val_set, batch_size=2*args.batchsize, 
            shuffle=False, num_workers=args.workers, pin_memory=True)

    # load the model
    print('Loading model...')
    model = get_model(args, args.arch, nclasses=num_classes, half=args.half)
    model = torch.nn.DataParallel(model).cuda()

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
    scheduler = get_scheduler(args, args.scheduler, optimizer, num_epochs=args.epochs)

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

        if i % args.k == 0:
            ''' Every k batches (and more frequently in early stages): log train/test errors. '''

            d = {'batch_num': i,
                'lr': lr,
                'n' : n_tot}

            test_m = test_all(te_loader, model, criterion)
            d.update({ f'Test {k}' : v for k, v in test_m.items()})

            if not args.iid:
                train_m = test_all(tr_loader, model, criterion)
                d.update({ f'Train {k}' : v for k, v in train_m.items()})

                print(f'Batch {i}.\t lr: {lr:.4f}\t Train Loss: {d["Train Loss"]:.4f}\t Train Error: {d["Train Error"]:.3f}\t Test Error: {d["Test Error"]:.3f}')
            else:
                print(f'Batch {i}.\t lr: {lr:.4f}\t Test Error: {d["Test Error"]:.3f}')

            
            logger.log_scalars(d)
            logger.flush()


        if (i+1) % args.batches_per_lr_step == 0:
            scheduler.step()

        if (i+1) % batches_per_epoch == 0:
            print(f'[ Epoch {i // batches_per_epoch} ]')

        if (i+1) == args.nbatches:
            break;



    predsTe = predict(te_loader, model)
    logger.log_final({'predsTe': predsTe})

    summary = {}
    summary.update({ f'Final Test {k}' : v for k, v in test_all(te_loader, model, criterion).items()})
    summary.update({ f'Final Train {k}' : v for k, v in test_all(tr_loader, model, criterion).items()})
    logger.log_summary(summary)

    logger.save_model(model)
    logger.flush()


if __name__ == '__main__':
    main()

