import os
import os.path
import os.path as path
from os.path import join as pjoin
import sys
import math
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random

from . import dload
from .. import gload

##
## Factories
##


def download_imgnet(dir='~/tmp/data', crc=True):
    ''' Downloads imagenet256.tar, untars, and returns the root path. '''
    import subprocess
    dir = path.expanduser(dir)
    root = pjoin(dir, 'imagenet256')

    if path.isdir(root):
        print("imagenet256 dir already exists.")
        return root

    print("Downloading imagenet256.tar")
    tarfile = dload('gs://gpreetum-central/datasets/imagenet256.tar', dir, crc=crc)
    print("Untarring...")
    subprocess.call(f'tar -xf {tarfile} -C {dir}', shell=True)
    return root


def download_dogbird(dir='~/tmp/data', crc=True):
    ''' Downloads dogbird256.tar, untars, and returns the root path. '''
    import subprocess
    dir = path.expanduser(dir)
    root = pjoin(dir, 'dogbird256')

    if path.isdir(root):
        print("dogbird256 dir already exists.")
        return root

    print("Downloading dogbird256.tar")
    tarfile = dload('gs://gpreetum-central/datasets/dogbird256.tar', dir, crc=crc)
    print("Untarring...")
    subprocess.call(f'tar -xf {tarfile} -C {dir}', shell=True)
    return root
    

def load_imgnet(root, size=224):
    if size == 224:
        resize = transforms.Compose([]) # noop
    else:
        resize = transforms.Resize(size)

    train_trans =   transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    resize,
                    transforms.ToTensor()])
    
    val_trans = transforms.Compose([
                # transforms.Resize(256),  # loaded images already resized to 256
                transforms.CenterCrop(224),
                resize,
                transforms.ToTensor()])
    
    Xtr, Ytr = make_dataset_index(pjoin(root, 'train'))
    Xte, Yte = make_dataset_index(pjoin(root, 'val'))
    
    nToi = names_to_idx(Ytr)
    Ytr = [ nToi[y] for y in Ytr ]
    Yte = [ nToi[y] for y in Yte ]
    
    tr_set = ImageDataset((Xtr, Ytr), transform=train_trans)
    val_set = ImageDataset((Xte, Yte), transform=val_trans)
    
    return tr_set, val_set


def load_imgnet_xy(root, remap_targets = True):
    ''' Returns Xtr, Ytr, Xte, Yte.
        where   X: list of file paths
                Y: list of targets (ints if remapped, or original targets)
        
        This is the appropriate object to construct an ImageDataset
    '''
    Xtr, Ytr = make_dataset_index(pjoin(root, 'train'))
    Xte, Yte = make_dataset_index(pjoin(root, 'val'))

    if remap_targets:
        # map class names --> integers
        nToi = names_to_idx(Ytr)
        Ytr = [ nToi[y] for y in Ytr ]
        Yte = [ nToi[y] for y in Yte ]

    return tuple(map(np.array, (Xtr, Ytr, Xte, Yte)))


## imagenet10

def remap_dataset_index(X, Y, class_map):
    '''
        Takes a dataset index (X, Y: wnid) and subsamples/remaps it according to class_map.
        class_map: Dict[int --> (List of wnids)]
        Outputs (X, Y) where Y : int
    '''
    all_wnids = set.union(*class_map.values()) # all of the relevant wnids
    remap = {} # wnid --> {0, 1, 2...}
    for i, s in class_map.items():
        for w in s:
            remap[w] = i
    
    X, Y = map(np.array, [X, Y])
    Xr, Yr = [], []
    for x, y in zip(X, Y):
        if y in all_wnids:
            Xr.append(x)
            Yr.append(remap[y])
    return map(np.array,[Xr, Yr])

def load_imgnet10(root):
    class_map = gload('gs://gpreetum/datasets/imagenet10/class_map.pkl') 
    return load_imgnet_remapped(root, class_map)

def load_imgnet10_xy(root):
    class_map = gload('gs://gpreetum/datasets/imagenet10/class_map.pkl') 

    Xtr, Ytr, Xte, Yte = load_imgnet_xy(root, remap_targets=False) # Ys: wnids

    # subsample and remap with the imagenet10 superclasses
    Xtr, Ytr = remap_dataset_index(Xtr, Ytr, class_map)
    Xte, Yte = remap_dataset_index(Xte, Yte, class_map)

    return Xtr, Ytr, Xte, Yte 

def load_superclasses_xy(root, cats = ['hunting_dog.n.01', 'bird.n.01']):
    ''' Load superclasses from imagenet, based on wordnet-heirachy. '''
    class_map = get_class_map(cats)
    for i, s in class_map.items():
        nelem = len(s)
        print(f'Class {i}:\t{nelem} subclasses')

    Xtr, Ytr, Xte, Yte = load_imgnet_xy(root, remap_targets=False) # Ys: wnids
    Xtr, Ytr = remap_dataset_index(Xtr, Ytr, class_map)
    Xte, Yte = remap_dataset_index(Xte, Yte, class_map)
    return Xtr, Ytr, Xte, Yte 


def load_imgnet_remapped(root, class_map, size=224):
    ''' deprecated, but convenient function. '''
    Xtr, Ytr = make_dataset_index(pjoin(root, 'train'))
    Xte, Yte = make_dataset_index(pjoin(root, 'val'))

    if size == 224:
        resize = transforms.Compose([]) # noop
    else:
        resize = transforms.Resize(size)

    # subsample and remap with the imagenet10 superclasses
    Xtr, Ytr = remap_dataset_index(Xtr, Ytr, class_map)
    Xte, Yte = remap_dataset_index(Xte, Yte, class_map)

    train_trans =   transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    resize,
                    transforms.ToTensor()])
    
    val_trans = transforms.Compose([
                transforms.CenterCrop(224),
                resize,
                transforms.ToTensor()
            ])
    tr_set = ImageDataset((Xtr, Ytr), transform=train_trans)
    val_set = ImageDataset((Xte, Yte), transform=val_trans)
    return tr_set, val_set


def get_class_map(cats = ['canine.n.02', 'bird.n.01']):
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus.reader.wordnet import Synset
    nltk.download('wordnet')

    wnids = gload('gs://gpreetum/datasets/imagenet10/imagenet_wnids.pkl') # list of wnids in imagenet

    def imgnet_classes_in(super_sys: Synset):
        '''
            Returns all of the imagenet sysnets under a given sysnet.
        '''
        classes = []
        for wid in wnids:
            wid_int = int(wid[1:])
            syn = wn.synset_from_pos_and_offset('n', wid_int)

            ancestry = list(syn.closure(lambda s:s.hypernyms())) # all hypernyms above this
            ancestry = set([syn] + ancestry)
            
            if super_sys in ancestry:
                classes.append(syn)
                
        return classes

    def get_imgnet_names(cat):
        cs = imgnet_classes_in(wn.synset(cat))
        return [f'n0{s.offset()}' for s in cs]

    
    class_map = {i: set(get_imgnet_names(c)) for i, c in enumerate(cats)}

    return class_map

##
## Image loading
##

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path: str):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

##
## Construction utils
##

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def make_dataset_index(dir): 
    '''
        Returns X, Y.
        X: List of file paths to images.
        Y: List of corresponding class names.
    '''
    dir = os.path.expanduser(dir)
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    
    x_idx = []
    y = []
    for c in tqdm(classes):
        d = os.path.join(dir, c)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    x_idx.append( path )
                    y.append(c)
    return x_idx, y

def names_to_idx(Y_names):
    ''' Returns dict mapping class names (str) to class indices (int). '''
    names = list(sorted(list(set(Y_names))))
    nToi = {names[i]: i for i in range(len(names))}
    # Yi = [ nToi[y] for y in Y_names ]
    return nToi

##
## ImageDataset
##

class ImageDataset(Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        dataset_index: tuple (X, Y) of image paths and labels, as produced by make_dataset_index

        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        paths (list)
        targets (list)
    """

    def __init__( self, dataset_index, transform=None, loader=default_loader):
        X, Y = dataset_index

        self.paths = X 
        self.targets = Y
        self.loader = loader
        self.transform = transform


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.paths[index], self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.paths)


    def __repr__(self) -> str:
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)



class FixedSeedDataset(Dataset):
    """
        A transforming-vision dataset wrapper that fixes the seed in torchvision transforms.
    """
    def __init__(self, base_dataset, seed=0, transform=None):
        self.transform = transform
        self.seed = seed
        self.base = base_dataset

    def __getitem__(self, index: int):
        sample, target = self.base[index]
        if self.transform is not None:
            rng = self.seed + index
            random.seed(rng)
            torch.manual_seed(rng)
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.base)

    def __repr__(self):
        return repr(self.base)