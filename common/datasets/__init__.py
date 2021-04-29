import torch
import torchvision.datasets as ds
import numpy as np
from torchvision.transforms import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os.path as path
from os.path import join as pjoin

def dload(gpath, localdir='~/tmp/data/', crc=True, overwrite=False):
    ''' Downloads object from GCS into localdir (if not exists), and returns the local filename'''
    import subprocess
    import pickle
    localdir = path.expanduser(localdir)
    local_fname =  pjoin(localdir, path.basename(gpath))
    if path.isfile(local_fname) and not overwrite:
        print("file already downloaded:", local_fname)
        return local_fname
    subprocess.call(f'mkdir -p {localdir}', shell=True)
    if not crc:
        # skip CRC hash check (for machines without crcmod installed)
        subprocess.call(f'gsutil -m -o GSUtil:check_hashes=never cp {gpath} {local_fname}', shell=True)
    else:
        subprocess.call(f'gsutil -m cp {gpath} {local_fname}', shell=True)
    return local_fname

def download_dir(gpath, localroot='~/tmp/data', no_clobber=True):
    ''' Downloads GCS dir into localdir (if not exists), and returns the local dir path.'''
    import subprocess
    import pickle
    localroot = path.expanduser(localroot)

    nc = '-n' if no_clobber else ''
    subprocess.call(f'mkdir -p {localroot}', shell=True)
    subprocess.call(f'gsutil -m cp {nc} -r {gpath} {localroot}', shell=True)
    localdir = pjoin(localroot, path.basename(gpath))
    return localdir



class TransformingTensorDataset(Dataset):
    """TensorDataset with support of torchvision transforms.
    """
    def __init__(self, X, Y, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        y = self.Y[index]

        return x, y

    def __len__(self):
        return len(self.X)

def load_cifar5m():
    '''
        Returns 5million synthetic samples.
        warning: returns as numpy array of unit8s, not torch tensors.
    '''

    nte = 10000 # num. of test samples to use (max 1e6)
    print('Downloading CIFAR 5mil...')
    local_dir = download_dir('gs://gresearch/cifar5m') # download all 6 dataset files

    npart = 1000448
    X_tr = np.empty((5*npart, 32, 32, 3), dtype=np.uint8)
    Ys = []
    print('Loading CIFAR 5mil...')
    for i in range(5):
        z = np.load(pjoin(local_dir, f'part{i}.npz'))
        X_tr[i*npart: (i+1)*npart] = z['X']
        Ys.append(torch.tensor(z['Y']).long())
        print(f'Loaded part {i+1}/6')
    Y_tr = torch.cat(Ys)
    
    z = np.load(pjoin(local_dir, 'part5.npz')) # use the 6th million for test.
    print(f'Loaded part 6/6')
    
    X_te = z['X'][:nte]
    Y_te = torch.tensor(z['Y'][:nte]).long()
    
    return X_tr, Y_tr, X_te, Y_te
    
def load_cifar500():
    import pickle

    big = dload('gs://gpreetum/datasets/cifar500/ti_500K_pseudo_labeled.pickle')
    data = pickle.load(open(big, 'rb'))
    X = data['data']
    Y_orig = torch.Tensor(data['extrapolated_targets']).long()
    X = torch.Tensor(np.transpose(X, (0, 3, 1, 2))).float() / 255.0 * 2.0 - 1.0 # [-1, 1]
    
    # Load Big-Transfer (BiT-M) predictions
    Y = torch.Tensor(np.load(dload('gs://gpreetum/datasets/cifar500/preds_bit.npy'))).long()
    
    I = np.flatnonzero(Y == Y_orig) # only keep indices where original and BiT predictions match (94% agree)
    I = np.random.RandomState(seed=42).permutation(I)
    return X[I], Y[I]

def load_cifar550():
    '''
        Train set: all of CIFAR-10 train + CIFAR-500.
        Test set: 10k iid samples from the above.
    '''
    def split(X, Y, nte=10000):
        ''' splits X, Y tensors in to train and test sets '''
        return X[nte:], Y[nte:], X[:nte], Y[:nte]

    X1, Y1, _, _ = load_cifar() # don't use the CIFAR-10 test set
    X2, Y2 = load_cifar500()

    X = torch.cat([X1, X2])
    Y = torch.cat([Y1, Y2])
    I = np.random.RandomState(seed=42).permutation(len(X))
    return split(X[I], Y[I]) # split 10k into the test set


def load_cifar(datadir='~/tmp/data'):
    train_ds = ds.CIFAR10(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.CIFAR10(root=datadir, train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0  # [0, 1]
        X = X * 2.0 - 1 # [-1, 1]
        Y = torch.Tensor(np.array(dataset.targets)).long()
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def cifar_labels():
    return ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar100(datadir='~/tmp/data'):
    train_ds = ds.CIFAR100(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.CIFAR100(root=datadir, train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0  # [0, 1]
        X = X * 2.0 - 1 # [-1, 1]
        Y = torch.Tensor(np.array(dataset.targets)).long()
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te


def get_cifar_data_aug():
    """
        Returns a torchvision transform that maps (normalized Tensor in [-1, 1]) --> (normalized Tensor)
        via a random data augmentation.
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    unnormalize = transforms.Compose([
        transforms.Normalize((0, 0, 0), (2, 2, 2)),
        transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
    ])

    return transforms.Compose(
        [unnormalize,
         transforms.ToPILImage(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])

def load_mnist():
    train_ds = ds.MNIST(root='~/tmp/data', train=True,
                                download=True, transform=None)
    test_ds = ds.MNIST(root='~/tmp/data', train=False,
                               download=True, transform=None)

    def to_xy(dataset):
        X = torch.Tensor(np.array(dataset.data)).float() / 255.0
        X = X.unsqueeze(1) # add the channel dim
        Y = torch.Tensor(np.array(dataset.targets)).long()
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def load_fmnist():
    train_ds = ds.FashionMNIST(root='~/tmp/data', train=True,
                                download=True, transform=None)
    test_ds = ds.FashionMNIST(root='~/tmp/data', train=False,
                               download=True, transform=None)

    def to_xy(dataset):
        X = torch.Tensor(np.array(dataset.data)).float() / 255.0
        X = X.unsqueeze(1) # add the channel dim
        Y = torch.Tensor(np.array(dataset.targets)).long()
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def load_svhn():
    '''Loads raw (non-class-balanced) SVHN.'''

    def to_xy(dataset):
        X = torch.Tensor(dataset.data).float() / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        Y = torch.Tensor(dataset.labels).long()
        return X, Y

    cf_train = ds.SVHN(root='~/tmp/data', split='train', download=True, transform=None)
    cf_test = ds.SVHN(root='~/tmp/data', split='test', download=True, transform=None)
    X_tr, Y_tr = to_xy(cf_train)
    X_te, Y_te = to_xy(cf_test)
    return X_tr, Y_tr, X_te, Y_te


def load_svhn_all():
    '''Loads all of SVHN, including the extra images.'''

    def to_xy(dataset):
        X = torch.Tensor(dataset.data).float() / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        Y = torch.Tensor(dataset.labels).long()
        return X, Y

    train = ds.SVHN(root='~/tmp/data', split='train', download=True, transform=None)
    test = ds.SVHN(root='~/tmp/data', split='test', download=True, transform=None)
    extra = ds.SVHN(root='~/tmp/data', split='extra', download=True, transform=None)
    X1, Y1 = to_xy(train)
    X2, Y2 = to_xy(test)
    X3, Y3 = to_xy(extra)
    
    X = torch.cat([X1, X2, X3])
    Y = torch.cat([Y1, Y2, Y3])
    I = np.random.RandomState(seed=42).permutation(len(X))
    
    return X[I], Y[I]