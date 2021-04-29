import torch
import torchvision.datasets as ds
import numpy as np
from torchvision.transforms import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm
from .datasets import dload, download_dir
import subprocess

import gcsfs
import pickle

def gopen(gsname, mode='rb'):
    fs = gcsfs.GCSFileSystem()
    if gsname.startswith('gs://'):
        gsname = gsname[len('gs://'):]
    return fs.open(gsname, mode)

def gsave(x, gsname):
    with gopen(gsname, 'wb') as f:
        pickle.dump(x, f)
        
def gload(gsname):
    with gopen(gsname, 'rb') as f:
        x = pickle.load(f)
    return x

def glob(gspath):
    fs = gcsfs.GCSFileSystem()
    return fs.glob(gspath)

def save_model(model, gcs_path):
    def unwrap_model(model): # unwraps DataParallel, etc
        return model.module if hasattr(model, 'module') else model
    local_path = './model.pt'
    torch.save(unwrap_model(model).state_dict(), local_path)
    subprocess.call(f'gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp {local_path} {gcs_path}', shell=True)
    subprocess.call(f'rm {local_path}', shell=True)

def load_state_dict(model, gcs_path, crc=False):
    local_path = dload(gcs_path, overwrite=True, crc=crc)
    model.load_state_dict(torch.load(local_path))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict(model, X, bs=256, dev='cuda:0'):
    yhat = torch.empty(len(X), dtype=torch.long).to(dev)

    model.eval()
    model.to(dev)
    with torch.no_grad():
        for i in range((len(X)-1)//bs + 1):
            xb = X[i*bs : i*bs+bs].to(dev)
            outputs = model(xb)
            _, preds = torch.max(outputs, dim=1)
            yhat[i*bs : i*bs+bs] = preds

    return yhat.cpu()

def predict_ds(model, ds: Dataset, bsize=128):
    ''' Returns loss, acc'''
    test_dl = DataLoader(ds, batch_size=bsize, shuffle=False, num_workers=4)

    model.eval()
    model.cuda()
    allPreds = []
    with torch.no_grad():
        for (xb, yb) in tqdm(test_dl):
            xb, yb = xb.cuda(), yb.cuda()
            outputs = model(xb)
            preds = torch.argmax(outputs[1], dim=1)
            allPreds.append(preds)

    preds = torch.cat(allPreds).long().cpu().numpy().astype(np.uint8)
    return preds

def evaluate(model, X, Y, bsize=512, loss_func=nn.CrossEntropyLoss().cuda()):
    ''' Returns loss, acc'''
    ds = TensorDataset(X, Y)
    test_dl = DataLoader(ds, batch_size=bsize, shuffle=False, num_workers=1)

    model.eval()
    model.cuda()
    nCorrect = 0.0
    nTotal = 0
    net_loss = 0.0
    with torch.no_grad():
        for (xb, yb) in test_dl:
            xb, yb = xb.cuda(), yb.cuda()
            outputs = model(xb)
            loss = len(xb) * loss_func(outputs, yb)
            _, preds = torch.max(outputs, dim=1)
            nCorrect += (preds == yb).float().sum()
            net_loss += loss
            nTotal += preds.size(0)

    acc = nCorrect.cpu().item() / float(nTotal)
    loss = net_loss.cpu().item() / float(nTotal)
    return loss, acc