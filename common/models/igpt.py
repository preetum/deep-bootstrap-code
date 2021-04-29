import os
import os.path as path
from os.path import join as pjoin
import transformers
from transformers.modeling_gpt2 import GPT2Model,GPT2LMHeadModel
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, Dataset
from torchvision.transforms import transforms
import torchvision.datasets as ds
import logging
logger = logging.getLogger(__name__)

from common import dload

##
## Constructor helpers
##

def download_igpt(model_size='s', localdir='~/tmp/data/igpt/', ckpt=1000000):
    ''' Downloads igpt model into localdir (if not exists), and returns the checkpoint filename.
        ckpt={131000,262000,1000000}
    '''
    GCS_DIR = 'gs://gpreetum/igpt/content/models' # location of the igpt checkpoints

    import subprocess
    import pickle
    localdir = path.expanduser(localdir)
    ckpt_name = pjoin(localdir, model_size, f'model.ckpt-{ckpt}.index')
    if path.isfile(ckpt_name):
        print(f"iGPT-{model_size} already downloaded:", ckpt_name)
        return ckpt_name
    subprocess.call(f'mkdir -p {localdir}', shell=True)
    gpath = pjoin(GCS_DIR, model_size)
    subprocess.call(f'gsutil -m cp -r {gpath} {localdir}', shell=True)
    return ckpt_name

def igpt_pretrained(model_size='s', freeze=False, num_classes=10, twoheaded=False):
    '''
        Downloads and returns a pre-trained iGPT model for classification.

        if two-headed: with both language-modeling and classification heads)
        if freeze: train only the last linear layer.
    '''
    MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24) } 
    size = model_size.lower()
    n_embd,n_head,n_layer=MODELS[size] # model hyperparameters
    # vocab_size = len(clusters) + 1 # add one for start of sentence token
    vocab_size = 513
    n_px = 32 # 32 x 32 images

    config = transformers.GPT2Config(
        vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        initializer_range=1.0/np.sqrt(n_embd)
        )
    logging.info(config)

    model_path = download_igpt(size)

    if twoheaded:
        return ImageGPTDoubleHeadModel.from_pretrained(model_path,from_tf=True,config=config, freeze=freeze, num_classes=num_classes)
    else:
        return ImageGPTClfHeadModel.from_pretrained(model_path,from_tf=True,config=config, freeze=freeze, num_classes=num_classes)


def make_igpt(model_size='s', freeze=False, num_classes=10, pretrained=True, ckpt=1000000):
    '''
        Downloads and returns a [pretrained] iGPT model for classification.
        if freeze: train only the last linear layer.
    '''
    MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24) } 
    size = model_size.lower()
    n_embd,n_head,n_layer=MODELS[size] # model hyperparameters
    # vocab_size = len(clusters) + 1 # add one for start of sentence token
    vocab_size = 513
    n_px = 32 # 32 x 32 images
    config = transformers.GPT2Config(
        vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        initializer_range=1.0/np.sqrt(n_embd)
        )
    logging.info(config)
    params = dict(config=config, freeze=freeze, num_classes=num_classes)

    if pretrained:
        model_path = download_igpt(size, ckpt=ckpt)
        return ImageGPTClfHeadModel.from_pretrained(model_path,from_tf=True,**params)
    else:
        return ImageGPTClfHeadModel(replace_norm=False, **params) # don't replace layernorm, because this fails for some reason?


##
## Datasets & Color-transform encoding/decoding.
##

def squared_euclidean_distance_np(a,b):
    b = b.T
    a2 = np.sum(np.square(a),axis=1)
    b2 = np.sum(np.square(b),axis=0)
    ab = np.matmul(a,b)
    d = a2[:,None] - 2*ab + b2[None,:]
    return d

def color_quantize_np(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    return np.argmin(d,axis=1)

def encode(x_norm, clusters):
    '''
        Prepares a batch of images for iGPT: Encodes with the color-clusters, and flattens into Tensor.
        x_norm: numpy array [0, 1]^(B, 32, 32, 3) 
        returns: [0, 511]^(B, 1024)
    '''
    x_enc = color_quantize_np(x_norm,clusters).reshape(x_norm.shape[:-1]) #map pixels to closest color cluster
    x_enc = x_enc.reshape(-1,32*32)
    return torch.Tensor(x_enc).long()

def decode(x_enc, clusters):
    # convert color clusters back to pixels in [0, 255]
    return [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in x_enc]


class igptDataset(Dataset):
    """
        TensorDataset that applies the igpt color-cluster encoding (on top of an optional data_aug transform)
        X: numpy unit8 array [0, 255]^(B, 32, 32, 3) 
    """
    def __init__(self, X, Y, clusters, data_aug=None):
        self.X = X
        self.Y = Y
        self.transform = data_aug
        self.clusters = clusters

    def __getitem__(self, index):
        y = self.Y[index]
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        else:
            x = transforms.ToTensor()(x)

        # x is now a Torch tensor in [0, 1]^(B, 32, 32, 3)
        x_np = x.permute(1, 2, 0).numpy() # back to WHC
        x = encode(x_np[np.newaxis,:], self.clusters).squeeze() # encode is written to accept a batch
        return x, y

    def __len__(self):
        return len(self.X)




def load_cifar_igpt(train_samps=50000, test_samps=10000, aug=True):
    '''
        Loads CIFAR-10 encoded into the igpt color clusters.
        Returns train, val DataSets.
    '''
    X_tr, Y_tr, X_te, Y_te = load_cifar_np()
    clusters = np.load(dload('gs://gpreetum/igpt/content/clusters/kmeans_centers.npy'))

    if aug:
        transf = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    else:
        transf = transforms.ToTensor()
    
    train_set = igptDataset(X_tr[:train_samps], Y_tr[:train_samps], clusters, data_aug=transf)
    val_set = igptDataset(X_te[:test_samps], Y_te[:test_samps], clusters)

    return train_set, val_set


def load_cifar_np(datadir='~/tmp/data'):

    train_ds = ds.CIFAR10(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.CIFAR10(root=datadir, train=False,
                          download=True, transform=None)
    X_tr = train_ds.data # [0, 255]^[N x 32 x 32 x 3]
    X_te = test_ds.data
    Y_tr = torch.Tensor(train_ds.targets).long()
    Y_te = torch.Tensor(test_ds.targets).long()
    return X_tr, Y_tr, X_te, Y_te


##
## Image-GPT Model Definitions
##

class ln_mod(nn.Module):
    def __init__(self, nx,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(nx))

    def forward(self,x):#input is not mean centered
        return x / torch.sqrt( torch.std(x,axis=-1,unbiased=False,keepdim=True)**2 + self.eps ) * self.weight.data[...,:] 

def replace_ln(m, name,config):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.LayerNorm:
            #print('replaced: ', name, attr_str)
            setattr(m, attr_str, ln_mod(config.n_embd,config.layer_norm_epsilon))

    for n, ch in m.named_children():
        replace_ln(ch, n,config)        

def gelu2(x):
    return x * torch.sigmoid(1.702 * x)

def load_tf_weights_in_image_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    print(tf_path)
    logger.debug("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.debug("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")

        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ) or name[-1] in ['_step']:
            logger.debug("Skipping {}".format("/".join(name)))
            continue
        
        pointer = model
        if name[-1] not in ["wtet"]:
            pointer = getattr(pointer, "transformer")
        
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            elif scope_names[0] in ['q_proj','k_proj','v_proj']:
                pointer = getattr(pointer, 'c_attn')
                pointer = getattr(pointer, 'weight')
            elif len(name) ==3 and name[1]=="attn" and scope_names[0]=="c_proj":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="wtet":
                pointer = getattr(pointer, "lm_head")
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="sos":
                pointer = getattr(pointer,"wte")
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if len(name) > 1 and name[1]=="attn" or name[-1]=="wtet" or name[-1]=="sos" or name[-1]=="wte":
            pass #array is used to initialize only part of the pointer so sizes won't match
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
          
        logger.debug("Initialize PyTorch weight {} <-- {}".format(type(pointer), name))

        if name[-1]=="q_proj":
            pointer.data[:,:config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif name[-1]=="k_proj":
            pointer.data[:,config.n_embd:2*config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif name[-1]=="v_proj":
            pointer.data[:,2*config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif (len(name) ==3 and name[1]=="attn" and name[2]=="c_proj" ):
            pointer.data = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) )
        elif name[-1]=="wtet":
            pointer.data = torch.from_numpy(array)
        elif name[-1]=="wte":
            pointer.data[:config.vocab_size-1,:] = torch.from_numpy(array)
        elif name[-1]=="sos":
            pointer.data[-1] = torch.from_numpy(array)
        else:
            pointer.data = torch.from_numpy(array)

    return model


class ImageGPT2LMHeadModel(GPT2LMHeadModel):
    load_tf_weights = load_tf_weights_in_image_gpt2
  
    def __init__(self, config, replace_norm=True):
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
        if replace_norm:
            replace_ln(self,"net",config) #replace layer normalization
        for n in range(config.n_layer):
            self.transformer.h[n].mlp.act = gelu2 #replace activation 

    def tie_weights(self): #image-gpt doesn't tie output and input embeddings
        pass 
    
    
class ImageGPTDoubleHeadModel(ImageGPT2LMHeadModel):
    '''
        iGPT with two heads:
            lm_head, the generative 'language-modeling' head, and
            clf_head, the classification head.

        WARNING: this class written by preetum (may have bugs).
    '''
    def __init__(self, config, num_classes=10, freeze=False):
        super().__init__(config)
        
        if freeze: # freeze everything except the classification head:
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.lm_head.requires_grad_(False)
                
        self.num_classes = num_classes
        self.clf_head = nn.Linear(config.n_embd, num_classes , bias=False)
        torch.nn.init.zeros_(self.clf_head.weight)
        
    def forward(self, x_enc):
        '''
            returns the (lm_head, clf_head) logits, given an encoded input.
        '''
        bs = len(x_enc) # batchsize
        # prepend sequence with start-of-sequence token, and drop last pixel
        inp = torch.cat( (torch.full( (bs,1), self.config.vocab_size - 1 ,device=x_enc.device, dtype=x_enc.dtype), x_enc[:, :-1] ,), axis=1 ).contiguous()
        
        out = self.transformer(inp)
        h = out[0] # last hidden states
        
        lm_logits = self.lm_head(h) # = [B x 1024 x n_embd]. Sequence length=1024.
        h_avg = h.mean(dim=1) # avg-pool over sequence
        clf_logits = self.clf_head(h_avg)
        
        return lm_logits, clf_logits

    def embed(self, x_enc, avg_pool=True):
        '''
            Returns the embedding.
            If avg_pool, avg-pools over the sequence length (as done in iGPT paper)
        '''
        bs = len(x_enc) # batchsize
        # prepend sequence with start-of-sequence token, and drop last pixel
        inp = torch.cat( (torch.full( (bs,1), self.config.vocab_size - 1 ,device=x_enc.device, dtype=x_enc.dtype), x_enc[:, :-1] ,), axis=1 ).contiguous()
        
        out = self.transformer(inp)
        h = out[0] # last hidden states
        
        if avg_pool:
            h_avg = h.mean(dim=1) # avg-pool over sequence
            return h_avg
        else:
            return h

        
    def loss(self, x_enc, clf_labels, lm_logits, clf_logits):
        # Flatten the tokens, compute losses
        xent = nn.CrossEntropyLoss()
        lm_loss = xent(lm_logits.view(-1, self.config.n_embd),  x_enc.view(-1) ) #generative loss
        clf_loss = xent(clf_logits, clf_labels)
            
        return lm_loss, clf_loss


class ImageGPTClfHeadModel(ImageGPT2LMHeadModel):
    '''
        iGPT with only the classification head (clf_head).

        WARNING: this class written by preetum (may have bugs).
    '''
    def __init__(self, config, num_classes=10, freeze=False, replace_norm=True):
        super().__init__(config, replace_norm)
        
        if freeze: # freeze everything except the classification head:
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.lm_head.requires_grad_(False)
                
        self.num_classes = num_classes
        self.clf_head = nn.Linear(config.n_embd, num_classes , bias=False)
        torch.nn.init.zeros_(self.clf_head.weight)
        
    def forward(self, x_enc):
        '''
            returns the (clf_head) logits, given an encoded input.
        '''
        bs = len(x_enc) # batchsize
        # prepend sequence with start-of-sequence token, and drop last pixel
        inp = torch.cat( (torch.full( (bs,1), self.config.vocab_size - 1 ,device=x_enc.device, dtype=x_enc.dtype), x_enc[:, :-1] ,), axis=1 ).contiguous()
        
        out = self.transformer(inp)
        h = out[0] # last hidden states
        
        h_avg = h.mean(dim=1) # avg-pool over sequence
        clf_logits = self.clf_head(h_avg)
        
        return clf_logits

    def embed(self, x_enc, avg_pool=True):
        '''
            Returns the embedding.
            If avg_pool, avg-pools over the sequence length (as done in iGPT paper)
        '''
        bs = len(x_enc) # batchsize
        # prepend sequence with start-of-sequence token, and drop last pixel
        inp = torch.cat( (torch.full( (bs,1), self.config.vocab_size - 1 ,device=x_enc.device, dtype=x_enc.dtype), x_enc[:, :-1] ,), axis=1 ).contiguous()
        
        out = self.transformer(inp)
        h = out[0] # last hidden states
        
        if avg_pool:
            h_avg = h.mean(dim=1) # avg-pool over sequence
            return h_avg
        else:
            return h