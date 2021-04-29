
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import string
import subprocess

from common import gsave, gload
from common import save_model as _save_model


def get_guid(k=6):
    return ''.join(np.random.choice(list(string.ascii_lowercase), size=k))

def _gen_comment(args):
    param_list = vars(args).keys()
    comment = ''
    for i, p in enumerate(param_list):
        if i > 0:
            comment += '_'
        if len(p) > 0:
            comment += f'{p}-{args.__dict__[p]}'
    return comment


def _run_hash(args):
    def valid_attr(k):
        return (k not in ['proj', 'dataset', 'epochs', 'workers', 'run_id', 'scalars', 'saveparam', 'half', 'k', 'iid', 'hash', 'comment']) and not k.startswith('Final')
    config = vars(args)
    attrs = [k for k in config.keys() if valid_attr(k)]
    # print('hashed attributes:', attrs)
    return '_'.join([f'{a}-{config[a]}' for a in attrs])


class VanillaLogger():
    '''
        Logs to GCS, wandb (internal).
    '''
    def __init__(self, args, wandb, gcs_root='gs://gpreetum', hash=False):
        
        if hash: args.hash = _run_hash(args) # for easy run-grouping
        self.wandb = wandb
        proj_name = args.proj
        wandb.config.update(args) # set wandb config to arguments
        self.run_id = wandb.run.id
        args.run_id = self.run_id

        self.gcs_logdir = f'{gcs_root}/logs/{proj_name}/{self.run_id}'
        self.gcs_modeldir = f'{gcs_root}/models/{proj_name}/{self.run_id}'

        print("GCS Logdir:", self.gcs_logdir)
        self.save(vars(args), 'config')

        self._step = 0
        comment = _gen_comment(args)
        # self.tbwriter = SummaryWriter(log_dir=f'gs://gpreetum/runs/{proj_name}/{self.run_id}_{comment}', flush_secs=30)
        self.scalars_log = []

    def save(self, obj, ext):
        gsave(obj, f'{self.gcs_logdir}/{ext}')

    def save_model(self, model):
        gcs_path = f'{self.gcs_modeldir}/model.pt'
        _save_model(model, gcs_path)

    def save_model_step(self, step, model):
        gcs_path = f'{self.gcs_modeldir}/step{step:06}/model.pt'
        _save_model(model, gcs_path)

    def log_root(self, D : dict):
        for k, v in D.items():
            self.save(v, k)

    def log_scalars(self, D : dict, step=None, log_wandb=True):
        if step is None:
            step = self._step
            self._step += 1

        if log_wandb: self.wandb.log(D)
        self.scalars_log.append(D)
        # for k, v in D.items():
        #     self.tbwriter.add_scalar(k, v, global_step = step)
        

    def log_summary(self, D):
        self.wandb.summary = D
        self.save(D, 'summary')

    def log_step(self, step, D : dict):
        prefix = f'steps/step{step:06}'
        for k, v in D.items():
            self.save(v, f'{prefix}/{k}')

    def log_final(self, D : dict):
        prefix = f'final'
        for k, v in D.items():
            self.save(v, f'{prefix}/{k}')
    
    def flush(self):
        ''' saves the result of all log_scalar calls '''
        self.save(self.scalars_log, 'scalars')
        # self.tbwriter.flush()

class VanillaTBLogger():
    def __init__(self, args, proj_name, gcs_root='gs://gpreetum/logs', comment=""):
        self.run_id = get_guid()
        args.run_id = self.run_id
        self.gcs_logdir = f'{gcs_root}/{proj_name}/{self.run_id}'
        print("GCS Logdir:", self.gcs_logdir)
        self.save(vars(args), 'config')

        self._step = 0
        self.tbwriter = SummaryWriter(log_dir=f'gs://gpreetum/runs/{proj_name}/{self.run_id}_{comment}', flush_secs=10)
        self.scalars_log = []

    def save(self, obj, ext):
        gsave(obj, f'{self.gcs_logdir}/{ext}')

    def log_root(self, D : dict):
        for k, v in D.items():
            self.save(v, k)

    def log_scalars(self, D : dict, step=None):
        if step is None:
            step = self._step
            self._step += 1

        self.scalars_log.append(D)
        for k, v in D.items():
            self.tbwriter.add_scalar(k, v, global_step = step)
        

    def log_summary(self, D):
        self.save(D, 'summary')

    def log_step(self, step, D : dict):
        prefix = f'steps/step{step:06}'
        for k, v in D.items():
            self.save(v, f'{prefix}/{k}')

    def log_final(self, D : dict):
        prefix = f'final'
        for k, v in D.items():
            self.save(v, f'{prefix}/{k}')
    
    def flush(self):
        ''' saves the result of all log_scalar calls '''
        self.save(self.scalars_log, 'scalars')
        self.tbwriter.flush()