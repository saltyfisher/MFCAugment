import copy
import torch
import aug_lib
import os
import gc
import itertools
import logging
import numpy as np
import multiprocessing as mp
import yaml
import subprocess
import time

# from core.MFCAugment import MFCAugment
from sklearnex import patch_sklearn
from copy import deepcopy
from pathlib import Path
from sklearn.metrics import f1_score,precision_score,recall_score
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data import get_data
from networks import get_model, num_class
from utils import initialize_setting
from theconf import Config as C
from lr_scheduler import adjust_learning_rate_resnet
from metrics import accuracy, Accumulator, f1_score,precision_score,recall_score
from warmup_scheduler import GradualWarmupScheduler
from common import get_logger
from core.MFCAugment import MFCAugment

logger = get_logger('MFC')
logger.setLevel(logging.DEBUG)

def run_epoch(args, config, model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None,sample_pairing_loader=None):
    cnt = 0
    eval_cnt = 0
    total_steps = len(loader)
    steps = 0
    device = next(model.parameters()).device
    metrics = Accumulator()
    gc.collect()
    torch.cuda.empty_cache()

    all_labels = []
    all_preds = []
    for batch in loader:
        data, label = batch[:2]
        steps += 1

        data, label = data.to(device), label.to(device)
        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)
        if optimizer:
            loss.backward()

        all_labels.extend(label)
        all_preds.extend(preds.detach())
        if config['optimizer'].get('clip', 5) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip', 5))
        if (steps-1) % config.get('step_optimizer_every', 1) == config.get('step_optimizer_nth_step', 0): # default is to step on the first step of each pack
            if optimizer:
                optimizer.step()
        #print(f"Time for forward/backward {time()-fb_time}")
        # if steps % 2 == 0:
        #     metrics.add('eval_top1', top1[0].item() * len(data)) # times 2 since it is only recorded every sec step
        #     eval_cnt += len(data)
        cnt += len(data)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)
        metrics.add_dict({'loss': loss.item()})
        #before_load_time = time()
        del preds, loss, data, label
        if optimizer:
            metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    precision = precision_score(config, torch.stack(all_preds), torch.stack(all_labels))
    recall = recall_score(config, torch.stack(all_preds), torch.stack(all_labels))
    f1 = f1_score(config, torch.stack(all_preds), torch.stack(all_labels))
    metrics.add_dict({
        'precision': precision[0].item(),
        'recall': recall[0].item(),
        'f1':f1[0].item()
    })
    metrics.metrics['loss'] = metrics.metrics['loss']/cnt
    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)
    return metrics

def train_val(gpu_id, task_id, args, config, itrs, dataroot, save_path=None, only_eval=False, only_train=False):
    os.makedirs('./logs/mfc', exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    save_name = Path(args.config).stem
    save_path = str(Path(save_path).joinpath(save_name+f'_itrs{itrs+1}'))
    writers = [SummaryWriter(log_dir=f'./logs/mfc/{save_path}/{x}/') for x in ['train', 'valid', 'test']]
    model = get_model(config['model'], config['batch'], num_class(config['dataset']), writer=writers[0])
    if args.use_parallel:
        model = model.to(f'cuda:{gpu_id}')
    else:
        model = model.to(f'cuda:{args.device}')

    traintestloader,testloader,trainvalloaders,valdloaders,traintest_dataset,trainval_datasets,val_datasets,test_dataset = get_data(config,config['dataset'], dataroot,config['batch'])
    rawdataloader = deepcopy(valdloaders[0])
    rawtrainloader = deepcopy(traintestloader)
    criterion = nn.CrossEntropyLoss()
    if config['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer']['decay'],
            nesterov=config['optimizer']['nesterov']
        )
    elif config['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            betas=(config['optimizer'].get('momentum',.9),.999)
        )

    lr_scheduler_type = config['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(config,optimizer)
    elif lr_scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.)

    if config['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=config['lr_schedule']['warmup']['multiplier'],
            total_epoch=config['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )
    max_epoch = config['epoch']
    epoch_start = 1
    rs = {'train':[],'test':[]}

    if only_eval:
        max_epoch = 0
    best_f1 = 0
    optimal_params = {'model':[],'optimizer':[],'epoch':[]}

    feature_list = ['brightness', 'contrast', 'blurrness', 'hed']
    if args.MFC:
        MFCAugment(traintest_dataset)
    if 'rand' in config['aug']:
        mfc = MFCAugmentEA(augmentation_space='rand')
    elif 'trivial' in config['aug']:
        mfc = MFCAugment(augmentation_space='trivial')
    # if 'rect' in config['dataset']:
    #     for _ in range(5):
    #         optimal_policy.extend(mfc.find_optimal_policy(traintest_dataset, feature_extractor, max_iter=80))
    #     if 'rand' in config['aug']: 
    #         traintestloader.dataset.transform.transforms.insert(0, MyRandAugment(optimal_policy))
    #     elif 'trivial' in config['aug']:
    #         traintestloader.dataset.transform.transforms.insert(0, MyTrivialAugment(optimal_policy))
    # else:
    #     for dataset in val_datasets:
    #         optimal_policy.extend(mfc.find_optimal_policy(dataset, feature_extractor, max_iter=10))
    #     if 'rand' in config['aug']: 
    #         traintestloader.dataset.transform.transforms.insert(0, MyRandAugment(optimal_policy))
    #     elif 'trivial' in config['aug']:
    #         traintestloader.dataset.transform.transforms.insert(0, MyTrivialAugment(optimal_policy))

    for epoch in tqdm(range(epoch_start, max_epoch + 1),desc=f"Task{task_id} Iter{itrs}"):
        model.train()     

        rs['train'].append(run_epoch(args,config,model, traintestloader, criterion, optimizer, desc_default='', epoch=epoch, writer=writers[0], verbose=0, scheduler=None,sample_pairing_loader=None))
        model.eval()

        result = OrderedDict()
        
        if only_eval or epoch % 20 == 0 or epoch == max_epoch:
            with torch.no_grad():
                rs['test'].append(run_epoch(args,config,model, testloader
                , criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True))

            if rs['test'][-1]['f1'] > best_f1:
                best_f1 = rs['test'][-1]['f1']
                if save_path and config.get('save_model', True):
                    # logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': [rs['train'][i].get_dict() for i in range(len(rs['train']))],
                            'test': [rs['test'][i].get_dict() for i in range(len(rs['test']))],
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                    }, save_path+'.pth')   

    del model    
    del rs

def run_python_file(args):
    # command = [python_executable, file_name] + list(args)
    # result = subprocess.run(args, capture_output=True, text=True)
    result = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout:
        print(line, end='')

    result.wait()
    return result.returncode

if __name__ == '__main__':
    patch_sklearn()
    args = initialize_setting()
    # C(args.config)
    dataset_names = ['lym','breakhis40X','breakhis100X','breakhis200X','breakhis400X']
    for d in dataset_names:
        args.dataset = d
        if 'lym' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('lym*'))
        elif 'lc' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('lc*'))
        elif 'breakhis40X' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('*40X*'))
        elif 'breakhis100X' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('*100X*'))
        elif 'breakhis200X' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('*200X*'))
        elif 'breakhis400X' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('*400X*'))
        elif 'rect' in args.dataset:
            all_config_files = list(Path('./networks/confs').glob('rect*'))

        all_config = []
        all_args = []
        for f in all_config_files:
            with open(str(f),'r') as file:
                cfg = yaml.load(file,Loader=yaml.FullLoader)
            if 'vgg' not in cfg['model']['type']:
                continue
            new_args = deepcopy(args)
            new_args.config = str(f)
            all_args.append(new_args)
            all_config.append(cfg)
            # train_val(0%num_gpus,0,args, all_config[0],1,'../data','./params_save')
        
        for itrs in range(1):        
            for args, cfg in zip(all_args, all_config):
                save_path = './params_save'
                save_name = args.config.split(".yaml")[0].split("/")[-1]
                # if 'lym' in save_name and 'rand' in save_name:
                # if 'rand' in save_name or 'trivial' in save_name:
                save_path = f'{save_path}/{save_name}_itrs{itrs+1}' 
                # if os.path.exists(save_path+'.pth'):
                #     continue
                print(save_name)
                train_val(0, 0, args, cfg, itrs, '../data','./params_save/mfc')

                # break


