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
import torchvision

# from core.MFCAugment import MFCAugment
from PIL import Image
from torchvision import transforms
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
from core.augmentations import MyAugment

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
        cnt += len(data)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)
        metrics.add_dict({'loss': loss.item()})
        #before_load_time = time()
        # del preds, loss, data, label
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
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics

def train_val(gpu_id, task_id, args, config, itrs, dataroot, save_path=None, log_path=None, save_name=None, pretrain_model=None):
    if save_path is not None:
        writers = [SummaryWriter(log_dir=str(log_path.joinpath(save_name,x))) for x in ['train', 'valid', 'test']]
    else:
        writers = [None for x in ['train', 'valid', 'test']]
    model = get_model(config['model'], args.batch_size, num_class(config['dataset']), writer=writers[0])
    model = model.to(f'cuda:{args.device}')
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.)
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
    best_f1 = 0

    traintest_dataset,test_dataset = get_data(config,config['dataset'],dataroot)
    transform = torchvision.transforms.Compose([transforms.Resize(config['img_size']),
                                                    transforms.ToTensor()])
    traintest_dataset.update_transform(transform, [], False)
    traintestloader = DataLoader(traintest_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if 'breakhis' in config['dataset']:
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        testloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    data_list, label_list = traintest_dataset.get_all_files()
    for epoch in tqdm(range(epoch_start, max_epoch+1),desc=f"Task{task_id} Iter{itrs}"):
        model.train()     
        rs['train'].append(run_epoch(args,config,model, traintestloader, criterion, optimizer, desc_default='', epoch=epoch, writer=writers[0], verbose=0, scheduler=None,sample_pairing_loader=None))
        model.eval()
        scheduler.step()
        result = OrderedDict()
        
        if (save_path is not None) and (epoch % 20 == 0 or epoch == max_epoch):
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
                    }, str(save_path.joinpath(save_name+'.pth')))   

        if args.MFC and epoch == epoch_start:
            if 'lym' in config['dataset']:
                cluster_num = 3
            if 'breakhis8' in config['dataset']:
                cluster_num = 8
            if pretrain_model is None:
                policy_subset, skill_factor, groups = MFCAugment(model, config, data_list, label_list, args, n_clusters=cluster_num)
            else:
                policy_subset, skill_factor, groups = MFCAugment(pretrain_model, config, data_list, label_list, args, n_clusters=cluster_num)
            optimal_policy = []
            groups = [list(range(len(traintest_dataset)))]
            policy = torchvision.transforms.Compose([MyAugment(policy_subset,mag_bin=args.mag_bin,prob_bin=args.prob_bin,num_ops=args.num_op),
                                                        transforms.Resize(config['img_size']),
                                                        transforms.ToTensor()])
            optimal_policy.append(policy)
            # for idx in range(len(groups)):
            #     policy = [torchvision.transforms.Compose([MyAugment(policy_subset[i],mag_bin=args.mag_bin,             prob_bin=args.prob_bin,num_ops=args.num_op),
            #                                             transforms.Resize(config['img_size']),
            #                                             transforms.ToTensor()]) for i in np.where(skill_factor==idx)[0]]
                # if 'rect' in config['dataset']:
                #     policy = [p.insert(0,transforms.Lambda(lambda image: transforms.F.crop(image,94,94,512,512))) for p in policy]
                # optimal_policy.append(policy)
            traintest_dataset.update_transform(optimal_policy, groups, True)
            traintestloader = DataLoader(traintest_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
            if 'breakhis' in config['dataset']:
                testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            else:
                testloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return model
    
def run_python_file(args):
    # command = [python_executable, file_name] + list(args)
    # result = subprocess.run(args, capture_output=True, text=True)
    result = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout:
        print(line, end='')

    result.wait()
    return result.returncode

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    patch_sklearn()
    args = initialize_setting()
    # C(args.config)
    dataset_names = ['lym','breakhis40X','breakhis100X','breakhis200X','breakhis400X']
    for d in dataset_names:
        # args.dataset = 'breakhis8400X'
        all_config_files = list(Path('./networks/confs').glob(f'{args.dataset}*'))

        all_config = []
        all_args = []
        for f in all_config_files:
            with open(str(f),'r') as file:
                cfg = yaml.load(file,Loader=yaml.FullLoader)
            if args.dataset != '':
                if args.dataset not in cfg['dataset']:
                    continue
            if args.MFC:
                if 'rand' in cfg['aug'] or 'trivial' in cfg['aug']:
                    continue    
            new_args = deepcopy(args)
            new_args.config = str(f)
            all_args.append(new_args)
            all_config.append(cfg)
            # train_val(0%num_gpus,0,args, all_config[0],1,'../data','./params_save')
        
        for itrs in range(10):        
            for args, cfg in zip(all_args, all_config):
                save_path = Path('./params_save')
                log_path = Path('./logs')
                save_name = Path(args.config).stem
                # if 'lym' in save_name and 'rand' in save_name:
                # if 'rand' in save_name or 'trivial' in save_name:
                # if os.path.exists(save_path+'.pth'):
                #     continue
                print(save_name+f'_itrs{itrs+1}')
                if args.MFC:
                    save_path.joinpath('mfc')
                    log_path.joinpath('mfc')
                    args.log_path = log_path
                    args.save_path = save_path
                print('Pretrain start')
                if os.path.exists(f'{save_name}_pretrained.pth'):
                    model = torch.load(f'{save_name}_pretrained.pth')
                else:
                    pretrain_args = deepcopy(args)
                    pretrain_args.MFC = False
                    model = train_val(0, 0, pretrain_args, cfg, itrs, '../MedicalImageClassficationData')
                    torch.save(model, f'{save_name}_pretrained.pth')
                print('Pretrain end')
                if args.resize:
                    save_name += '_resize'
                if args.proxy:
                    save_name += '_proxy'
                    args.proxy_log_path = log_path.joinpath('proxy')
                    args.proxy_save_path = save_path.joinpath('proxy')
                    with open('./proxy_config.yaml') as f:
                        proxy_cfg = yaml.load(f,Loader=yaml.FullLoader)
                    args.proxy_config = proxy_cfg
                if args.GD:
                    save_name += '_GD'
                    args.GD_log_path = log_path.joinpath('GD')
                    args.GD_save_path = save_path.joinpath('GD')
                    with open('./GD_config.yaml') as f:
                        GD_cfg = yaml.load(f,Loader=yaml.FullLoader)
                    args.GD_config = GD_cfg
                save_name += f'_itrs{itrs+1}'
                args.save_name = save_name
                # if os.path.exists(f'{save_path}/{save_name}.pth'):
                #     continue
                train_val(0, 0, args, cfg, itrs, '../MedicalImageClassficationData',save_path,log_path,save_name)
                # break   



