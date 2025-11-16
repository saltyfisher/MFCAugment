import torch
import aug_lib
import os
import gc
import itertools
import logging
import random
import torchmetrics
# import torch.multiprocessing as mp
import multiprocessing as mp
import torch.distributed as dist
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from copy import deepcopy
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from fastaa.augmentations import augment_list, policy_decoder, Augmentation, remove_deplicates
from hyperopt import hp, tpe, fmin, Trials
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data import get_data
from networks import get_model, num_class
from utils import initialize_setting
from lr_scheduler import adjust_learning_rate_resnet
from metrics import accuracy, Accumulator, f1_score,precision_score,recall_score
from warmup_scheduler import GradualWarmupScheduler
from common import get_logger
from pathlib import Path
import yaml
import argparse
# from core.MFCAugment import MFCAugment

logger = get_logger('MFC')
logger.setLevel(logging.DEBUG)

def run_epoch(gpu_id, args, model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None,sample_pairing_loader=None):
    cnt = 0
    eval_cnt = 0
    total_steps = len(loader)
    steps = 0

    metrics = Accumulator()
    gc.collect()
    torch.cuda.empty_cache()

    for batch in loader:
        data, label = batch[:2]
        steps += 1

        data, label = data.to(f'cuda:{gpu_id}'), label.to(f'cuda:{gpu_id}')

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)
        if optimizer:
            loss.backward()

        # 替换 C.get() 相关代码为 args
        if hasattr(args, 'optimizer') and isinstance(args.optimizer, dict) and args.optimizer.get('clip', 5) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.optimizer.get('clip', 5))
        if (steps-1) % getattr(args, 'step_optimizer_every', 1) == getattr(args, 'step_optimizer_nth_step', 0): # default is to step on the first step of each pack
            if optimizer:
                optimizer.step()
        #print(f"Time for forward/backward {time()-fb_time}")
        precision = precision_score(preds, label)
        recall = recall_score(preds, label)
        f1 = f1_score(preds, label)
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'precision': precision[0].item() * len(data),
            'recall': recall[0].item()* len(data),
            'f1':f1[0].item() * len(data)
        })
        # if steps % 2 == 0:
        #     metrics.add('eval_top1', top1[0].item() * len(data)) # times 2 since it is only recorded every sec step
        #     eval_cnt += len(data)
        cnt += len(data)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        #before_load_time = time()
        del preds, loss, data, label
        if optimizer:
            metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    metrics = metrics.divide(cnt, eval_top1=eval_cnt)
    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)
    return metrics

def train_val(gpu_id, args, itrs, dataloader, save_path=None, run_mode=0):
    # 移除 C.get() 和 C.get().conf = args
    os.makedirs('./logs/', exist_ok=True)
    save_name = args.config.split(".yaml")[0].split("/")[-1]
    save_path = f'{save_path}/{save_name}_itrs{itrs+1}' 
    writers = [SummaryWriter(log_dir=f'./logs/{save_path}/{x}/') for x in ['train', 'valid', 'test']]
    # 替换 C.get() 相关参数为 args
    model = get_model(args.model, args.batch_size, num_class(args.dataset), writer=writers[0])
    model = model.to(f'cuda:{gpu_id}')

    criterion = nn.CrossEntropyLoss()
    # 替换 C.get() 相关参数为 args
    if hasattr(args, 'optimizer') and isinstance(args.optimizer, dict) and args.optimizer.get('type') == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.optimizer.get('momentum', 0.9),
            weight_decay=args.optimizer.get('decay', 0),
            nesterov=args.optimizer.get('nesterov', False)
        )
    elif hasattr(args, 'optimizer') and isinstance(args.optimizer, dict) and args.optimizer.get('type') == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.optimizer.get('momentum', 0.9), 0.999)
        )
    else:
        # 默认使用 Adam 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 替换 C.get() 相关参数为 args
    lr_scheduler_type = getattr(args, 'lr_schedule', {}).get('type', 'cosine') if hasattr(args, 'lr_schedule') else 'cosine'
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(args, 'epoch', 100), eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.)
    else:
        scheduler = None

    if hasattr(args, 'lr_schedule') and isinstance(args.lr_schedule, dict) and args.lr_schedule.get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.lr_schedule['warmup']['multiplier'],
            total_epoch=args.lr_schedule['warmup']['epoch'],
            after_scheduler=scheduler
        )
    # 替换 C.get() 相关参数为 args
    max_epoch = getattr(args, 'epoch', 100)
    epoch_start = 1

    if run_mode == 1:
        max_epoch = 0
        mode1 = 'test'
        model.eval()
        d_loader1 = dataloader
        rs = {mode1:[]}
    elif run_mode == 0:
        mode1 = 'train'
        d_loader1 = dataloader
        rs = {mode1:[]}
    elif run_mode == 2:
        mode1 = 'train'
        mode2 = 'test'
        d_loader1, d_loader2 = dataloader
        rs = {mode1:[],mode2:[]}

    # 添加缺失的变量定义
    task_id = 0
    
    for epoch in tqdm(range(epoch_start, max_epoch + 1), desc=f"Task{task_id}"):
        if run_mode == 0:
            model.train()
        rs[mode1].append(run_epoch(gpu_id,args,model,d_loader1, criterion, optimizer, desc_default='', epoch=epoch, writer=writers[0], verbose=0, scheduler=None,sample_pairing_loader=None))
        model.eval()

        result = OrderedDict()
        best_f1 = 0
        if run_mode == 2 and (epoch % 20 == 0 or epoch == max_epoch):
            with torch.no_grad():
                rs[mode2].append(run_epoch(gpu_id,args,model,d_loader2
                , criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True))

            if rs['test'][-1]['f1'] > best_f1:
                best_f1 = rs['test'][-1]['f1']
                #writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                # writers[2].add_scalar('test_top1/best', rs['test'][-1]['top1'], epoch)

                # save checkpoint
                # 替换 C.get() 相关参数为 args
                if save_path and getattr(args, 'save_model', True):
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': [rs['train'][i].get_dict() for i in range(len(rs['train']))],
                            'test': [rs['test'][i].get_dict() for i in range(len(rs['test']))],
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path+'.pth')
        if epoch > 3:
            break    
    return model    

def eval_ops(augment,args,config,model,dataset):
    # 移除 C.get() 和 C.get().conf = config
    model.eval()
    device = next(model.parameters()).device

    # 替换 C.get() 相关参数为 args
    policy = policy_decoder(augment, args.num_policy, args.num_op, getattr(args, 'aug', 'none'))
    loaders = []
    dataset.transform.transforms.insert(0, Augmentation(policy, getattr(args, 'aug', 'none')))
    for _ in range(args.num_policy):
        valloader = MyDataloader(dataset, batch_size=8, shuffle=False, num_workers=8)
        loaders.append(valloader)

    # 替换 C.get() 相关参数为 args
    f1_metric = torchmetrics.F1Score(task='multiclass',average='macro',num_classes=num_class(args.dataset)).to(device)
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)
                data = data.to(device)
                label = label.to(device)

                preds = model(data)

                loss = loss_fn(preds, label)
                losses.append(loss.detach().cpu().numpy())

                _, preds = preds.topk(1, 1, True, True)
                preds = preds.t()
                correct = preds.eq(label.view(1, -1).expand_as(preds)).detach().cpu().numpy()
                corrects.append(correct)
                del loss, correct, preds, data, label

            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()

            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict({
                'minus_loss': np.sum(losses_min),
                'correct': np.sum(corrects_max),
                'cnt': len(corrects_max)
            })
            del corrects, corrects_max
    except StopIteration:
        pass
    
    del model
    metrics = metrics / 'cnt'
    return -1*metrics['correct']

def search_policy(task_id,args,config,space, model, dataset):
    trials = Trials()
    best_params = fmin(
        fn = lambda x : eval_ops(x,args,config,model,dataset),
        space = space,
        algo = tpe.suggest,
        max_evals = 4,
        trials=trials
    )

    return task_id, best_params, trials

def fast_autoaugment(gpu_id,task_id,args,itrs,trainloader,valloader,space,save_path=None,run_mode=0):
    # 移除 C.get() 和 C.get().conf = config
    model = train_val(gpu_id, args, itrs, trainloader, save_path)
    print('searching policy\n')
    _, best_params, trials = search_policy(task_id,args,args,model,valloader)

    final_policy_set = []
    trials = sorted(trials, key=lambda x:x['result']['loss'])
    for trial in trials[:10]:
        trial['misc']['vals'] = dict(map(lambda x:(x[0],x[1][0]),trial['misc']['vals'].items()))
        # 替换 C.get() 相关参数为 args
        final_policy = policy_decoder(trial['misc']['vals'],args.num_policy,args.num_op,getattr(args, 'aug', 'none'))
        final_policy = remove_deplicates(final_policy)
        final_policy_set.extend(final_policy)

    return final_policy_set

if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = initialize_setting()
    parser = argparse.ArgumentParser(description='Medical Image Classification with UncertaintyMixup')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', 
                        help='数据集目录路径')
    parser.add_argument('--model', type=str, default='resnet18', help='模型选择')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=180, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--strategy', type=str, default='', 
                        help='训练策略')
    parser.add_argument('--dataset', type=str, default='chestct', choices=['chestct', 'breakhis', 'padufes'],
                        help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None, choices=['40', '100', '200', '400', None],
                        help='BreakHis数据集的放大倍数，None表示使用所有倍数')
    parser.add_argument('--test_split', type=float, default=0.2, help='BreakHis数据集的测试集比例')
    parser.add_argument('--num_trials', type=int, default=10, help='独立实验次数')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--save_mixed_results', action='store_true', help='是否保存混合结果可视化图像')
    parser.add_argument('--pretrain', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--test_all', action='store_true', help='是否测试所有方法')
    parser.add_argument('--resize', action='store_true', help='搜索增广策略时是否缩放')
    parser.add_argument('--mfc', action='store_true', help='是否优化增广策略') 
    parser.add_argument('--online', action='store_true', help='是否在线优化增广策略') 
    parser.add_argument('--proxy', action='store_true', help='是否使用代理') 
    parser.add_argument('--GD', action='store_true', help='是否使用生成式模型') 
    parser.add_argument('--num_ops', type=int, default=3, help='增广策略中的操作数') 
    parser.add_argument('--use_prob', action='store_true', help='增广策略中是否需要激活概率') 
    parser.add_argument('--multitask', action='store_true', help='是否启用多任务算法') 
    parser.add_argument('--generative', action='store_true', help='是否在多任务算法中启用生成式模型') 
    parser.add_argument('--group', action='store_true', help='每个数据子集是否单独适配增广策略') 
    parser.add_argument('--lambda', type=int, default=1, help='目标函数权重')
    parser.add_argument('--mag_bin', type=int, default=31, help='变换操作强度离散个数')
    parser.add_argument('--prob_bin', type=int, default=10, help='变换概率离散个数')
    parser.add_argument('--aug', type=str, default='none', help='增强方法')
    parser.add_argument('--epoch', type=int, default=100, help='训练轮数')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--lr_schedule', type=dict, default={'type': 'cosine'}, help='学习率调度器')
    parsed_args = parser.parse_args()
    
    # 将 parsed_args 的属性复制到 args 中
    for arg in vars(parsed_args):
        setattr(args, arg, getattr(parsed_args, arg))

    dataset_names = ['chestct','breakhis']
    magnification = [40,100,200,400]
     
    for dataset in dataset_names:
        save_name = f'fastaa_{dataset}'
        for mag in magnification:
            if dataset == 'breakhis':
                save_name += f'_{args.magnification}'
            if os.path.exists(f'result/{save_name}.csv'):
                continue
            print(save_name)

        # 替换 C.get() 相关参数为 args
            train_dataset,val_dataset, test_dataset,resize_size,transform_train = get_data([],args.dataset,args.magnification,'../MedicalImageClassficationData')
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
            train_loaders = [DataLoader(t, batch_size=args.batch_size, shuffle=True, num_workers=8) for t in train_dataset]
            val_loaders = [t[1] for t in val_dataset]
            
            num_tasks = len(train_loaders)
            max_process = 1
            num_gpus = torch.cuda.device_count()
            
            # 替换 C.get() 相关参数为 args
            ops = augment_list(getattr(args, 'aug', 'none'))
            space = {}
            for i in range(args.num_policy):
                for j in range(args.num_op):
                    space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
                    space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_%d' % (i, j), 0.0, 1.0)
                    space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_%d' % (i, j), 0.0, 1.0)

            # fast_autoaugment(0, 0, args, 1, trainvalloaders[0],val_datasets[0],space,'./params_save/fastaa')
            with mp.Pool(processes=max_process) as pool:
                results = pool.starmap(fast_autoaugment, [(3, i, args, i, train_loaders[i],val_loaders[i],space,'./params_save/fastaa') for i in range(len(val_loaders))])

            final_policy_set = []
            for result in results:
                final_policy_set.extend(result)
            
            model = train_val(0, args, 0, [traintestloader,testloader], './params_save/fastaa', run_mode=2)