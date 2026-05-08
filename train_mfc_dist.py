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
import argparse
import models
# from core.MFCAugment import MFCAugment
from PIL import Image
from torchvision import transforms
# from sklearnex import patch_sklearn
from copy import deepcopy
from pathlib import Path
# from sklearn.metrics import f1_score,precision_score,recall_score
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_dist import get_data
from networks import get_model, num_class
from utils import initialize_setting
from theconf import Config as C
from lr_scheduler import adjust_learning_rate_resnet
from metrics import Accumulator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from warmup_scheduler import GradualWarmupScheduler
from common import get_logger
from core.MFCAugment import MFCAugment
from core.augmentations import MyAugment, RandAugment, TrivialAugmentWide
import csv
import statistics

logger = get_logger('MFC')
logger.setLevel(logging.DEBUG)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot

def run_epoch(model, loader, loss_fn, optimizer, device):
    cnt = 0
    steps = 0
    train_loss = 0.0
    all_labels = []
    all_preds = []
    for batch in loader:
        data, label = batch[:2]
        steps += 1
        data = data.to(device) 
        label = label.to(device)
        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)
        if optimizer:
            loss.backward()
        if optimizer:
            optimizer.step()

        _, preds = preds.max(1)
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        cnt += len(data)

        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / cnt
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {}
    metrics['loss'] = train_loss
    metrics['accuracy'] = accuracy
    metrics['recall'] = recall
    metrics['precision'] = precision
    metrics['f1'] = f1

    return metrics

def train_val(model, optimizer, num_classes, args, itrs, dataroot, save_path=None, log_path=None, save_name=None, pretrain_model=None, rank=0):
    
    criterion = nn.CrossEntropyLoss()
    
    max_epoch = args.num_epochs
    epoch_start = 1
    rs = {'train':[],'test':[]}
    best_f1 = 0
    best_metrics = None

    traintest_dataset,test_dataset,resize_size,transform_train = get_data(args.strategy,args.dataset,args.magnification,dataroot)

    # Create distributed samplers
    train_sampler = DistributedSampler(traintest_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    traintestloader = DataLoader(traintest_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False)
    data_list, label_list = traintest_dataset.get_all_files()

    policy = []
    policy_subset = []
    for epoch in range(epoch_start, max_epoch+1):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        
        model.train()    
        st = time.time() 
        metrics = run_epoch(model, traintestloader, criterion, optimizer, rank)
        print('time elapsed: %.2f' % (time.time()-st))
        rs['train'].append(metrics)
        loss = metrics['loss']
        accuracy = metrics['accuracy']
        print(f'Epoch [{epoch}/{max_epoch}] - Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}')
        model.eval()
        # scheduler.step()
        result = OrderedDict()
        
        if (save_path is not None) and (epoch % 20 == 0 or epoch == max_epoch):
            with torch.no_grad():
                metrics = run_epoch(model, testloader, criterion, None, rank)
                rs['test'].append(metrics)
                print(f'Epoch [{epoch}/{max_epoch}] - Test Loss: {metrics["loss"]:.4f}, Test Acc: {metrics["accuracy"]:.4f}')
            if rs['test'][-1]['f1'] > best_f1:
                best_f1 = rs['test'][-1]['f1']
                best_metrics = rs['test'][-1]
            if save_path and rank == 0:  # Only save on rank 0
                # logger.info('save model@%d to %s' % (epoch, save_path))
                
                torch.save({
                'epoch': epoch,
                'log': {
                    'train': [rs['train'][i] for i in range(len(rs['train']))],
                    'test': [rs['test'][i] for i in range(len(rs['test']))],
                },
                # 'optimizer': optimizer.state_dict(),
                # 'model': model.state_dict(),
                'BestPolicy':policy
            }, str(save_path.joinpath(save_name+'.pth')))   
        if args.online:
            trigger = (epoch % 40 == 0)
            # trigger = (epoch == epoch_start)
        else:
            trigger = (epoch == epoch_start)
        if args.mfc and trigger:
            if 'corona' in args.dataset:
                cluster_num = 8
            elif 'breakhis' in args.dataset:
                cluster_num = 8
            elif 'chestct' in args.dataset:
                cluster_num = 3
            else:
                cluster_num = 4
            policy_subset, skill_factor, groups = MFCAugment(model, resize_size, data_list, label_list, args, n_clusters=cluster_num)
            if args.resize:
                    optimal_policy = [torchvision.transforms.Compose([transforms.Resize(resize_size),MyAugment(p,mag_bin=args.mag_bin,prob_bin=args.prob_bin,num_ops=args.num_ops),
                                                        transforms.ToTensor()]) for p in policy_subset]
            else:
                optimal_policy = [torchvision.transforms.Compose([MyAugment(p,mag_bin=args.mag_bin,prob_bin=args.prob_bin,num_ops=args.num_ops),
                                                        transforms.Resize(resize_size),
                                                        transforms.ToTensor()]) for p in policy_subset]
            if args.group:
                g = [0]*len(traintest_dataset)
                for groups_idx, ind_idx in enumerate(groups):
                    for i in ind_idx:
                        g[i] = groups_idx
                groups = g
            else:
                groups = np.unique(np.concatenate(groups))
            policy.append(policy_subset)
            traintest_dataset.update_transform(optimal_policy, transform_train, groups)
            
            # Recreate samplers and loaders after updating transforms
            train_sampler = DistributedSampler(traintest_dataset, shuffle=True)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            
            traintestloader = DataLoader(traintest_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
            testloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False)

    # 输出本次训练的最优结果
    if best_metrics is not None:
        print(f"最佳测试结果 (第 {itrs + 1} 次实验):")
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return model, best_metrics
    
def run_python_file(args):
    # command = [python_executable, file_name] + list(args)
    # result = subprocess.run(args, capture_output=True, text=True)
    result = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout:
        print(line, end='')

    result.wait()
    return result.returncode

def main(rank, world_size, args):
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Create results directory only on rank 0
    if rank == 0:
        os.makedirs('result', exist_ok=True)
    
    # Synchronize all processes
    dist.barrier()
    
    # Each process runs all trials
    all_trials_results = []
    
    for itrs in range(0, args.num_trials):        
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"开始第 {itrs + 1}/{args.num_trials} 次独立实验")
            print(f"{'='*50}")

        num_classes = num_class(args.dataset)
        model = models.__dict__[args.model](num_classes=num_classes).to(rank)
        model = DDP(model, device_ids=[rank])

        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            if rank == 0:
                print(f"Using SGD optimizer with learning rate {args.lr} and momentum {args.momentum}")
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if rank == 0:
                print(f"Using Adam optimizer with learning rate {args.lr}")

        save_path = Path('./params_save')
        log_path = Path('./logs')
        if args.dataset == 'breakhis':
            save_name = f"{args.dataset}_{args.magnification}X"
        else:
            save_name = f"{args.dataset}"
        if args.mfc:
            save_name = save_name + '_mfc'
        else:
            if args.strategy != '':
                save_name = save_name + f'_{args.strategy}'
        if rank == 0:
            print(save_name+f'_itrs{itrs+1}')
        if args.mfc:
            save_path = save_path.joinpath('mfc')
            log_path = log_path.joinpath('mfc')
            args.log_path = log_path
            args.save_path = save_path
        if args.online:
            save_name += '_online'
        if args.proxy:
            save_name += '_proxy'
            args.proxy_log_path = log_path.joinpath('proxy')
            args.proxy_save_path = save_path.joinpath('proxy')
        if args.GD:
            save_name += '_GD'
            args.GD_log_path = log_path.joinpath('GD')
            args.GD_save_path = save_path.joinpath('GD')
        args.save_name = save_name

        os.makedirs(save_path, exist_ok=True)
        model, best_metrics = train_val(model, optimizer, num_classes, args, itrs, '../MedicalImageClassficationData',save_path,log_path,save_name, model, rank)
        # 保存本次实验的最佳结果
        if best_metrics is not None:
            all_trials_results.append(best_metrics)
        # break   
    
    # 所有轮次结束后计算统计信息 (only on rank 0)
    if rank == 0 and all_trials_results:
        print(f"\n{'='*50}")
        print("所有实验结束，统计结果如下:")
        print(f"{'='*50}")
        
        # 获取所有的指标名称
        metric_names = list(all_trials_results[0].keys())
        
        # 计算每个指标的统计信息
        stats = {}
        for metric in metric_names:
            values = [result[metric] for result in all_trials_results]
            stats[metric] = {
                'mean': f'{np.mean(values):.4f}',
                'std': f'{np.std(values):.4f}',
                'max': f'{np.max(values):.4f}',
                'min': f'{np.min(values):.4f}'
            }
            print(f"{metric}:")
            print(f"  均值: {stats[metric]['mean']}")
            print(f"  方差: {stats[metric]['std']}")
            print(f"  最大值: {stats[metric]['max']}")
            print(f"  最小值: {stats[metric]['min']}")
        
        # 保存统计结果到CSV文件
        strategy_name = 'mfc' if args.mfc else args.strategy
        if args.dataset == 'breakhis':
            csv_filename = f"result/{args.dataset}_{args.magnification}X_{strategy_name}_{args.model}.csv"
        else:
            csv_filename = f"result/{args.dataset}_{strategy_name}_{args.model}.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['metric', 'mean', 'std', 'max', 'min']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in metric_names:
                writer.writerow({
                    'metric': metric,
                    'mean': stats[metric]['mean'],
                    'std': stats[metric]['std'],
                    'max': stats[metric]['max'],
                    'min': stats[metric]['min']
                })
        
        print(f"\n统计结果已保存到 {csv_filename}")
    
    cleanup()

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # patch_sklearn()
    parser = argparse.ArgumentParser(description='Medical Image Classification with UncertaintyMixup')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', 
                        help='数据集目录路径')
    parser.add_argument('--model', type=str, default='resnet18', help='模型选择')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=180, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--strategy', type=str, default='', 
                        help='训练策略')
    parser.add_argument('--dataset', type=str, default='chestct', choices=['chestct', 'breakhis', 'padufes','corona'],
                        help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None, choices=['40', '100', '200', '400', None],
                        help='BreakHis数据集的放大倍数，None表示使用所有倍数')
    parser.add_argument('--test_split', type=float, default=0.2, help='BreakHis数据集的测试集比例')
    parser.add_argument('--num_trials', type=int, default=10, help='独立实验次数')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--save_mixed_results', action='store_true', help='是否保存混合结果可视化图像')
    parser.add_argument('--gpu', action='store_true', help='是否在显存上进行计算')
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
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    args = parser.parse_args()
    
    world_size = args.world_size
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)