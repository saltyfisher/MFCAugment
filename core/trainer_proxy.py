import os
import shutil
import torch
import datetime
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from core.model import G_D
from core.utils import loss_domain, loss_hinge_dis, loss_hinge_gen, AddNoise, LayerMonitor

def test_proxy(args, p, data, label, batch_size):
    rand_idx = torch.randperm(data.shape[0])
    data = data[rand_idx]
    label = label[rand_idx]

    data = torch.split(data, batch_size)
    label = torch.split(label, batch_size)

    metrics = {'loss':[]}
    all_preds = []
    for k, (x, y) in enumerate(zip(data, label)):
        x = x.to(args.device)
        y = y.to(args.device)

        pred = p(x)
        loss = F.l1_loss(pred, y)
        # loss = F.mse_loss(pred, y)
        all_preds.append(pred.item())
        metrics['loss'].append(loss.item())

    corr = stats.pearsonr(np.array(all_preds), torch.tensor(label).cpu().numpy())
    corr1 = stats.spearmanr(np.array(all_preds), torch.tensor(label).cpu().numpy())
    return np.mean(metrics['loss']), corr, corr1

def train_proxy_one_epoch(args, epoch, p, optimizer, data, label, batch_size, writer):
    rand_idx = torch.randperm(data.shape[0])
    data = data[rand_idx]
    label = label[rand_idx]
    
    data = torch.split(data, batch_size)
    label = torch.split(label, batch_size)

    cfg = args.proxy_config
    
    metrics = {'loss':[]}
    transform = transforms.Compose([
        AddNoise(max=0.01)
    ])
    for k, (x, y) in enumerate(zip(data, label)):
        optimizer.zero_grad()
        x = x.to(args.device)
        y = y.to(args.device)

        # x = transform(x)
        pred = p(x)
        # loss = F.mse_loss(pred, y)
        # loss = F.l1_loss(pred, y)
        loss = F.smooth_l1_loss(pred, y)
        loss.backward()
        clip_grad_value_(p.parameters(), cfg['optimizer']['clip'])
        optimizer.step()
        metrics['loss'].append(loss.item())
    
    return np.mean(metrics['loss'])

def train_proxy(p_idx, p, params, data, label):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    kf = KFold(n_splits=5, shuffle=True)
    args = params['args']
    cfg = args.proxy_config
    log_path = args.proxy_log_path
    save_path = args.proxy_save_path
    save_name = args.save_name
    if os.path.exists(str(log_path.joinpath(save_name))):
        shutil.rmtree(str(log_path.joinpath(save_name)))    
    writers = [SummaryWriter(log_dir=str(log_path.joinpath(save_name,f'fold{x+1}/'))) for x in range(5)]
    args = params['args']
    batch_size = args.proxy_config['batch_size']       
    p = p.to(args.device)
    epoch_start = 1
    data = torch.tensor(data, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.float).unsqueeze(1)
    best_p = []
    best_loss = torch.inf
    for j, (train_index, test_index) in enumerate(kf.split(data)):
        train_p = deepcopy(p)
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
        if cfg['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(train_p.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['decay'])
        elif cfg['optimizer']['type'] == 'sgd':
            optimizer = torch.optim.SGD(train_p.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        for e in range(epoch_start, args.proxy_epochs):
            train_p.train()
            loss = train_proxy_one_epoch(args, e, train_p, optimizer, train_data, train_label, batch_size, writers[j])
            train_p.eval()
            writers[j].add_scalar(f"Loss/Train", loss, e)
            for name, param in train_p.named_parameters():
                if param.grad is not None:
                    writers[j].add_scalar(f"Grad/{name}_max", param.grad.abs().max().item(), e)
                    writers[j].add_scalar(f"Grad/{name}_mean", param.grad.mean().item(), e)
            test_loss = 0
            with torch.no_grad():
                test_loss, corr, corr1 = test_proxy(args, train_p, test_data, test_label, 1)
            scheduler.step(test_loss)
            writers[j].add_scalar('Loss/Lr', scheduler.get_last_lr(), e)
            writers[j].add_scalar('Loss/Test', test_loss, e)
            writers[j].add_scalar('Loss/Pearson', corr[0], e)
            writers[j].add_scalar('Loss/Spearman', corr1[0], e)
            writers[j].add_scalar('Loss/P-value', corr[1], e)
            if test_loss < best_loss:
                best_loss = test_loss
                best_p = deepcopy(train_p)    
        pass
    return best_p