import os
import shutil
import torch
import datetime
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torcheval.metrics.functional as Feval
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from tensorboardX import SummaryWriter
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from core.model import G_D
from core.utils import loss_domain, loss_hinge_dis, loss_hinge_gen, AddNoise, LayerMonitor

def train_GD_one_epoch(args, cfg, epoch, GD, G_optim, D_optim, data, task_id, batch_size, device, writer,monitor):
    rand_idx = torch.randperm(data.shape[0])
    train_data = data[rand_idx]
    task_id = task_id[rand_idx]
    rand_idx = torch.randperm(data.shape[0])
    fake_task_id = task_id[rand_idx]
    
    train_data = torch.split(train_data, batch_size)
    train_task_id = torch.split(task_id, batch_size)
    fake_task_id = torch.split(fake_task_id, batch_size)

    if train_data[-1].shape[0] < 16:
        train_data = train_data[:-1]
        train_task_id = train_task_id[:-1]
        fake_task_id = fake_task_id[:-1]

    t_num = task_id.max()+1
    metrics = {'G_loss_dist':0, 'G_loss_cls':0, 'G_loss_recon':0, 'G_loss':0, 'D_loss_dist':0, 'D_loss_cls':0, 'D_loss':0}
    for i, (x, y_train, y_fake) in enumerate(zip(train_data, train_task_id, fake_task_id)):
        x = x.to(device)
        y_train = y_train.to(device)
        y_fake = y_fake.to(device)

        out_real_dist, out_real_cls = GD(x, y_train)
        out_fake_dist, out_fake_cls = GD(x, y_train, y_fake)
        D_loss_dist = loss_hinge_dis(out_fake_dist, out_real_dist)
        D_loss_cls = loss_domain(out_real_cls, y_train)
        D_loss = D_loss_dist + cfg['lambda_domain'] * D_loss_cls
        # D_loss = D_loss_cls
        
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # for name, param in GD.named_parameters():
        #     if 'real_fake' in name and 'bias' in name:
        #         print(f"Bias grad: {param.grad}")
        #         # 同时打印对应层的输入输出
        #         print(f"Layer input range: [{param.min():.2f}, {param.max():.2f}]")
        if epoch%cfg['GD_D_step'] == 0:
            x_fake, out_fake_dist, out_fake_cls = GD(x, y_fake=y_fake, train_G=True)
            G_loss_dist = loss_hinge_gen(out_fake_dist)
            G_loss_cls = loss_domain(out_fake_cls, y_fake)
            G_loss_recon = torch.mean(torch.abs(x_fake - x))
            G_loss = G_loss_dist + cfg['lambda_domain'] * G_loss_cls + cfg['lambda_recon'] * G_loss_recon
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()
        
            metrics['G_loss_dist'] += G_loss_dist.item()
            metrics['G_loss_cls'] += G_loss_cls.item()
            metrics['G_loss_recon'] += G_loss_recon.item()
            metrics['G_loss'] += G_loss.item()
        if monitor is not None:
            monitor.update_step()
        metrics['D_loss_dist'] += D_loss_dist.item()
        metrics['D_loss_cls'] += D_loss_cls.item()
        metrics['D_loss'] += D_loss.item()
    
    for key, value in metrics.items():
        metrics[key] = metrics[key]/len(train_data)
      
    return metrics

def train_GD(params, GD, data, task_id, writers):
    scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    args = params['args']
    cfg = args.GD_config
    log_path = args.GD_log_path
    save_path = args.GD_save_path
    save_name = args.save_name
    currtime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    # if os.path.exists(str(log_path.joinpath(save_name))):
    #     shutil.rmtree(str(log_path.joinpath(save_name)))    
    if writers == None:
        writers = SummaryWriter(log_dir=f'{log_path}/{save_name}-{currtime}/')
    in_channel = params['Lb'].shape[0]
    t_num = len(params['groups'])
    device = args.device

    GD = GD.to(device)

    # G_optim = torch.optim.RMSprop(GD.G.parameters(), lr=cfg['G_optimizer']['lr'], weight_decay=cfg['G_optimizer']['decay'])
    # D_optim = torch.optim.RMSprop(GD.D.parameters(), lr=cfg['G_optimizer']['lr'], weight_decay=cfg['G_optimizer']['decay'])
    if cfg['G_optimizer']['type'] == 'adam':
        G_optim = torch.optim.Adam(GD.G.parameters(), lr=cfg['G_optimizer']['lr'], weight_decay=cfg['G_optimizer']['decay'])
        D_optim = torch.optim.Adam(GD.D.parameters(), lr=cfg['D_optimizer']['lr'], weight_decay=cfg['D_optimizer']['decay'])
    elif cfg['G_optimizer']['type'] == 'sgd':
        G_optim = torch.optim.SGD(GD.G.parameters(), lr=cfg['G_optimizer']['lr'], weight_decay=cfg['G_optimizer']['decay'])
        D_optim = torch.optim.SGD(GD.D.parameters(), lr=cfg['D_optimizer']['lr'], weight_decay=cfg['D_optimizer']['decay'])
    epoch_start = 1

    # writers.add_graph(GD, (data[:cfg['batch_size']].to(device), task_id[:cfg['batch_size']].to(device), task_id[:cfg['batch_size']].to(device)))
    # writers.add_text('Config', str(cfg))
    monitor = None
    # monitor = LayerMonitor(GD, writers, log_interval=1)
    for e in tqdm(range(epoch_start, args.GD_epochs+1), desc='GD training'):
        GD.train()
        metrics = train_GD_one_epoch(args,cfg,e,GD,G_optim,D_optim,data,task_id,cfg['batch_size'],device,writers,monitor)
        for key, value in metrics.items():
            if value == 0:
                continue
            writers.add_scalar(f'Loss/{key}', value, GD.get_counter())
        # for name, param in GD.named_parameters():
        #         if param.grad is not None:
        #             writers.add_scalar(f"Grad/{name}_max", param.grad.abs().max().item(), GD.get_counter())
        #             writers.add_scalar(f"Grad/{name}_mean", param.grad.mean().item(), GD.get_counter())
        GD.eval()
        GD.add_counter()
    
    pass
    return GD

def test_GD(params, GD, data, task_id):
    args = params['args']
    device = args.device
    cfg = args.GD_config
    batch_size = cfg['batch_size']
    data = torch.FloatTensor(data)
    task_id = torch.LongTensor(task_id)
    t_num = task_id.max()+1
    data = torch.split(data, batch_size)
    task_id = torch.split(task_id, batch_size)

    acc = Accuracy(task='multiclass', num_classes=t_num.item()).to(device)
    total_acc = 0
    for i, (x, y) in enumerate(zip(data, task_id)):
        x = x.to(device)
        y = y.to(device)

        out_real_dist, out_real_cls = GD(x, y)
        total_acc += acc(out_real_cls.softmax(dim=1), y)
        # D_loss = D_loss_cls
        
    return total_acc / len(data)

def train_DCls_one_epoch(args, cfg, epoch, DCls, optim, data, task_id, batch_size, device, writer):
    rand_idx = torch.randperm(data.shape[0])
    train_data = data[rand_idx]
    task_id = task_id[rand_idx]
    
    train_data = torch.split(train_data, batch_size)
    train_task_id = torch.split(task_id, batch_size)
    if train_data[-1].shape[0] < 16:
        train_data = train_data[:-1]
        train_task_id = train_task_id[:-1]

    t_num = task_id.max()+1
    metrics = {'DClsLoss':0,'DClsAcc':0}
    for i, (x, y_train) in enumerate(zip(train_data, train_task_id)):
        x = x.to(device)
        y_train = y_train.to(device).unsqueeze(1)

        pred = DCls(x)
        loss = loss_domain(pred, y_train)
        acc = Feval.multiclass_accuracy(pred, y_train.squeeze())
        # D_loss = D_loss_cls
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        metrics['DClsLoss'] += loss.item()
        metrics['DClsAcc'] += acc
    
    for key, value in metrics.items():
        metrics[key] = metrics[key]/len(train_data)
      
    return metrics

def train_DCls(params, DCls, data, task_id, writers):
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    args = params['args']
    cfg = args.GD_config
    # log_path = args.GD_log_path
    # save_path = args.GD_save_path
    # save_name = args.save_name
    # currtime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    # if os.path.exists(str(log_path.joinpath(save_name))):
    #     shutil.rmtree(str(log_path.joinpath(save_name)))    
    # writers = SummaryWriter(log_dir=f'{log_path}/{save_name}-{currtime}/')
    t_num = len(params['groups'])
    device = args.device
    batch_size = cfg['batch_size']
    #TODO:是否要用生成的数据训练分类器

    if cfg['DCls_optimizer']['type'] == 'adam':
        optim = torch.optim.Adam(DCls.parameters(), lr=cfg['DCls_optimizer']['lr'], weight_decay=cfg['DCls_optimizer']['decay'])
    elif cfg['G_optimizer']['type'] == 'sgd':
        optim = torch.optim.SGD(DCls.parameters(), lr=cfg['G_optimizer']['lr'], weight_decay=cfg['G_optimizer']['decay'])
    epoch_start = 1

    monitor = None
    # monitor = LayerMonitor(GD, writers, log_interval=1)
    for e in tqdm(range(epoch_start, args.GD_epochs+1), desc='DCls training'):
        DCls.train()
        metrics = train_DCls_one_epoch(args,cfg,e,DCls,optim,data,task_id,batch_size,device,writers)
        for key, value in metrics.items():
            if value == 0:
                continue
            writers.add_scalar(f'Loss/{key}', value, DCls.get_counter())
        # for name, param in GD.named_parameters():
        #         if param.grad is not None:
        #             writers.add_scalar(f"Grad/{name}_max", param.grad.abs().max().item(), GD.get_counter())
        #             writers.add_scalar(f"Grad/{name}_mean", param.grad.mean().item(), GD.get_counter())
        DCls.eval()
        DCls.add_counter()
    
    pass
    return DCls

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
    if data[-1].shape[0] < 16:
        data = data[:-1]
        label = label[:-1]
    cfg = args.proxy_config
    
    metrics = {'ProxyLoss':0}
    for k, (x, y) in enumerate(zip(data, label)):
        x = x.to(args.device)
        y = y.to(args.device)

        pred = p(x)
        loss = F.smooth_l1_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics['ProxyLoss'] += loss.item()
    
    for key, value in metrics.items():
        metrics[key] = metrics[key]/len(data)

    return metrics

def train_proxy(params, p, data, label, writers):
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    args = params['args']
    cfg = args.GD_config
    # log_path = args.proxy_log_path
    # save_path = args.proxy_save_path
    # save_name = args.save_name
    # if os.path.exists(str(log_path.joinpath(save_name))):
    #     shutil.rmtree(str(log_path.joinpath(save_name)))    
    # writers = [SummaryWriter(log_dir=str(log_path.joinpath(save_name,f'fold{x+1}/'))) for x in range(5)]
    # args = params['args']
    batch_size = args.proxy_config['batch_size']       
    p = p.to(args.device)
    epoch_start = 1
    if cfg['P_optimizer']['type'] == 'adam':
        optim = torch.optim.Adam(p.parameters(), lr=cfg['P_optimizer']['lr'], weight_decay=cfg['P_optimizer']['decay'])
    elif cfg['P_optimizer']['type'] == 'sgd':
        optim = torch.optim.SGD(p.parameters(), lr=cfg['G_optimizer']['lr'], weight_decay=cfg['G_optimizer']['decay'])

    epoch_start = 1
    for e in tqdm(range(epoch_start, args.GD_epochs+1), desc='Proxy training'):
        p.train()
        metrics = train_proxy_one_epoch(args,e,p,optim,data,label,batch_size,writers)
        p.eval()
        for key, value in metrics.items():
            if value == 0:
                continue
            writers.add_scalar(f'Loss/{key}', value, p.get_counter())
        p.eval()
        p.add_counter()
    return p