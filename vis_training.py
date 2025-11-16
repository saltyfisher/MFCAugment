import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from pathlib import Path
import pandas as pd
import seaborn as sns
from collections import Counter

metrics = ['precision','recall','f1']
datasets = ['breakhis840X','breakhis8100X','breakhis8200X','breakhis8400X','EndoscopicBladder','ChestCTScan','kvasir-dataset']
augs = ['','_randaugment','_trivialaugment','mfc']

# all_dirs = Path('./logs/mfc').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize_resize_itrs*')
# all_dirs = Path('./logs').glob(f'{datasets[0]}_resnet18_180epochs_450imsize_randaugment_resize_itrs*')
# all_dirs = Path('./logs').glob(f'{datasets[1]}_resnet18_180epochs_450imsize_resize_itrs*')
# all_dirs = Path('./params_save').glob(f'{datasets[1]}_resnet18_180epochs_450imsize_resize_itrs*')
dataset_id = 0
for aug in augs:
    file_names = f'{datasets[dataset_id]}_resnet18'
    if 'ChestCTScan' in file_names:
        file_names += '_gray_180epochs_320imsize'
    else:
        file_names += '_180epochs_450imsize'
    if aug == 'mfc':
        result_root = './params_save/mfc'
        file_names += '_resize_online_itrs*'
    else:
        result_root = './params_save'
        file_names += f'{aug}_resize_itrs*'
    all_dirs = list(Path(result_root).glob(file_names))
    # all_dirs = Path('./params_save').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize{augs[0]}_resize_itrs*')
    # all_dirs = Path('./params_save/mfc').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize_resize_online_itrs*')
    
    # 存储所有训练loss数据
    all_train_losses = []
    
    for f in all_dirs:
        result = torch.load(str(f))
        
        # 提取训练loss
        if 'log' in result and 'train' in result['log']:
            train_losses = [epoch['loss'] for epoch in result['log']['train'] if 'loss' in epoch]
            all_train_losses.append(train_losses)
    
    # 如果有数据则绘制曲线图
    if all_train_losses:
        # 将所有训练loss对齐到相同长度
        min_length = min(len(losses) for losses in all_train_losses)
        aligned_losses = [losses[:min_length] for losses in all_train_losses]
        
        # 转换为numpy数组便于计算
        losses_array = np.array(aligned_losses)
        
        # 计算平均值和标准差
        mean_losses = np.mean(losses_array, axis=0)
        std_losses = np.std(losses_array, axis=0)
        
        # 绘制loss曲线图
        epochs = np.arange(1, len(mean_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_losses, label=f'{aug} mean loss', color='blue')
        plt.fill_between(epochs, mean_losses - std_losses, mean_losses + std_losses, 
                         color='blue', alpha=0.2, label=f'{aug} std')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss for {datasets[dataset_id]} with {aug} augmentation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        output_dir = Path('./vis/loss_curves')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'{datasets[dataset_id]}{aug}_loss_curve.png')
        plt.close()