import cv2
import numpy as np
import torch
import random
import torchvision

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, Subset
from PIL import Image
import logging
import os
import random
import torchvision
import torch
import pandas as pd
import numpy as np

from torchvision import transforms, datasets
from operator import itemgetter
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from core.augmentations import MyRandAugment, MyTrivialAugmentWide
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

class Mydata(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if hasattr(self, 'groups'):
            if index in self.groups:
                sample = self.mfc_transform[0](sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_all_files(self):
        img_name = self.full_filenames[0]
        data_list = [Image.open(self.full_filenames[idx]).convert('RGB') for idx in range(len(self.full_filenames))]
        label_list = [self.labels[idx] for idx in range(len(self.full_filenames))]
        return data_list, label_list
    
    def get_labels(self):
        return self.labels
    
    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups

def get_data(strategy,dataset,magnification,dataroot,random_state=42,test_split=0.2,test_validation=False):
     
    if dataset == 'breakhis':
        resize_size = (448, 448)
    else:  # chestct
        resize_size = (224, 224)

    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
    
    if strategy == 'randaugment':
        train_transform = transforms.Compose([
            RandAugment(),
            train_transform,
        ])

    if strategy == 'trivialaugment':
        train_transform = transforms.Compose([
            TrivialAugmentWide(),
            train_transform,
        ])

    if strategy == 'randaugment_raw':
        train_transform = transforms.Compose([
            MyRandAugment(),
            train_transform,
        ])

    if strategy == 'trivialaugment_raw':
        train_transform = transforms.Compose([
            MyTrivialAugmentWide(),
            train_transform,
        ])

    if 'breakhis' in dataset:
        root_dir = os.path.join(dataroot, 'BreakHis', magnification)
        # 加载完整数据集
        full_dataset = Mydata(root=root_dir, transform=train_transform)
        
        # 获取标签
        labels = [sample[1] for sample in full_dataset.samples]
        
        # 使用StratifiedShuffleSplit进行分层抽样
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
        train_indices, test_indices = next(sss.split(range(len(full_dataset)), labels))
        
        # 创建训练集和测试集的子集
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        # 为测试集设置不同的transform
        # 注意：Subset不直接支持transform，我们需要为子集中的每个样本手动应用测试transform
        # 这里我们创建一个包装类来处理不同的transform
        class TransformSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.samples = subset
                self.transform = transform
                self.targets = [s[1] for s in subset]
                
            def __getitem__(self, index):
                x, y = self.samples[index]
                # 检查x是否已经是Tensor，如果是则不需要再应用transform
                if self.transform and not isinstance(x, torch.Tensor):
                    x = self.transform(x)
                elif self.transform and isinstance(x, torch.Tensor):
                    # 如果x已经是Tensor，我们需要先将其转换回PIL Image再应用transform
                    from torchvision.transforms import ToPILImage
                    to_pil = ToPILImage()
                    x = to_pil(x)
                    x = self.transform(x)
                return x, y
            
            def __len__(self):
                return len(self.samples)
        
        # 应用不同的transform
        traintest_dataset = TransformSubset(train_dataset, train_transform)
        test_dataset = TransformSubset(test_dataset, test_transform)

    if 'chestct' in dataset:
        data, label = [], []
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','train')
        traintest_dataset = Mydata(str(all_folders), transform=train_transform)
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','test')
        test_dataset = Mydata(str(all_folders), transform=test_transform)
    

    return traintest_dataset,test_dataset,resize_size,train_transform











