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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold

class Mydata(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if hasattr(self, 'groups'):
            if index in self.groups:
                sample = self.mfc_transform[0](sample)
            else:
                sample = self.transform(sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_all_files(self):
        data_list = [self.transform(self.loader(path)) for path, target in self.samples]
        label_list = self.targets
        return data_list, label_list
    
    def get_labels(self):
        return self.labels
    
    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups

class Mydatasubset(Subset):
    def __getitem__(self, index):
        path, target = self.dataset.samples[self.indices[index]]
        sample = self.dataset.loader(path)
        if hasattr(self, 'groups'):
            if index in self.groups:
                sample = self.mfc_transform[0](sample)
            else:
                sample = self.transform(sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_all_files(self):
        data_list = [self.transform(self.dataset.loader(self.dataset.samples[i][0])) for i in self.indices]
        label_list = [self.dataset.targets[i] for i in self.indices]
        return data_list, label_list
    
    def get_labels(self):
        return [self.dataset.targets[i] for i in self.indices]
    
    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups

def split_train_val_dataset(traintest_dataset, train_transform, test_transform, validation_folds=5, random_state=42):
    """
    从训练集中划分验证集
    
    Args:
        traintest_dataset: 原始训练集
        train_transform: 训练集数据增强
        test_transform: 测试集/验证集数据变换
        validation_folds: 交叉验证折数，1表示简单划分，>1表示K折交叉验证
        random_state: 随机种子
        
    Returns:
        tuple: (traintest_dataset, trainval_datasets, val_datasets) 
               完整训练集，用于交叉验证的训练集列表和验证集列表
    """
    # 获取训练集标签
    traintest_labels = traintest_dataset.get_labels()
    
    skf = StratifiedKFold(n_splits=validation_folds, shuffle=True, random_state=random_state)
    trainval_indices = list(range(len(traintest_dataset)))
    trainval_splits = list(skf.split(trainval_indices, traintest_labels))
    
    # 返回多个(train, val)数据集对用于交叉验证
    trainval_datasets = []
    val_datasets = []
    
    for train_idx, val_idx in trainval_splits:
        train_subset = Mydatasubset(traintest_dataset, train_idx)
        val_subset = Mydatasubset(traintest_dataset, val_idx)
        
        train_subset.transform = train_transform
        val_subset.transform = test_transform
        
        trainval_datasets.append(train_subset)
        val_datasets.append(val_subset)
        
    return traintest_dataset, trainval_datasets, val_datasets

def get_data(strategy,dataset,magnification,dataroot,random_state=42,test_split=0.3,validation=False,validation_folds=5):
     
    if dataset == 'breakhis':
        resize_size = (448, 448)
    else:  # chestct
        resize_size = (224, 224)

    train_transform = transforms.Compose([
        # RandAugment(),
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
        traintest_dataset = Mydatasubset(full_dataset, train_indices)
        test_dataset = Mydatasubset(full_dataset, test_indices)

        traintest_dataset.transform = train_transform
        test_dataset.transform = test_transform
        
        # 如果需要验证集，则从训练集中进一步划分
        if validation:         
            trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    if 'chestct' in dataset:
        data, label = [], []
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','train')
        traintest_dataset = Mydata(str(all_folders), transform=train_transform)
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','test')
        test_dataset = Mydata(str(all_folders), transform=test_transform)
        
        # 如果需要验证集，则从训练集中进一步划分
        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    if 'corona' in dataset:
        data, label = [], []
        all_folders = Path(dataroot).joinpath('Coronahack-Chest-XRay-Dataset','train')
        traintest_dataset = Mydata(str(all_folders), transform=train_transform, allow_empty=True)
        all_folders = Path(dataroot).joinpath('Coronahack-Chest-XRay-Dataset','test')
        test_dataset = Mydata(str(all_folders), transform=test_transform, allow_empty=True)
        
        # 如果需要验证集，则从训练集中进一步划分
        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets
    

    return traintest_dataset,test_dataset,resize_size,train_transform











