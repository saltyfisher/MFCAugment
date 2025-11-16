import logging
import os
import random
import torchvision
import torch
import pandas as pd
import numpy as np

from torchvision import transforms
from operator import itemgetter
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from core.augmentations import MyRandAugment, MyTrivialAugmentWide
from torchvision.transforms import RandAugment, TrivialAugmentWide
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from datasets.dataset import Mydata

# 修改函数签名，添加validation_folds参数，默认为1表示不使用交叉验证
def get_data(strategy,dataset,magnification,dataroot,validation=False,validation_folds=1):
     
    if dataset == 'breakhis':
        resize_size = (448, 448)
    else:  # chestct
        resize_size = (224, 224)
    if dataset == 'chestct':
        transform_test = [transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor()
        ])]
    else:
        transform_test = [transforms.Compose([
            transforms.ToTensor()
        ])]
    if strategy == 'trivialaugment':            
        transform_train = [transforms.Compose([TrivialAugmentWide(),
                                             transforms.Resize(resize_size),
                                             transforms.ToTensor()])]
    elif strategy == 'randaugment':
        transform_train = [transforms.Compose([RandAugment(),
                                             transforms.Resize(resize_size),
                                             transforms.ToTensor()])]
    elif strategy == 'trivialaugment_raw':
        transform_train = [transforms.Compose([MyTrivialAugmentWide(),
                                             transforms.Resize(resize_size),
                                             transforms.ToTensor()])]
    elif strategy == 'randaugment_raw':
        transform_train = [transforms.Compose([MyRandAugment(),
                                             transforms.Resize(resize_size),
                                             transforms.ToTensor()])]
    else:
        transform_train = [transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            ])]
    
    label, data = [], []
    if 'breakhis' in dataset:
        all_folders = Path(dataroot).joinpath('BreakHis','histology_slides','breast')
        label_type = ['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma','ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma']
        for obj in all_folders.rglob('*.png'):  
            if magnification in str(obj.parent):
                data.append(str(obj))
                for i, label_name in enumerate(label_type):
                    if label_name in str(obj.parent):
                        label.append(i)

    if data != []:
        sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
        traintest_idx, test_idx = next(sss.split(data, label))
        getter = itemgetter(*test_idx)
        test_dataset = Mydata(getter(data), getter(label), transform_test)
        getter = itemgetter(*traintest_idx)
        traintest_dataset = Mydata(getter(data), getter(label), transform_train)
        
        # 如果需要验证集，则从训练集中分出一部分作为验证集
        if validation:
            train_data, train_label = getter(data), getter(label)
            # 使用分层抽样将训练集分为训练集和验证集 (80%训练, 20%验证)
            # 使用validation_folds参数控制交叉验证折数
            sss_val = StratifiedShuffleSplit(n_splits=validation_folds, test_size=0.2)
            
            # 如果是多折验证，返回训练集和验证集的列表
            if validation_folds > 1:
                train_datasets = []
                val_datasets = []
                for train_idx, val_idx in sss_val.split(train_data, train_label):
                    # 创建训练集和验证集
                    train_getter = itemgetter(*train_idx) if len(train_idx) > 1 else itemgetter(train_idx[0])
                    val_getter = itemgetter(*val_idx) if len(val_idx) > 1 else itemgetter(val_idx[0])
                    
                    # 创建训练集
                    train_dataset = Mydata(
                        train_getter(train_data) if len(train_idx) > 1 else [train_data[train_idx[0]]], 
                        train_getter(train_label) if len(train_idx) > 1 else [train_label[train_idx[0]]], 
                        transform_train
                    )
                    
                    # 创建验证集
                    val_dataset = Mydata(
                        val_getter(train_data) if len(val_idx) > 1 else [train_data[val_idx[0]]], 
                        val_getter(train_label) if len(val_idx) > 1 else [train_label[val_idx[0]]], 
                        transform_test  # 验证集使用测试变换
                    )
                    
                    train_datasets.append(train_dataset)
                    val_datasets.append(val_dataset)
                
                # 返回训练集列表、验证集列表、完整训练集和测试集
                return train_datasets, val_datasets, traintest_dataset, test_dataset, resize_size, transform_train
            else:
                # 单折验证，保持原有逻辑
                train_idx, val_idx = next(sss_val.split(train_data, train_label))
                
                # 创建训练集和验证集
                train_getter = itemgetter(*train_idx) if len(train_idx) > 1 else itemgetter(train_idx[0])
                val_getter = itemgetter(*val_idx) if len(val_idx) > 1 else itemgetter(val_idx[0])
                
                # 更新训练集为分割后的训练部分
                split_train_dataset = Mydata(
                    train_getter(data) if len(train_idx) > 1 else [train_data[train_idx[0]]], 
                    train_getter(label) if len(train_idx) > 1 else [train_label[train_idx[0]]], 
                    transform_train
                )
                
                # 创建验证集
                val_dataset = Mydata(
                    val_getter(data) if len(val_idx) > 1 else [train_data[val_idx[0]]], 
                    val_getter(label) if len(val_idx) > 1 else [train_label[val_idx[0]]], 
                    transform_test  # 验证集使用测试变换
                )
                
                # 返回训练集、验证集、完整训练集和测试集
                return split_train_dataset, val_dataset, traintest_dataset, test_dataset, resize_size, transform_train

    if 'chestct' in dataset:
        data, label = [], []
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','train')
        label_type = ['adenocarcinoma','large','normal','squamous']
        for obj in all_folders.rglob('*.png'):  
            data.append(str(obj))
            for i, label_name in enumerate(label_type):
                if label_name in str(obj.parent):
                    label.append(i)
                    
        # 创建完整训练集
        traintest_dataset = Mydata(data, label, transform_train)
        
        # 如果需要验证集，则从训练集中分出一部分作为验证集
        if validation:
            # 使用分层抽样将训练集分为训练集和验证集 (80%训练, 20%验证)
            # 使用validation_folds参数控制交叉验证折数
            sss = StratifiedShuffleSplit(n_splits=validation_folds, test_size=0.2)
            
            # 如果是多折验证，返回训练集和验证集的列表
            if validation_folds > 1:
                train_datasets = []
                val_datasets = []
                for train_idx, val_idx in sss.split(data, label):
                    # 创建训练集和验证集
                    train_getter = itemgetter(*train_idx) if len(train_idx) > 1 else itemgetter(train_idx[0])
                    val_getter = itemgetter(*val_idx) if len(val_idx) > 1 else itemgetter(val_idx[0])
                    
                    # 创建训练集
                    train_dataset = Mydata(
                        train_getter(data) if len(train_idx) > 1 else [data[train_idx[0]]], 
                        train_getter(label) if len(train_idx) > 1 else [label[train_idx[0]]], 
                        transform_train
                    )
                    
                    # 创建验证集
                    val_dataset = Mydata(
                        val_getter(data) if len(val_idx) > 1 else [data[val_idx[0]]], 
                        val_getter(label) if len(val_idx) > 1 else [label[val_idx[0]]], 
                        transform_test  # 验证集使用测试变换
                    )
                    
                    train_datasets.append(train_dataset)
                    val_datasets.append(val_dataset)
                
                # 处理测试集
                data_test, label_test = [], []
                all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','test')
                for obj in all_folders.rglob('*.png'):  
                    data_test.append(str(obj))
                    for i, label_name in enumerate(label_type):
                        if label_name in str(obj.parent):
                            label_test.append(i)
                test_dataset = Mydata(data_test, label_test, transform_test)
                
                # 返回训练集列表、验证集列表、完整训练集和测试集
                return train_datasets, val_datasets, traintest_dataset, test_dataset, resize_size, transform_train
            else:
                # 单折验证，保持原有逻辑
                train_idx, val_idx = next(sss.split(data, label))
                
                # 创建训练集和验证集
                train_getter = itemgetter(*train_idx) if len(train_idx) > 1 else itemgetter(train_idx[0])
                val_getter = itemgetter(*val_idx) if len(val_idx) > 1 else itemgetter(val_idx[0])
                
                # 创建训练集
                train_dataset = Mydata(
                    train_getter(data) if len(train_idx) > 1 else [data[train_idx[0]]], 
                    train_getter(label) if len(train_idx) > 1 else [label[train_idx[0]]], 
                    transform_train
                )
                
                # 创建验证集
                val_dataset = Mydata(
                    val_getter(data) if len(val_idx) > 1 else [data[val_idx[0]]], 
                    val_getter(label) if len(val_idx) > 1 else [label[val_idx[0]]], 
                    transform_test  # 验证集使用测试变换
                )
                
                # 处理测试集
                data_test, label_test = [], []
                all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','test')
                for obj in all_folders.rglob('*.png'):  
                    data_test.append(str(obj))
                    for i, label_name in enumerate(label_type):
                        if label_name in str(obj.parent):
                            label_test.append(i)
                test_dataset = Mydata(data_test, label_test, transform_test)
                
                # 返回训练集、验证集、完整训练集和测试集
                return train_dataset, val_dataset, traintest_dataset, test_dataset, resize_size, transform_train
        else:
            data_test, label_test = [], []
            all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','test')
            for obj in all_folders.rglob('*.png'):  
                data_test.append(str(obj))
                for i, label_name in enumerate(label_type):
                    if label_name in str(obj.parent):
                        label_test.append(i)
            test_dataset = Mydata(data_test, label_test, transform_test)

    
    return traintest_dataset,test_dataset,resize_size,transform_train