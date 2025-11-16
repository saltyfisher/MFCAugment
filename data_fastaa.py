import logging
import os
import random
import torchvision
import torch

from torchvision import transforms
from operator import itemgetter
from pathlib import Path
from PIL import Image
from collections import Counter
from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset, Subset
from torchvision.transforms import transforms, TrivialAugmentWide, AutoAugment, RandAugment
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from theconf import Config as C
from datasets.dataset import Mydata, MyDataloader

def get_data(dataset,dataroot,batch,split_ratio=0.4,validation=False):
    if dataset == 'lc25000':
        transform_train = transforms.Compose([
            transforms.Resize(C.get()['img_size'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            ])
    if dataset == 'lymphoma':
        transform_train = transforms.Compose([
            transforms.Resize(C.get()['img_size'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            ])
    if 'breakhis' in dataset:
        transform_train = transforms.Compose([
            transforms.Resize(C.get()['img_size'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
   
    if C.get()['aug'] == 'trivialaugment':
        transform_train.transforms.insert(0, TrivialAugmentWide())
    elif C.get()['aug'] == 'randaugment':
        transform_train.transforms.insert(0, RandAugment())
    elif C.get()['aug'] == 'none':
        pass
    
    label, data = [], []
    if dataset == 'lymphoma':
        types = ['CLL','FL','MCL']        
        for i, x in enumerate(types):
            files = list(Path(dataroot).joinpath('lymphoma',x).glob('*.tif'))
            data.extend(list(map(str, files)))
            label.extend(len(files)*[i])
        sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
        traintest_idx, test_idx = next(sss.split(data, label))
        getter = itemgetter(*test_idx)
        test_dataset = Mydata(getter(data), getter(label), transform_test)
        getter = itemgetter(*traintest_idx)
        traintest_dataset = Mydata(getter(data), getter(label), transform_train)
        
        sss = StratifiedShuffleSplit(n_splits=5,test_size=split_ratio)
        trainval_idx, val_idx = next(sss.split(getter(data), getter(label)))
        getters = [itemgetter(*idx) for idx in trainval_idx]
        trainval_datasets = [Mydata(getter(data), getter(label), transform_train) for getter in getters]
        getters = [itemgetter(*idx) for idx in val_idx]
        val_datasets = [Mydata(getter(data), getter(label), transform_test) for getter in getters]

    if dataset == 'lc25000':
        with open(str(Path(dataroot).joinpath('LC25000','train_list.txt')), "r") as file:
            content = file.readlines()
            for c in content:
                d, l = c.strip('\n').split(' ')
                data.append(dataroot+'/LC25000/'+d)
                label.append(int(l))
        traintest_dataset = Mydata(data, label, transform_train)
        sss = StratifiedShuffleSplit(n_splits=5,test_size=split_ratio)
        traintest_idx, val_idx = next(sss.split(data, label))
        getters = [itemgetter(*idx) for idx in trainval_idx]
        trainval_datasets = [Mydata(getter(data), getter(label), transform_train) for getter in getters]
        getters = [itemgetter(*idx) for idx in val_idx]
        val_datasets = [Mydata(getter(data), getter(label), transform_test) for getter in getters]
        data, label = [], []
        with open(str(Path(dataroot).joinpath('LC25000','val_list.txt')), "r") as file:
            content = file.readlines()
            for c in content:
                d, l = c.strip('\n').split(' ')
                data.append(dataroot+'/LC25000/'+d)
                label.append(int(l))
        test_dataset = Mydata(data, label, transform_test)
    
    if 'breakhis' in dataset:
        all_folders = Path(dataroot).joinpath('BreakHis','histology_slides','breast')
        strs = dataset.split('breakhis')
        c_num = strs[1][0]
        if c_num == '2':
            label_type = ['benign','malignant']
        if c_num == '8':
            label_type = ['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma','ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma']
        mag = strs[1][1:]
        for obj in all_folders.rglob('*.png'):  
            if mag in str(obj.parent):
                data.append(str(obj))
                for i, label_name in enumerate(label_type):
                    if label_name in str(obj.parent):
                        label.append(i)
        
        sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
        traintest_idx, test_idx = next(sss.split(data, label))
        getter = itemgetter(*test_idx)
        test_dataset = Mydata(getter(data), getter(label), transform_test)
        getter = itemgetter(*traintest_idx)
        traintest_dataset = Mydata(getter(data), getter(label), transform_train)
        
        sss = StratifiedShuffleSplit(n_splits=5,test_size=split_ratio)
        trainval_idx = list(sss.split(getter(data), getter(label)))
        getters = [itemgetter(*idx[0]) for idx in trainval_idx]
        trainval_datasets = [Mydata(getter(data), getter(label), transform_train) for getter in getters]
        getters = [itemgetter(*idx[1]) for idx in trainval_idx]
        val_datasets = [Mydata(getter(data), getter(label), transform_test) for getter in getters]

    traintestloader = MyDataloader(traintest_dataset, batch_size=batch, shuffle=True, num_workers=4)
    trainvalloaders = [MyDataloader(d, batch_size=batch, shuffle=True, num_workers=4) for d in trainval_datasets]
    
    if 'breakhis' in dataset:
        testloader = MyDataloader(test_dataset, batch_size=1, shuffle=False)
        valdloaders = [MyDataloader(d, batch_size=1, shuffle=False) for d in val_datasets]
    else:
        testloader = MyDataloader(test_dataset, batch_size=8, shuffle=False)
        valdloaders = [MyDataloader(d, batch_size=8, shuffle=False) for d in val_datasets]
        
    return traintestloader,testloader,trainvalloaders,valdloaders,traintest_dataset,trainval_datasets,val_datasets,test_dataset

def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0, distributed=False, started_with_spawn=False, summary_writer=None):
    if dataset == 'lc25000':
        transform_train = transforms.Compose([
            transforms.Resize(C.get()['img_size'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            ])
    if dataset == 'lymphoma':
        transform_train = transforms.Compose([
            transforms.Resize(C.get()['img_size'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            ])
    if 'breakhis' in dataset:
        transform_train = transforms.Compose([
            transforms.Resize(C.get()['img_size'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
   
    if C.get()['aug'] == 'trivialaugment':
        transform_train.transforms.insert(0, TrivialAugmentWide())
    elif C.get()['aug'] == 'randaugment':
        transform_train.transforms.insert(0, RandAugment())
    elif C.get()['aug'] == 'none':
        pass
    
    traintest_dataset, trainval_dataset, val_dataset, test_dataset = get_data(C.get()['dataset'],dataroot,transform_train,transform_test)

    traintestloader = torch.utils.data.DataLoader(traintest_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    trainvalloader = torch.utils.data.DataLoader(traintest_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    if 'breakhis' in dataset:
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    else:
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    return traintestloader, trainvalloader, validloader, testloader, traintest_dataset


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)







