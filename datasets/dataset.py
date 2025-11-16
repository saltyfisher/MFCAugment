import cv2
import numpy as np
import torch
import random
import torchvision

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from PIL import Image

class Mydata(Dataset):
    def __init__(self, file_list, labels, transform = None, mfc=False):
        self.full_filenames = file_list
        self.labels = labels
        self.transform = transform
        self.mfc = False
        self.groups = []
    
    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        img_name = self.full_filenames[idx]
        image = Image.open(img_name).convert('RGB')
        # print(idx)
        if idx in self.groups:
            image = self.mfc_transform[0](image)
        else:
            image = self.transform[0](image)
        label = self.labels[idx]
        return image, label

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