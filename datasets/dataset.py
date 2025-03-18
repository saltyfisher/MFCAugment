import cv2
import numpy as np
import torch
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from PIL import Image

class Mydata(Dataset):
    def __init__(self, file_list, labels, transform = None, mfc=False):
        self.full_filenames = file_list
        self.labels = labels
        self.transform = transform
        self.mfc = False
    
    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        if self.mfc:
            t_idx = None
            for i, group in enumerate(self.groups):
                if idx in group:
                    t_idx = i
                    break
            all_transforms = self.transform[t_idx]
            transform = all_transforms[random.randint(0, len(all_transforms)-1)]
        else:
            transform = self.transform
        image = transform(image)
        label = self.labels[idx]
        return image, label

    def get_all_files(self):
        data_list = [Image.open(self.full_filenames[idx]) for idx in range(len(self.full_filenames))]
        label_list = [self.labels[idx] for idx in range(len(self.full_filenames))]
        return data_list, label_list
    
    def get_labels(self):
        return self.labels
    
    def update_transform(self, transform, groups, mfc):
        self.transform = transform
        self.mfc = mfc
        self.groups = groups