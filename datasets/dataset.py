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
            all_transforms = self.transform[self.groups[idx]]
            transform = all_transforms[random.randint(0, len(all_transforms))]
        else:
            transform = self.transform
        image = transform(image)
        label = self.labels[idx]
        return image, label

    def get_all_files(self):
        return [Image.open(self.full_filenames[idx]) for idx in range(len(self.full_filenames))]
    
    def get_labels(self):
        return self.labels
    
    def update_transform(self, transform, groups, mfc):
        self.transform = transform
        self.mfc = mfc
        self.groups = groups

class MyDataloader:
    def __init__(self,dataset,batch_size=1,shuffle=False,num_workers=1):
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.current_idx = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.indices):
            self.current_idx = 0
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx+self.batch_size]        

        # with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        #     futures = [executor.submit(self.load_sample, idx) for idx in batch_indices]
        #     results = [future.result() for future in as_completed(futures)]
        # data_batch, label_batch = zip(*results)
        batch = [self.dataset[idx] for idx in batch_indices]
        data_batch, label_batch = zip(*batch)
        
        self.current_idx += self.batch_size
        return torch.stack(data_batch), torch.tensor(label_batch)
    
    def load_sample(self,idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)