import numpy as np
import os
import random
import copy
import time
import cv2
import multiprocessing as mp
import torch
import time
import joblib
# import geatpy as ea

from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from core.FeatureExtractor import FeatureExtractor
from core.augmentations import MyAugment, augmentation_space
from core.utils import KL_loss_all, KL_loss_cycle
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from scipy.spatial import distance
from EA.MFPSO import MFPSO
from dataclasses import dataclass

class SingleTask(object):
    def __init__(self,dim,Lb,Ub,encode,fnc):
        self.dim = dim
        self.Lb = Lb
        self.Ub = Ub
        self.encode = encode
        self.evalfnc = fnc

def evalFunc(policy, group_id, params):
    feat_extractor = params['feat_extracor']
    data_list = params['data_list']
    groups = params['groups']
    scaler = params['scaler']
    gm = params['gm']
    w = params['w']
    aug = MyAugment(policy)
    aug_imgs = [aug(data_list[i]) for i in list(np.argwhere(groups == group_id))]
    aug_feat = feat_extractor(aug_imgs)
    aug_feat = scaler.transform(aug_feat)
    aug_mean = np.mean(aug_feat, axis=0)
    aug_cov = np.cov(aug_feat, rowvar=False)
    loss = KL_loss_all(aug_mean, aug_cov, gm.means_, gm.covariances_, group_id, w)
    loss = np.sum(np.hstack(loss))
    return loss, policy
    
def MFCAugment(dataset, gaussian_component=3,n_op=5,parameter_max=31):
    total_op_num = len(augmentation_space())
    ###提取特征###
    data_list = dataset.all_files
    data_list = [np.array(d) for d in data_list]
    feat_extractor = FeatureExtractor()
    feat_list = feat_extractor(data_list)
    scaler = MinMaxScaler()
    feat_list = scaler.fit_transform(feat_list)
    gm = GaussianMixture(n_components=gaussian_component)
    groups = gm.fit_predict(feat_list)
    w = distance.squareform(distance.pdist(gm.means_))
    w = 1/(w/np.linalg.norm(w, axis=1) + np.eye(w.shape[0]))
    ###定义子任务###
    Lb = [0]*(n_op*3)
    Ub = ([total_op_num-1]*(n_op)).extend([parameter_max]*n_op).extend([parameter_max]*n_op)
    tasks = [SingleTask(n_op*3,Lb,Ub,[0]*(n_op*3)) for _ in range(np.max(groups),evalFunc)]
    params = {'feat_extractor':feat_extractor, 'data_list':data_list, 'groups':groups, 'scaler':scaler, 'gm':gm, 'w':w}
    MFPSO(tasks, 100, 50, 0.3, 0, 20, params)
    











