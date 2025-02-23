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
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from core.FeatureExtractor import FeatureExtractor
from core.augmentations import MyAugment, augmentation_space
from core.utils import KL_loss_all, KL_loss_cycle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from scipy.spatial import distance
from EA.MFPSO import MFPSO
from dataclasses import dataclass

class SingleTask(object):
    def __init__(self,dim:int,Lb,Ub,encode:list[int],fnc):
        self.dims = dim
        self.Lb = Lb
        self.Ub = Ub
        self.encode = encode
        self.evalfnc = fnc

    def evaluate(self, x, params):
        params.update({'Lb':self.Lb,'Ub':self.Ub})
        return self.evalfnc(x, params)
    
def evalFunc(policy, params):
    feat_extractor = params['feat_extractor']
    data_list = params['data_list']
    groups = params['groups']
    scaler = params['scaler']
    pca = params['pca']
    gm = params['gm']
    w = params['w']
    group_id = params['task_id']
    Lb = params['Lb']
    Ub = params['Ub']
    policy = np.floor(policy*(Ub-Lb)+Lb)
    aug = MyAugment(policy,num_ops=params['n_op'])
    data_list = torch.from_numpy(np.array(data_list))
    aug_imgs = [aug(torch.from_numpy(data_list[i]).permute(2,0,1)).numpy().astype(np.uint8).transpose(1,2,0) for i in list(np.argwhere(groups == group_id).flatten())]
    aug_feat = feat_extractor(aug_imgs)
    aug_feat = scaler.transform(aug_feat)
    aug_feat = pca.transform(aug_feat)
    aug_mean = np.mean(aug_feat, axis=0)
    aug_cov = np.cov(aug_feat, rowvar=False)
    loss = KL_loss_all(aug_mean, aug_cov, gm.means_, gm.covariances_, group_id, w)
    loss = np.sum(np.hstack(loss))
    return loss
    
def MFCAugment(dataset, args, gaussian_component=3):
    mag_bin = args.mag_bin
    prob_bin = args.prob_bin
    n_op = args.num_op
    total_op_num = len(augmentation_space())
    ###提取特征###
    data_list = dataset.get_all_files()
    data_list = [np.array(d) for d in data_list]
    feat_extractor = FeatureExtractor()
    feat_list = feat_extractor(data_list)
    scaler = StandardScaler()
    feat_list = scaler.fit_transform(feat_list)
    pca = PCA(n_components=int(0.1*feat_list.shape[1]))
    feat_list = pca.fit_transform(feat_list)
    gm = GaussianMixture(n_components=gaussian_component)
    groups = gm.fit_predict(feat_list)
    w = distance.squareform(distance.pdist(gm.means_))
    w = 1 - w/np.linalg.norm(w, axis=1)
    ###定义子任务###
    Lb = np.array([0]*(n_op*3))
    Ub = np.array(([total_op_num-1]*(n_op)) + ([mag_bin-1]*n_op) + ([prob_bin-1]*n_op))
    tasks = [SingleTask(n_op*3,Lb,Ub,[0]*(n_op*3), evalFunc) for _ in range(np.max(groups)+1)]
    params = {'feat_extractor':feat_extractor, 'data_list':data_list, 'groups':groups, 'scaler':scaler, 'pca':pca, 'gm':gm, 'w':w,'n_op':n_op,'mag_bin':mag_bin,'prob_bin':prob_bin}
    options = {'popsize':50,'maxgen':10,'rmp':0.3,'reps':2}
    bestPop = MFPSO(tasks, options, params)
    bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb)] for i in range(options['reps']) for j in range(options['popsize'])]

    return bestPolicy
    











