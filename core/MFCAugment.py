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
from core.utils import KL_loss_all
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from scipy.spatial import distance
from EA.MFPSO import MFPSO
from dataclasses import dataclass
from core.utils import get_deepfeat

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
    feat_list = params['feat_list']
    groups = params['groups']
    pca = params['pca']
    w = params['w']
    group_id = params['task_id']
    Lb = params['Lb']
    Ub = params['Ub']
    args = params['args']
    config = params['config']
    policy = np.floor(policy*(Ub-Lb)+Lb)
    aug = MyAugment(policy,num_ops=params['n_op'])
    aug_imgs = [aug((data_list[i])) for i in list(np.argwhere(groups == group_id).flatten())]
    aug_feat = [get_deepfeat(config['model'],feat_extractor,ToTensor()(d).unsqueeze(0).to(args.device)) for d in aug_imgs]
    aug_feat = pca.transform(aug_feat)
    target_imgs = [feat_list[i] for i in list(np.argwhere(groups != group_id).flatten())]
    loss = KL_loss_all(target_imgs, aug_feat, group_id, w)
    loss = np.sum(np.hstack(loss))
    return loss
    
def MFCAugment(model, config, data_list, args, n_clusters=3):
    mag_bin = args.mag_bin
    prob_bin = args.prob_bin
    n_op = args.num_op
    total_op_num = len(augmentation_space())
    ###提取特征###
    feat_list = [get_deepfeat(config['model'],model,ToTensor()(d).unsqueeze(0).to(args.device)) for d in data_list]
    pca = PCA(n_components=int(0.05*feat_list[0].shape[0]))
    feat_list = pca.fit_transform(feat_list)
    kmeans = KMeans(n_clusters=n_clusters)
    groups = kmeans.fit_predict(feat_list)
    centers = kmeans.cluster_centers_
    w = distance.squareform(distance.pdist(centers))
    w = 1 - w/np.linalg.norm(w, axis=1)
    ###定义子任务###
    Lb = np.array([0]*(n_op*3))
    Ub = np.array(([total_op_num-1]*(n_op)) + ([mag_bin-1]*n_op) + ([prob_bin-1]*n_op))
    tasks = [SingleTask(n_op*3,Lb,Ub,[0]*(n_op*3), evalFunc) for _ in range(np.max(groups)+1)]
    params = {'feat_extractor':model, 'data_list':data_list, 'feat_list':feat_list, 'groups':groups, 'pca':pca, 'w':w,'n_op':n_op,'mag_bin':mag_bin,'prob_bin':prob_bin,'args':args,'config':config}
    options = {'popsize':50,'maxgen':10,'rmp':0.3,'reps':2}
    bestPop = MFPSO(tasks, options, params)
    bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb)] for i in range(options['reps']) for j in range(options['popsize'])]

    return bestPolicy
    











