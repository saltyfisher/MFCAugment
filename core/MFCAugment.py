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
import cvxpy as cp
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
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from ortools.linear_solver import pywraplp
from EA.MFPSO import MFPSO
from dataclasses import dataclass
from core.utils import get_deepfeat

def constrained_kmeans_ilp(X, k, max_iters=100):
    n_samples, _ = X.shape
    cluster_size = n_samples // k
    L = round(cluster_size - cluster_size*0.3)
    U = round(cluster_size + cluster_size*0.3)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    kmeans = KMeans(n_clusters=k)
    groups = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    # 初始化簇中心（例如通过 K-means++）
    # centers = X[np.random.choice(n_samples, k, replace=False)]
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2).astype(np.double)
    
    # 定义变量
    x = {}
    for i in range(n_samples):
        for j in range(k):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
    
    # 约束1：每个样本必须分配到一个簇
    for i in range(n_samples):
        solver.Add(sum(x[i, j] for j in range(k)) == 1)
    
    # 约束2：每个簇的样本数在 [L, U] 范围内
    for j in range(k):
        solver.Add(sum(x[i, j] for i in range(n_samples)) >= L)
        solver.Add(sum(x[i, j] for i in range(n_samples)) <= U)
    
    # 目标函数：最小化总距离
    objective = solver.Objective()
    for i in range(n_samples):
        for j in range(k):
            objective.SetCoefficient(x[i, j], distances[i, j])
    objective.SetMinimization()
    
    # 求解
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        labels = np.array([np.argmax([x[i, j].solution_value() for j in range(k)]) for i in range(n_samples)])
        centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        return labels, centers
    else:
        raise ValueError("No optimal solution found.")
    
def balanced_kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    cluster_size = n_samples // k  # 每个簇的理论样本数
    r = n_samples % k
    # 初始化中心点（如K-means++）
    indices = np.random.choice(n_samples, k, replace=False)
    centers = X[indices]
    
    for _ in range(max_iters):
        # 1. 计算样本到簇中心的距离矩阵（shape: n_samples x k）
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        repeats = [cluster_size + 1] * r + [cluster_size] * (k - r)
        # 2. 构建匈牙利算法输入矩阵（垂直复制 cluster_size 次）
        expanded_distances = np.repeat(distances, repeats, axis=1)  # shape: n_samples x (k*cluster_size)
        
        # 3. 调用匈牙利算法（行：样本，列：每个簇的虚拟槽位）
        row_ind, col_ind = linear_sum_assignment(expanded_distances.T)  # 转置为 (k*cluster_size) x n_samples
        
        # 4. 根据分配结果生成标签（每个样本被分配到哪个簇的槽位）
        labels = np.zeros(n_samples, dtype=int)
        virtual_slot_to_cluster = np.repeat(np.arange(k), repeats)  # 槽位对应的真实簇编号
        for sample_idx, slot_idx in zip(row_ind, col_ind):
            labels[sample_idx] = virtual_slot_to_cluster[slot_idx]
        
        # 5. 更新中心点
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return labels, centers

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
    batch_size = params['batch_size']
    groups = params['groups']
    sampled_groups = params['sampled_groups']
    pca = params['pca']
    w = params['w']
    group_id = params['task_id']
    Lb = params['Lb']
    Ub = params['Ub']
    args = params['args']
    config = params['config']
    policy = np.floor(policy*(Ub-Lb)+Lb)
    aug = MyAugment(policy,num_ops=params['n_op'],resize=False,resize_size=config['img_size'])
    # indices = sampled_groups[group_id].tolist()
    # aug_imgs = [ToTensor()(aug((data_list[i]))).unsqueeze(0) for i in indices]
    aug_feat = []
    data_list = data_list[group_id]
    for data in data_list:
        data = data.to(args.device)
        aug_imgs = aug(data)/255.0
        feat = [get_deepfeat(config['model'],feat_extractor,img.unsqueeze(0)) for img in aug_imgs]
        feat = torch.cat(feat, dim=0)
        aug_feat.append(feat)
    aug_feat = torch.cat(aug_feat, dim=0).cpu().numpy()
    aug_feat = pca.transform(aug_feat)
    target_imgs = feat_list[groups!=group_id]
    loss = KL_loss_all(target_imgs, aug_feat, group_id, w)
    loss = np.sum(np.hstack(loss))
    return loss
    
def MFCAugment(model, config, data_list, args, n_clusters=8, max_samples=100):
    mag_bin = args.mag_bin
    prob_bin = args.prob_bin
    n_op = args.num_op
    total_op_num = len(augmentation_space())
    ###提取特征###
    # input_list = [ToTensor()(d).unsqueeze(0) for d in data_list]
    # feat_list = [get_deepfeat(config['model'],model,d.to(args.device)) for d in input_list]
    # feat_list = torch.cat(feat_list, dim=0).numpy()
    # pca = PCA(n_components=int(0.05*feat_list[0].shape[0]))
    # feat_list = pca.fit_transform(feat_list) 
    # transformer = transforms.Compose([transforms.Resize(config['img_size'], interpolation=Image.BICUBIC), ToTensor()])
    transformer = ToTensor()
    input_list = [(transformer(d)).unsqueeze(0) for d in data_list]
    feat_list = [get_deepfeat(config['model'],model,d.to(args.device)) for d in input_list]
    feat_list = torch.cat(feat_list, dim=0).cpu().numpy()
    pca = PCA(n_components=int(0.05*feat_list[0].shape[0]))
    feat_list = pca.fit_transform(feat_list)
    groups, centers = constrained_kmeans_ilp(feat_list, n_clusters)
    # while True:
    #     kmeans = KMeans(n_clusters=n_clusters)
    #     groups = kmeans.fit_predict(feat_list)
    #     centers = kmeans.cluster_centers_
    #     unique_groups, counts = np.unique(groups, return_counts=True)
    #     if np.all(counts>1):
    #         # sampled_groups = []
    #         # for group in unique_groups:
    #         #     indices = np.where(groups == group)[0]
    #         #     if len(indices) > max_samples:
    #         #         # 计算每个点到簇中心的距离
    #         #         distances = pairwise_distances(feat_list[indices], centers[group].reshape(1, -1)).flatten()
    #         #         # 按照距离排序
    #         #         sorted_indices = indices[np.argsort(distances)]
    #         #         # 计算采样数量
    #         #         num_per_section = len(indices) // 3
    #         #         num_samples_per_section = max_samples // 3
    #         #         # 远、中、近采样
    #         #         far_indices = random.sample(sorted_indices[:num_per_section].tolist(), num_samples_per_section)
    #         #         mid_indices = random.sample(sorted_indices[num_per_section:2*num_per_section].tolist(), num_samples_per_section)
    #         #         near_indices = random.sample(sorted_indices[2*num_per_section:3*num_per_section].tolist(), num_samples_per_section)
    #         #         # 合并采样结果
    #         #         sampled_indices = np.concatenate([far_indices, mid_indices, near_indices])
    #         #     else:
    #         #         sampled_indices = indices
    #         #     sampled_groups.append(sampled_indices)
    #         break
    #     else:
    #         n_clusters -= 2
    w = distance.squareform(distance.pdist(centers))
    w = 1 - w/np.sum(w, axis=1)
    sampled_groups = [np.argwhere(groups==i).squeeze() for i in range(max(groups)+1)]   
    new_data_list = []
    batch_size = args.batch_size//8
    for group_id in range(len(sampled_groups)):
        all_batch_imgs = []
        indices = sampled_groups[group_id]
        indices = np.array_split(indices, np.arange(0, len(indices), batch_size)[1:])
        for idx in indices:
            batch_imgs = [torch.from_numpy(np.array(data_list[i])) for i in idx]
            batch_sizes = torch.tensor([i.shape[:2] for i in batch_imgs])
            batch_sizes, counts = torch.unique(batch_sizes, dim=0, return_counts=True)
            resize_size = batch_sizes[torch.argmax(counts)]
            transformer = transforms.Resize(resize_size.tolist())
            batch_imgs = [transformer(d.permute(2,0,1)).unsqueeze(0) for d in batch_imgs]
            batch_imgs = torch.cat(batch_imgs, dim=0).to(torch.uint8)
            all_batch_imgs.append(batch_imgs)
        new_data_list.append(all_batch_imgs)
    ###定义子任务###
    Lb = np.array([0]*(n_op*3))
    Ub = np.array(([total_op_num-1]*(n_op)) + ([mag_bin-1]*n_op) + ([prob_bin-1]*n_op))
    tasks = [SingleTask(n_op*3,Lb,Ub,[0]*(n_op*3), evalFunc) for _ in range(np.max(groups)+1)]
    params = {'feat_extractor':model,
              'data_list':new_data_list, 
              'feat_list':feat_list, 
              'batch_size':args.batch_size, 
              'groups':groups, 
              'sampled_groups':sampled_groups, 
              'pca':pca, 
              'w':w,'n_op':n_op,'mag_bin':mag_bin,'prob_bin':prob_bin,
              'args':args,'config':config}
    options = {'popsize':50,'maxgen':20,'rmp':0.3,'reps':2}
    bestPop = MFPSO(tasks, options, params)
    bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb)] for i in range(options['reps']) for j in range(options['popsize'])]
    skillFactor = [bestPop[i,j].skill_factor for i in range(options['reps']) for j in range(options['popsize'])]

    return bestPolicy, skillFactor, groups
    











