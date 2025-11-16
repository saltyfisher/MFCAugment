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
import pyDOE3
import pickle
import datetime
import matplotlib.pyplot as plt
# import geatpy as ea

from tensorboardX import SummaryWriter
from torch_pca import PCA as PCA_torch
from tqdm import tqdm
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from core.FeatureExtractor import FeatureExtractor
from core.augmentations import MyAugment, augmentation_space
from core.utils import KL_loss_all, KL_loss, Jensen_loss, Sinkhorn_dist, kl_divergence_kde
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from EA.SBX import SBX
from EA.MFSBX  import MFSBX
from EA.MFPSO import MFPSO
from EA.GDMFPSO import GDMFPSO
from EA.GDPMFPSO import GDPMFPSO
from dataclasses import dataclass
from core.utils import get_deepfeat, get_clsprob
from core.model import Proxy, RBFNetwork, G_D
from core.trainer_GD import train_GD, test_GD
from core.trainer_proxy import train_proxy, test_proxy
# from core.dataCluster import constrained_kmeans_ilp

def getdatafeat(args, resize_size, data_list, model):
    if args.resize:
        transformer = transforms.Compose([transforms.Resize(resize_size), ToTensor()])
        batch = 32
    else:
        transformer = ToTensor()
        batch = 1
    input_list = [(transformer(d)).unsqueeze(0) for d in data_list]
    input_list = [torch.cat(input_list[i:i+batch]) for i in range(0,len(input_list),batch)]
    feat_list = [get_deepfeat(args,model,d.to(args.device)) for d in input_list]
    cls_list = [f[1] for f in feat_list]
    feat_list = [f[0] for f in feat_list]

    return feat_list, cls_list

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

def evalFuncBatch(policies, params):
    """
    批量评估函数，将所有个体的增强数据合并后批量处理以提高效率
    """
    feat_extractor = params['feat_extractor']
    data_list = params['data_list']
    feat_list = params['feat_list']
    batch_size = params['batch_size']
    groups = params['groups']
    pca = params['pca']
    w = params['w']
    group_id = params['task_id']
    Lb = params['Lb']
    Ub = params['Ub']
    args = params['args']
    model = params['model']
    resize_size = params['resize_size']
    
    # 格式化所有策略
    formatted_policies = [formatPolicy(params, policy) for policy in policies]
    
    # 为每个策略生成增强数据
    aug_data_list = []
    data = [data_list[i] for i in groups[group_id]]
    
    for policy in formatted_policies:
        aug = MyAugment(policy[0], num_ops=params['n_op'])
        aug_data = []
        for d in data:
            aug_data.append(aug(d))
        aug_data_list.append(aug_data)
    
    # 合并所有增强数据以便批量处理
    all_aug_data = []
    policy_indices = []  # 记录每个策略的数据在合并列表中的起始和结束位置
    
    start_idx = 0
    for aug_data in aug_data_list:
        all_aug_data.extend(aug_data)
        end_idx = start_idx + len(aug_data)
        policy_indices.append((start_idx, end_idx))
        start_idx = end_idx
    
    # 批量处理所有增强数据
    aug_feat, _ = getdatafeat(args, resize_size, all_aug_data, model)
    if args.gpu:
        aug_feat = torch.cat(aug_feat).detach()
    else:
        aug_feat = torch.cat(aug_feat).detach().cpu().numpy()
    
    # 使用PCA转换特征
    aug_feat = pca.transform(aug_feat)
    
    # 拆分结果并计算每个策略的损失
    losses = []
    for start_idx, end_idx in policy_indices:
        policy_feat = aug_feat[start_idx:end_idx]
        
        st = time.time()
        if args.gpu:
            loss1 = kl_divergence_kde(feat_list, policy_feat)
            loss2 = kl_divergence_kde(feat_list[groups[group_id]], policy_feat)
        else:
            loss1 = KL_loss(feat_list, policy_feat)
            loss2 = KL_loss(feat_list[groups[group_id]], policy_feat)
        
        l = 1
        loss = loss1 - l * loss2
        losses.append(loss)
    
    return losses

def evalFunc(policy, params):
    feat_extractor = params['feat_extractor']
    data_list = params['data_list']
    feat_list = params['feat_list']
    batch_size = params['batch_size']
    groups = params['groups']
    pca = params['pca']
    w = params['w']
    group_id = params['task_id']
    Lb = params['Lb']
    Ub = params['Ub']
    args = params['args']
    model = params['model']
    resize_size = params['resize_size']
    # policy = np.floor(policy*(Ub-Lb)+Lb)
    policy = formatPolicy(params, policy)
    aug = MyAugment(policy[0],num_ops=params['n_op'])
    aug_data = []
    data = [data_list[i] for i in groups[group_id]]
    for d in data:
        aug_data.append(aug(d))
    aug_feat, _ = getdatafeat(args,resize_size,aug_data,model)
    if args.gpu:
        aug_feat = torch.cat(aug_feat).detach()
    else:
        aug_feat = torch.cat(aug_feat).detach().cpu().numpy()
    # if args.resize:
    #     aug_feat = get_deepfeat(args, config['model'],feat_extractor, aug_imgs).cpu().numpy()
    # else:
    #         feat = [get_deepfeat(args, config['model'],feat_extractor,img.unsqueeze(0)) for img in aug_imgs]
    #         feat = torch.cat(feat, dim=0)
    #         aug_feat.append(feat)
    #     aug_feat = torch.cat(aug_feat, dim=0).cpu().numpy()
    aug_feat = pca.transform(aug_feat)
    st = time.time()
    if args.gpu:
        # loss1 = Sinkhorn_dist(feat_list, aug_feat)
        # loss2 = Sinkhorn_dist(feat_list[groups[group_id]], aug_feat)
        loss1 = kl_divergence_kde(feat_list, aug_feat)
        loss2 = kl_divergence_kde(feat_list[groups[group_id]], aug_feat)
    else:
        loss1 = KL_loss(feat_list, aug_feat)
        loss2 = KL_loss(feat_list[groups[group_id]], aug_feat)
    # print(time.time()-st)
    # if 'Chest' in args.save_name:
    #     loss1 = Sinkhorn_dist(feat_list, aug_feat)
    #     loss2 = Sinkhorn_dist(feat_list[groups[group_id]], aug_feat)
    # else:
    #     loss1 = KL_loss(feat_list, aug_feat)
    #     loss2 = KL_loss(feat_list[groups[group_id]], aug_feat)
    l = 1
    # if 'Chest' in args.save_name:
    #     l = 10
    # else:
    #     l = 1
    # loss3 = 0
    # for i in range(len(groups)):
    #     if i != group_id:
    #         loss3 += KL_loss(feat_list[groups[i]], aug_feat)
    # loss = loss1 - l*loss2 + loss3
    loss = loss1 - l*loss2
    # print(loss)
    # loss = loss2
    return loss

def evalFuncProxy(policy, params):
    group_id = params['task_id']
    proxies = params['proxies']

    p = proxies[group_id]
    p.eval()
    with torch.no_grad():
        policy = torch.FloatTensor(policy).to(params['args'].device).unsqueeze(0)
        loss = p(policy)
        loss = loss.item()
    return loss

def generate_proxy_data(params,sample_num=500, sampled_data=None):
    # 生成训练数据
    print('Collecting data')
    dataset = params['args'].dataset
    input_channels = len(params['Lb'])
    if sampled_data is not None:
        sampled_policies = sampled_data['sampled_policies']
        labels = sampled_data['labels']
        task_id = sampled_data['task_id']
    else:
        sampled_policies = [np.random.rand(sample_num, input_channels) for _ in range(len(params['groups']))]
        # sampled_policies = [pyDOE3.lhs(input_channels, sample_num) for _ in range(len(params['groups']))]
        labels = []
        task_id = []
        # 计算适应值函数值
        with joblib.Parallel(n_jobs=10,backend='threading') as parallel:
            for i, policies in enumerate(sampled_policies):
                params['task_id'] = i
                results = parallel(joblib.delayed(evalFunc)(policies[j], params) for j in range(sample_num))
                labels.append(results)
                task_id.append([i]*sample_num)
        result = {'sampled_policies':np.concatenate(sampled_policies), 'labels':np.concatenate(labels), 'task_id':np.concatenate(task_id), 'groups':params['groups'], 'centers':params['centers']}
        with open(f'./training_data_{dataset}_{sample_num}.pkl', 'wb') as f:
            pickle.dump(result, f)
    return sampled_policies, labels, task_id

def cluster_data(feat_list, label_list, n_clusters):
    skf = StratifiedShuffleSplit(n_splits=n_clusters,test_size=1-1/n_clusters)
    groups = []
    for i, (train_index, test_index) in enumerate(skf.split(feat_list, label_list)):
        groups.append(train_index)
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    return groups, centers

def cluster_data_weighted(feat_list, label_list, n_clusters):
    groups = []
    sample_num = int(feat_list.shape[0]/n_clusters)
    label_list = np.array(label_list)
    if isinstance(feat_list, torch.Tensor):
        feat_list = feat_list.cpu().numpy()
    weights = -np.log(feat_list[np.arange(feat_list.shape[0]),label_list]+1e-10)*np.sum(-feat_list * np.where(feat_list > 0, np.log(feat_list), 0), axis=1)
    weights = np.nan_to_num(weights, nan=0.0, neginf=0)
    mu = np.argwhere(np.cumsum(weights / np.sum(weights))-np.random.rand() <= 0)[-1]
    mu = weights[mu]
    weights = 1 / (np.sqrt(2 * np.pi)) * np.exp(- (weights - mu) ** 2 / 2)
    weights = weights / np.sum(weights)
    # _, idx = np.unique(label_list, return_index=True)
    # weights = softmax(1 - softmax(feat_list, axis=1), axis=1)
    # weights = [softmax(weights[idx[i], i]) for i in range(len(idx))]
    # _, counts = np.unique(label_list, return_counts=True)
    # counts = counts/np.sum(counts)
    # sample_num = counts*sample_num
    # groups = []
    for _ in range(n_clusters):
        # groups.append(np.concatenate([np.random.choice(idx[i], int(sample_num[i]), p=weights[i]) for i in range(len(idx))]))
        groups.append(np.random.choice(np.arange(label_list.shape[0]), int(sample_num), p=weights))
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    return groups, centers
def MFCAugment(model, resize_size, data_list, label_list, args, n_clusters, mag_bin=31, prob_bin=10, num_ops=3, max_samples=100):
    total_op_num = len(augmentation_space())
    sample_num = 500
    ###提取特征### 
    feat_list, cls_list = getdatafeat(args, resize_size, data_list, model)
    if args.gpu:
        feat_list = torch.cat(feat_list, dim=0)
        cls_list = torch.cat(cls_list, dim=0)
    else:
        feat_list = torch.cat(feat_list, dim=0).cpu().numpy()
        cls_list = torch.cat(cls_list, dim=0).cpu().numpy()
    groups, centers = cluster_data_weighted(cls_list, label_list, n_clusters)    
    groups = [np.unique(np.concatenate(groups))]
    centers = []
    # centers = np.mean(feat_list[groups[0]], axis=0)
    # breakhis:0.05
    # plt.figure()
    # pca = TSNE(n_components=2)
    # feat_list = pca.fit_transform(feat_list)
    # tsne_f = pca.fit_transform(feat_list)
    # plt.subplot(2,n_clusters+1,1)
    # plt.scatter(tsne_f[:,0], tsne_f[:,1], c=label_list, s=10)
    # plt.title('tsne')
    # pca = PCA(n_components=2)
    # pca_f = pca.fit_transform(feat_list)
    # plt.subplot(2,n_clusters+1,n_clusters+2)
    # plt.scatter(pca_f[:,0], pca_f[:,1], c=label_list, s=10)
    # plt.title('pca')
    # pca = KernelPCA(n_components=int(0.01*feat_list[0].shape[0]), kernel='rbf')
    if args.gpu:
        if 'breakhis' in args.dataset:
            pca = PCA_torch(n_components=int(0.05*feat_list[0].shape[0]))
        else:
            pca = PCA_torch(n_components=int(0.01*feat_list[0].shape[0]))
        feat_list = pca.fit_transform(feat_list)
    else:
        if 'breakhis' in args.dataset:
            pca = PCA(n_components=int(0.05*feat_list[0].shape[0]))
        else:
            pca = PCA(n_components=int(0.01*feat_list[0].shape[0]))
        feat_list = pca.fit_transform(feat_list)
    # if os.path.exists(f'./training_data_{args.dataset}_{sample_num}.pkl'):
    #     with open(f'./training_data_{args.dataset}_{sample_num}.pkl', 'rb') as f:    
        # groups, centers = cluster_data(feat_list, label_list, n_clusters)
    # all_f = [tsne_f, pca_f]
    # all_f = [feat_list]
    # for j in range(len(all_f)):
    #     f = all_f[j]
    #     for i in range(n_clusters):
    #         plt.subplot(len(all_f),n_clusters+1,j*(n_clusters+1)+i+2)
    #         plt.scatter(f[groups[i],0], f[groups[i],1], c=np.array(label_list)[groups[i]], s=10)
    # plt.savefig('./breakhis840X_groups_kernel_pca.png')
    w = 0
    # w = distance.squareform(distance.pdist(centers))
    # w = 1 - w/np.sum(w, axis=1)
    # new_data_list = []

    # if args.resize:
    #     transformer = transforms.Compose([transforms.Resize(resize_size), ToTensor()])
    #     for indices in groups:
    #         new_data_list.append(torch.cat([transformer(data_list[i]).unsqueeze(0)*255 for i in indices], dim=0).to(torch.uint8))
    # else:
    #     batch_size = 1
    #     for indices in groups:
    #         all_batch_imgs = []
    #         indices = np.array_split(indices, np.arange(0, len(indices), batch_size)[1:])
    #         for idx in indices:
    #             batch_imgs = [data_list[i] for i in idx]
    #             # batch_sizes = torch.tensor([i.shape[1:] for i in batch_imgs])
    #             # batch_sizes, counts = torch.unique(batch_sizes, dim=0, return_counts=True)
    #             # resize_size = batch_sizes[torch.argmax(counts)]
    #             # transformer = transforms.Resize(resize_size.tolist())
    #             # transformer = transforms.Compose([transforms.Resize(config['img_size']), ToTensor()])
    #             transformer = transforms.Compose([ToTensor()])
    #             batch_imgs = [transformer(d).unsqueeze(0)*255 for d in batch_imgs]
    #             batch_imgs = torch.cat(batch_imgs, dim=0).to(torch.uint8)
    #             all_batch_imgs.append(batch_imgs)
    #         new_data_list.append(all_batch_imgs)
    # ###定义子任务###
    if args.use_prob:
        n_dims = 3
    else:
        n_dims = 2
    var_dim = num_ops*n_dims
    Lb = np.array([0]*var_dim)
    Ub = [total_op_num-1]*(num_ops) + [mag_bin-1]*num_ops
    if args.use_prob:
        Ub = Ub  + [prob_bin-1]*num_ops
    Ub = np.array(Ub)
    # if args.proxy:
    #     tasks = [SingleTask(n_op*3,Lb,Ub,[0]*var_dim, evalFuncProxy) for _ in range(len(groups))]
    # else:
    #     tasks = [SingleTask(n_op*3,Lb,Ub,[0]*var_dim, evalFunc) for _ in range(len(groups))]
    tasks = [SingleTask(num_ops*n_dims,Lb,Ub,[0]*var_dim, evalFunc) for _ in range(len(groups))]
    params = {'feat_extractor':model,
              'data_list':data_list, 
              'feat_list':feat_list, 
              'batch_size':args.batch_size, 
              'groups':groups, 
              'centers':centers,
              'pca':pca, 
              'Lb':Lb, 'Ub':Ub,
              'w':w,'n_op':num_ops,'mag_bin':mag_bin,'prob_bin':prob_bin,
              'args':args,
              'resize_size':resize_size,
              'model':model,
              }
    options = {'popsize':50,'maxgen':50,'rmp':0.3,'reps':1,'proxy_update':10}
    currtime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    log_name = f'MFCAugment-{currtime}'
    writer = [SummaryWriter(log_dir=str(args.log_path.joinpath(args.save_name,log_name,f'task{x}'))) for x in range(len(tasks))]
    if args.multitask:
        if args.generative:
            if args.proxy:
                bestPop = GDPMFPSO(tasks, options, params, writer)
            else:
                bestPop = GDMFPSO(tasks, options, params, writer)
        else:
            # bestPop, skillFactor, bestInd = MFPSO(tasks, options, params, writer)
            bestPop, skillFactor, bestInd = MFSBX(tasks, options, params, writer)
    else:
         bestPop, skillFactor, bestInd = SBX(tasks, options, params, writer)
    if args.group:
        bestPolicy = formatPolicy(params, bestPop, skillFactor)
    else:
        bestPolicy = formatPolicy(params, bestPop)
    bestInd = formatPolicy(params, bestInd)
    skillFactor = np.array([bestPop[i,j].skill_factor for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])])
    for i, p in enumerate(bestPolicy):
        writer[0].add_text('Best Policy', str(p), i)
    for i, p in enumerate(bestInd):
        writer[0].add_text('Best Ind', str(p), i)
    # raise NotImplementedError
    return bestPolicy, skillFactor, groups
    
def formatPolicy(param, bestPop, skillFactor=None):
    Ub = param['Ub']
    Lb = param['Lb']
    args = param['args']
    n_op = args.num_ops
    formattedPolicyOut = []
    if isinstance(bestPop, np.ndarray):
        bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb)] for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])]
        bestPolicy = np.array(bestPolicy)
        if len(bestPolicy.shape) > 2:
            bestPolicy = bestPolicy.squeeze()
        if len(bestPolicy.shape) < 2:
            bestPolicy = bestPolicy.reshape(1,-1)
    else:
        bestPolicy = np.floor(bestPop.rnvec.T*(Ub-Lb)+Lb).reshape(1,-1)
    if skillFactor!=None:
        skillFactor = np.array(skillFactor).reshape(1,-1)
        for k in range(skillFactor.max()+1):
            if skillFactor.size == 1:
                idx = 0
            else:
                idx = np.where(skillFactor==k)[1]
            op_index = bestPolicy[idx,:n_op]
            mag_index = bestPolicy[idx,n_op:2*n_op]
            if args.use_prob:
                prob_index = bestPolicy[idx,2*n_op:3*n_op]
            uni_op_index = np.unique(op_index, axis=0)
            idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
            formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
            formattedPolicy['op_index'] = uni_op_index
            for i in idx:
                formattedPolicy['magnitude_index'].append(np.unique(mag_index[i,:],axis=0))
                if args.use_prob:
                    formattedPolicy['prob_index'].append(np.unique(prob_index[i,:],axis=0))
            formattedPolicyOut.append(formattedPolicy)
    else:
        op_index = np.unique(bestPolicy[:,:n_op], axis=0)
        uni_op_index = np.unique(op_index, axis=0)
        idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
        formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        formattedPolicy['op_index'] = uni_op_index
        for i in idx:
            formattedPolicy['magnitude_index'].append(np.unique(bestPolicy[i,n_op:2*n_op],axis=0))
            if args.use_prob:
                formattedPolicy['prob_index'].append(bestPolicy[:,2*n_op:3*n_op])
        formattedPolicyOut.append(formattedPolicy)

    return formattedPolicyOut











