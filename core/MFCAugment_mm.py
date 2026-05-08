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
import kornia.augmentation as K
from functools import reduce
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tensorboardX import SummaryWriter
from torch_pca import PCA as PCA_torch
from tqdm import tqdm
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from core.FeatureExtractor import FeatureExtractor
from core.augmentations import MyAugmentMM, augmentation_space
from core.utils import KL_loss_all, KL_loss, Jensen_loss, Sinkhorn_dist, kl_divergence_kde, kl_divergence_multivariate_torch
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
from core.augmentations import MM
# from core.dataCluster import constrained_kmeans_ilp

def getdatafeat(args, resize_size, data_list, model):
    # model = torch.nn.DataParallel(model)
    st = time.time()
    if args.resize:
        # transformer = transforms.Compose([transforms.Resize(resize_size), ToTensor()])
        batch = 8192
        # if args.dataset == 'chestct':
        #     batch = 16
        # if args.dataset == 'breakhis':
    else:
        # transformer = ToTensor()
        batch = 1
    
    all_feat_list = []
    all_cls_list = []

    for i in range(0,len(data_list),batch):
        batch_data = data_list[i:i+batch]

        # 创建当前批次的输入
        # input_list = [(transformer(d)).unsqueeze(0) for d in batch_data]
        input_list = [(d).unsqueeze(0) for d in batch_data]
        batch_input = torch.cat(input_list).to(args.device)
        
        # 获取特征
        feat, cls = get_deepfeat(args, model, batch_input)
        # 立即释放输入张量内存
        del input_list, batch_input
        
        all_feat_list.append(feat)
        all_cls_list.append(cls)

        if args.gpu:
            torch.cuda.empty_cache()
    # print('getdatafeat time:', time.time()-st)
    return all_feat_list, all_cls_list

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

def process_policy(args_tuple):
    """
    处理单个增强策略的函数
    """
    p, data, args, num_ops, resize_size, model, pca, feat_list, groups, group_id = args_tuple
    
    aug = MyAugment(p, num_ops)
    aug_data = [aug(d) for d in data]
    aug_feat, _ = getdatafeat(args, resize_size, aug_data, model)
    
    if args.gpu:
        aug_feat = torch.cat(aug_feat).detach()
    else:
        aug_feat = torch.cat(aug_feat).detach().cpu().numpy()
    
    aug_feat = pca.transform(aug_feat)
    
    if args.gpu:
        # loss1 = kl_divergence_kde(feat_list, aug_feat)
        # loss2 = kl_divergence_kde(feat_list[groups[group_id]], policy_feat)
        loss1 = kl_divergence_multivariate_torch(feat_list, aug_feat)
        loss2 = kl_divergence_multivariate_torch(feat_list[groups[group_id]], aug_feat)
    else:
        loss1 = KL_loss(feat_list, aug_feat)
        loss2 = KL_loss(feat_list[groups[group_id]], aug_feat)
        
    l = 1
    loss = loss1 - l * loss2
    return loss 

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
    batch_size = 24
    # 格式化所有策略
    formatted_policies = [formatPolicy(params, p, verbose=True)[0] for p in policies]
    # 为每个策略生成增强数据
    data = [data_list[i] for i in groups[group_id]]
    all_losses = []

    process_args = [
        (p, data, args, params['n_op'], resize_size, model, pca, feat_list, groups, group_id)
        for p in formatted_policies
    ]
    
    # 使用ThreadPoolExecutor进行并行处理
    st = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_policy = {
            executor.submit(process_policy, args_tuple): i 
            for i, args_tuple in enumerate(process_args)
        }
        
        # 收集结果
        results = {}
        for future in as_completed(future_to_policy):
            index = future_to_policy[future]
            result = future.result()
            results[index] = result
        
        # 按顺序排列结果
        all_losses = [results[i] for i in sorted(results.keys())]    
    return all_losses

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
        # loss1 = kl_divergence_kde(feat_list, aug_feat)
        # loss2 = kl_divergence_kde(feat_list[groups[group_id]], aug_feat)
        loss1 = kl_divergence_multivariate_torch(feat_list, aug_feat)
        loss2 = kl_divergence_multivariate_torch(feat_list[groups[group_id]], aug_feat)
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
    l = args.l
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
    # print(f'loss1: {loss1}, loss2: {loss2}')
    # print(loss)
    # loss = loss2
    return loss

def evalFuncBayes(policy, params):
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
    matting_method = params['matting_method']
    # policy = np.floor(policy*(Ub-Lb)+Lb)
    # policy = formatPolicy(params, policy)
    aug = MyAugmentMM(policy,num_ops=params['n_op'], post_augment=args.post_augment)
    aug_data = []
    target_data = [data_list[i] for i in groups[group_id]]
    source_data = [data_list[i] for i in np.random.permutation(len(data_list))[:len(target_data)]]
    for i, d in enumerate(source_data):
        mix_data, mix_mask, _ = aug(d, matting_method)
        aug_data.append(mix_data+(1-mix_mask)*np.array(target_data[i]))
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
        # loss1 = kl_divergence_kde(feat_list, aug_feat)
        # loss2 = kl_divergence_kde(feat_list[groups[group_id]], aug_feat)
        loss1 = kl_divergence_multivariate_torch(feat_list, aug_feat)
        loss2 = kl_divergence_multivariate_torch(feat_list[groups[group_id]], aug_feat)
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
    l = args.l
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
    # print(f'loss1: {loss1}, loss2: {loss2}')
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
    # sample_num = int(feat_list.shape[0] * 0.8)
    sample_counts = n_clusters
    label_list = np.array(label_list)
    if isinstance(feat_list, torch.Tensor):
        feat_list = feat_list.cpu().numpy()
    weights = -np.log(feat_list[np.arange(feat_list.shape[0]),label_list]+1e-6)*np.sum(-feat_list * np.where(feat_list > 0, np.log(feat_list+1e-6), 0), axis=1)
    weights = np.nan_to_num(weights, nan=0.0, neginf=0)
    weights = weights / np.sum(weights)
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
    groups = []
    for _ in range(sample_counts):
        # groups.append(np.concatenate([np.random.choice(idx[i], int(sample_num[i]), p=weights[i]) for i in range(len(idx))]))
        groups.append(np.random.choice(np.arange(label_list.shape[0]), int(sample_num), p=weights))
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    # groups_weights = [np.sum(weights[groups[i]]) for i in range(len(groups))]
    # idx = np.argmax(groups_weights)
    # return groups[idx], centers

    # intersection = reduce(np.intersect1d, groups)
    # union = reduce(np.union1d, groups)
    # ratio = len(intersection)/len(union)
    # return groups, centers, intersection, ratio

    return groups, centers
def MFCAugment(model, resize_size, data_list, label_list, args, n_clusters, mag_bin=31, prob_bin=10, num_ops=2, max_samples=100, matting_method=None):
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
    # groups, centers, intersection, ratio = cluster_data_weighted(cls_list, label_list, n_clusters) 
    # print(f"Jaccard ratio: {ratio}\t Intersection size: {len(intersection)}")
    # # if ratio < 0.1:
    # #     return [], [], [] 
    # # groups = [np.unique(np.concatenate(groups))]
    # groups = [intersection]
    # centers = []
    groups, centers = cluster_data_weighted(cls_list, label_list, n_clusters)
    # groups = [group]
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
    num_ops += 1
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
    if args.bayes:
        tasks = [SingleTask(num_ops*n_dims,Lb,Ub,[0]*var_dim, evalFuncBayes) for _ in range(len(groups))]
    else:
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
              'matting_method':matting_method
              }
    options = {'popsize':30,'maxgen':2,'rmp':0.3,'reps':2,'proxy_update':10}
    currtime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    log_name = f'MFCAugment-{currtime}-{args.num_ops}'
    writer = [SummaryWriter(log_dir=os.path.join(args.log_path, args.save_name, log_name, f'task{x}')) for x in range(len(tasks))]
    if args.multitask:
        if args.generative:
            if args.proxy:
                bestPop = GDPMFPSO(tasks, options, params, writer)
            else:
                bestPop = GDMFPSO(tasks, options, params, writer)
        else:
            # bestPop, skillFactor, bestInd = MFPSO(tasks, options, params, writer)
            bestPop, skillFactor, bestInd = MFSBX(tasks, options, params, writer)
    elif args.bayes:
         bestPolicy = bayesian_optimization_tasks_parallel(tasks, args, params, rep=args.bayes_rep, topk=args.bayes_topk, max_evals=args.bayes_max_eval)
    else:
         bestPop, skillFactor, bestInd = SBX(tasks, options, params, writer)
    if not args.bayes:
        if args.group:
            bestPolicy = formatPolicy(params, bestPop, skillFactor)
        else:
            bestPolicy = formatPolicy(params, bestPop)
    # bestInd = formatPolicy(params, bestInd)
    # skillFactor = np.array([bestPop[i,j].skill_factor for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])])
    for i, p in enumerate(bestPolicy):
        writer[0].add_text('Best Policy', str(p), i)
    # for i, p in enumerate(bestInd):
    #     writer[0].add_text('Best Ind', str(p), i)
    # raise NotImplementedError
    return bestPolicy, groups
    
def formatPolicy(param, bestPop, skillFactor=None,verbose=False):
    Ub = param['Ub']
    Lb = param['Lb']
    args = param['args']
    n_op = param['n_op']
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
        if verbose:
            op_index = bestPolicy[:,:n_op]
            uni_op_index = op_index.copy()
            idx = np.arange(bestPolicy.shape[0])
        else:
            op_index = bestPolicy[:,:n_op]
            uni_op_index = np.unique(op_index, axis=0)
            idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
        formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        formattedPolicy['op_index'] = uni_op_index
        for i in idx:            
            if args.use_prob:
                if verbose:
                    formattedPolicy['prob_index'].append(bestPolicy[i,n_op:2*n_op].reshape(1,-1))
                    formattedPolicy['magnitude_index'].append(bestPolicy[i,2*n_op:3*n_op].reshape(1,-1))
                else:
                    formattedPolicy['prob_index'].append(np.unique(bestPolicy[i,n_op:2*n_op],axis=0))
                    formattedPolicy['magnitude_index'].append(np.unique(bestPolicy[i,2*n_op:3*n_op],axis=0))
            else:
                if verbose:
                    formattedPolicy['magnitude_index'].append(bestPolicy[i,n_op:2*n_op].reshape(1,-1))
                else:
                    formattedPolicy['magnitude_index'].append(np.unique(bestPolicy[i,n_op:2*n_op],axis=0))
        formattedPolicyOut.append(formattedPolicy)

    return formattedPolicyOut

import logging
# 在文件开头或在导入hyperopt之前添加
logging.getLogger('hyperopt').setLevel(logging.WARNING)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from core.augmentations_fastaa import augment_list
import numpy as np

def policy_decoder(augment, use_prob, n_op):
    formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
    op_idx = []
    op_level = []
    op_prob = []
    for i in range(0, n_op):
        op_idx.append(augment['policy_%d' % i])
        op_level.append(augment['level_%d' % i])
        if i == 0:
            op_prob.append(augment['prob_%d' % i])
        elif use_prob:
            op_prob.append(augment['prob_%d' % i])
    formattedPolicy['op_index'] = np.array([op_idx])
    formattedPolicy['magnitude_index'] = np.array([op_level])
    formattedPolicy['prob_index'] = np.array([op_prob])

    return formattedPolicy

def bayesian_optimization_tasks(tasks, args, params, max_evals=200):
    """
    使用HyperOpt的贝叶斯优化来求解任务
    
    Parameters:
    tasks : list
        任务列表，每个任务应具有evaluate方法
    params : dict
        参数字典，传递给任务评估函数
    max_evals : int
        最大评估次数
    
    Returns:
    best_policy : list
        最优策略列表
    skill_factors : list
        技能因子列表
    best_ind : list
        最佳个体列表
    """
    
    # 存储每个任务的结果
    task_results = []

    # 对每个任务分别进行贝叶斯优化
    for task_idx, task in enumerate(tasks):
        trial_history = []
        print(f"Optimizing task {task_idx+1}/{len(tasks)}")
        params['task_id'] = task_idx
        # 定义搜索空间
        space = {}
        for i in range(args.num_ops):
            if i == 0:
                space['policy_%d' % i] = hp.choice('policy_%d' % i, 100)
                space['prob_%d' % i] = hp.choice('prob_%d' % i, 10)
                space['level_%d' % i] = hp.choice('level_%d' % i, 10)
            else:
                space['policy_%d' % i] = hp.choice('policy_%d' % i, list(range(0, len(augment_list()))))
                if args.use_prob:                
                    space['prob_%d' % i] = hp.choice('prob_%d' % i, list(range(0, args.prob_bin-1)))
                space['level_%d' % i] = hp.choice('level_%d' % i, list(range(0, args.mag_bin-1)))
        
        # 定义目标函数
        def objective(x):
            # 将字典转换为数组
            policy = policy_decoder(x, args.use_prob, args.num_ops)
            # 评估策略
            loss = task.evaluate(policy, params)
            trial = {'policy':policy,'loss':loss}
            trial_history.append(trial)
            return loss
        
        # 执行贝叶斯优化
        trials = Trials()
        best = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=max_evals,
                   trials=trials,
                   show_progressbar=True)
        trial_history = sorted(trial_history, key=lambda x: x['loss'], reverse=False)
        trial_history = trial_history[:50]
        merged_policies = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        for r in trial_history:
            p = r['policy']
            merged_policies['op_index'].append(p['op_index'])
            merged_policies['magnitude_index'].append(p['magnitude_index'])
            if args.use_prob:
                merged_policies['prob_index'].append(p['prob_index'])
        final_policies = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        op_index = merged_policies['op_index']
        uni_op_index = np.unique(op_index, axis=0)
        idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
        final_policies['op_index'] = uni_op_index
        mag_idx = np.array(merged_policies['magnitude_index']).squeeze()
        prob_idx = np.array(merged_policies['prob_index']).squeeze()
        for i in idx:            
            if args.use_prob:
                final_policies['prob_index'].append(np.unique(prob_idx[i,:],axis=0))
                final_policies['magnitude_index'].append(np.unique(mag_idx[i,:],axis=0))
            else:
                final_policies['magnitude_index'].append(np.unique(mag_idx[i,:],axis=0))
        # 获取最优策略
        task_results.append(final_policies)
        print(f"Task {task_idx+1} best loss: {trials.best_trial['result']['loss']}")
    
    return task_results

def bayesian_optimization_tasks_parallel(tasks, args, params, rep=1, topk=100, max_evals=200):
    """
    使用HyperOpt的贝叶斯优化来求解任务
    
    Parameters:
    tasks : list
        任务列表，每个任务应具有evaluate方法
    params : dict
        参数字典，传递给任务评估函数
    max_evals : int
        最大评估次数
    
    Returns:
    best_policy : list
        最优策略列表
    skill_factors : list
        技能因子列表
    best_ind : list
        最佳个体列表
    """
    
    # 存储每个任务的结果
    final_results = []
    for r in range(rep):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(bayesian_optimization_single_task, task_idx, args, task, params, rep, topk, max_evals): task_idx 
                for task_idx, task in enumerate(tasks)
            }
            
            # 收集结果
            results = [None] * len(tasks)
            task_times = [None] * len(tasks)
            task_best_loss = [None] * len(tasks)
            for future in as_completed(future_to_task):
                result, task_idx, elapsed_time, best_loss = future.result()
                results[task_idx] = result
                task_times[task_idx] = elapsed_time
                task_best_loss[task_idx] = best_loss
            final_results.extend(results)
            print("\n=== Task Execution Summary ===")
            for i in range(len(tasks)):
                print(f'Rep {r+1} Task {i+1}: Time = {task_times[i]:.2f}s, Best Loss = {task_best_loss[i]:.4f}')   
    return final_results

def bayesian_optimization_single_task(task_idx, args, task, params, rep=1, topk=100, max_evals=200):
    """
    对单个任务使用贝叶斯优化
    
    Parameters:
    task : SingleTask
        单个任务对象
    params : dict
        参数字典
    max_evals : int
        最大评估次数
    
    Returns:
    best_policy : array
        最优策略
    best_loss : float
        最佳损失值
    """
    
    st = time.time()
    # print(f"Optimizing task {task_idx+1}")
    params['task_id'] = task_idx
    # 定义搜索空间
    space = {}
    for i in range(args.num_ops+1):
        if i == 0:
            space['policy_%d' % i] = hp.choice('policy_%d' % i, [100])
            space['prob_%d' % i] = hp.choice('prob_%d' % i, np.arange(0,1,0.1))
            space['level_%d' % i] = hp.choice('level_%d' % i, np.arange(0,1,0.1))
        else:
            space['policy_%d' % i] = hp.choice('policy_%d' % i, list(range(0, len(augment_list()))))
            if args.use_prob:                
                space['prob_%d' % i] = hp.choice('prob_%d' % i, list(range(0, args.prob_bin-1)))
            space['level_%d' % i] = hp.choice('level_%d' % i, list(range(0, args.mag_bin-1)))
    
    # 定义目标函数
    trial_history = []
    def objective(x):
        # 将字典转换为数组
        policy = policy_decoder(x, args.use_prob, args.num_ops+1)
        # 评估策略
        loss = task.evaluate(policy, params)
        trial = {'policy':policy,'loss':loss}
        trial_history.append(trial)
        return loss

    # 执行贝叶斯优化
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                show_progressbar=False,
                verbose=False)
    trial_history = sorted(trial_history, key=lambda x: x['loss'], reverse=False)[:topk]
    merged_policies = {'op_index':[],'prob_index':[],'magnitude_index':[]}
    for r in trial_history:
        p = r['policy']
        merged_policies['op_index'].append(p['op_index'])
        merged_policies['magnitude_index'].append(p['magnitude_index'])
        if (100 in p['op_index']) or args.use_prob:
            merged_policies['prob_index'].append(p['prob_index'])
    final_policies = {'op_index':[],'prob_index':[],'magnitude_index':[]}
    op_index = merged_policies['op_index']
    uni_op_index = np.unique(op_index, axis=0)
    all_idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
    final_policies['op_index'] = uni_op_index.squeeze()
    mag_idx = np.array(merged_policies['magnitude_index']).squeeze()
    if len(mag_idx.shape) < 2:
        mag_idx = mag_idx[:, np.newaxis]
    if args.use_prob:
        prob_idx = np.array(merged_policies['prob_index']).squeeze()
        if len(prob_idx.shape) < 2:
            prob_idx = prob_idx[:, np.newaxis]
    elif (100 in uni_op_index[i]):
        prob_idx = np.array(merged_policies['prob_index']).squeeze()
        if len(prob_idx.shape) < 2:
            prob_idx = prob_idx[:, np.newaxis]
    if len(mag_idx) < 2:
        mag_idx = mag_idx.unsqueeze(0)
    for i, idx in enumerate(all_idx):
        if (100 in op_index[i]) or args.use_prob:
            final_policies['prob_index'].append(np.unique(prob_idx[idx,:],axis=0))
            final_policies['magnitude_index'].append(np.unique(mag_idx[idx,:],axis=0))
        else:
            final_policies['magnitude_index'].append(np.unique(mag_idx[idx,:],axis=0))
    best_loss = min([trial['loss'] for trial in trial_history])
    elapsed_time = time.time() - st
    return final_policies, task_idx, elapsed_time, best_loss









