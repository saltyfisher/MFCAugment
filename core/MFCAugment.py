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
# import geatpy as ea

from tensorboardX import SummaryWriter
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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from EA.MFPSO import MFPSO
from EA.GDMFPSO import GDMFPSO
from EA.GDPMFPSO import GDPMFPSO
from dataclasses import dataclass
from core.utils import get_deepfeat
from core.model import Proxy, RBFNetwork, G_D
from core.trainer_GD import train_GD, test_GD
from core.trainer_proxy import train_proxy, test_proxy
# from core.dataCluster import constrained_kmeans_ilp

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
    pca = params['pca']
    w = params['w']
    group_id = params['task_id']
    Lb = params['Lb']
    Ub = params['Ub']
    args = params['args']
    config = params['config']
    # policy = np.floor(policy*(Ub-Lb)+Lb)
    policy = formatPolicy(params, policy)
    aug = MyAugment(policy,num_ops=params['n_op'],resize=args.resize,resize_size=config['img_size'])
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
    # target_imgs = feat_list[groups!=group_id]
    # loss = KL_loss_intergroup(target_imgs, aug_feat, group_id, w)
    # loss = np.sum(np.hstack(loss))
    loss = KL_loss_all(feat_list, aug_feat)
    # loss1 = KL_loss_all(feat_list, aug_feat)
    # loss2 = KL_loss_all(feat_list[group_id], aug_feat)
    # loss = loss1 - loss2
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
    skf = StratifiedShuffleSplit(n_splits=max(label_list)+1,test_size=1-1/n_clusters)
    groups = []
    for i, (train_index, test_index) in enumerate(skf.split(feat_list, label_list)):
        groups.append(train_index)
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    return groups, centers

def MFCAugment(model, config, data_list, label_list, args, n_clusters=8, max_samples=100):
    mag_bin = args.mag_bin
    prob_bin = args.prob_bin
    n_op = args.num_op
    total_op_num = len(augmentation_space())
    sample_num = 500
    ###提取特征### 
    if args.resize:
        transformer = transforms.Compose([transforms.Resize(config['img_size']), ToTensor()])
    else:
        transformer = ToTensor()
    input_list = [(transformer(d)).unsqueeze(0) for d in data_list]
    feat_list = [get_deepfeat(config['model'],model,d.to(args.device)) for d in input_list]
    feat_list = torch.cat(feat_list, dim=0).cpu().numpy()
    pca = PCA(n_components=int(0.05*feat_list[0].shape[0]))
    feat_list = pca.fit_transform(feat_list)
    sampled_data = None
    # if os.path.exists(f'./training_data_{args.dataset}_{sample_num}.pkl'):
    #     with open(f'./training_data_{args.dataset}_{sample_num}.pkl', 'rb') as f:
    if os.path.exists(f'./training_data_{args.dataset}.pkl'):
        with open(f'./training_data_{args.dataset}.pkl', 'rb') as f:
            sampled_data = pickle.load(f)
            groups = sampled_data['groups']
            centers = sampled_data['centers']
    else:
        groups, centers = cluster_data(feat_list, label_list, n_clusters)
    w = distance.squareform(distance.pdist(centers))
    w = 1 - w/np.sum(w, axis=1)
    new_data_list = []
    batch_size = args.batch_size//8
    for indices in groups:
        all_batch_imgs = []
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
    var_dim = n_op*3
    Lb = np.array([0]*var_dim)
    Ub = np.array(([total_op_num-1]*(n_op)) + ([mag_bin-1]*n_op) + ([prob_bin-1]*n_op))
    # if args.proxy:
    #     tasks = [SingleTask(n_op*3,Lb,Ub,[0]*var_dim, evalFuncProxy) for _ in range(len(groups))]
    # else:
    #     tasks = [SingleTask(n_op*3,Lb,Ub,[0]*var_dim, evalFunc) for _ in range(len(groups))]
    tasks = [SingleTask(n_op*3,Lb,Ub,[0]*var_dim, evalFunc) for _ in range(len(groups))]
    params = {'feat_extractor':model,
              'data_list':new_data_list, 
              'feat_list':feat_list, 
              'batch_size':args.batch_size, 
              'groups':groups, 
              'centers':centers,
              'pca':pca, 
              'Lb':Lb, 'Ub':Ub,
              'w':w,'n_op':n_op,'mag_bin':mag_bin,'prob_bin':prob_bin,
              'args':args,'config':config
              }
    # if args.proxy:
    #     if os.path.exists(f'./params_save/proxy/proxy_{args.dataset}.pth'):
    #         proxies = torch.load(f'./params_save/proxy/proxy_{args.dataset}.pth')
    #         params.update({'proxies':proxies})
    #     else:
    #         sampled_policies, labels, task_id = generate_proxy_data(params, sample_num, sampled_data)
    #         cfg = args.proxy_config
    #         if cfg['model']['type'] == 'mlp':
    #             proxies = [Proxy(var_dim, cfg['model']['fnum']) for _ in range(len(params['groups']))] 
    #         elif cfg['model']['type'] == 'rbf':
    #             proxies = [RBFNetwork(var_dim, cfg['model']['fnum'], 1, cfg['model']['gamma']) for _ in range(len(params['groups']))] 
    #             # for i, p in enumerate(proxies):
    #             #     p.init_centers_kmeans(sampled_policies[i])
    #         for i, p in enumerate(proxies):
    #             print(f'Proxy {i} training...')
    #             proxies[i] = train_proxy(i, p, params, sampled_policies[i], labels[i])
    #             print(f'Proxy {i} trained.')
    #         print('Proxy generating finished')
    #         os.mkdirs('./params_save/proxy', exist_ok=True)
    #         torch.save(proxies, f'./params_save/proxy/proxy_{args.dataset}.pth')
    #         params.update({'proxies':proxies})
        # raise NotImplementedError
    # if args.GD:
    #     sampled_policies, labels, task_id = generate_proxy_data(params, sample_num, sampled_data)
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    #     for train_index, test_index in sss.split(sampled_policies, task_id):
    #         sampled_policies_train = sampled_policies[train_index]
    #         sampled_policies_test = sampled_policies[test_index]
    #         task_id_train = task_id[train_index]
    #         task_id_test = task_id[test_index]
    #     cfg = args.GD_config
    #     t_num = len(params['groups'])
    #     in_channel = params['Lb'].shape[0]
    #     GD = G_D(t_num, in_channel, cfg['G_model']['f_num'], cfg['D_model']['f_num'], D_activation='relu',D_norm='layer',G_activation='relu',G_norm='layer')
    #     GD.train()
    #     GD = train_GD(params, GD, sampled_policies_train, task_id_train)
    #     GD.eval()
    #     test_result = test_GD(params, GD, sampled_policies_test, task_id_test)
    #     print(f'Test Cls Acc:{test_result:.2f}')
    #     print('GD trained.')
    #     raise NotImplementedError
    options = {'popsize':100,'maxgen':100,'rmp':0.3,'reps':2,'proxy_update':10}
    currtime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    log_name = f'MFCAugment-{currtime}'
    writer = [SummaryWriter(log_dir=str(args.log_path.joinpath(args.save_name,log_name,f'task{x}'))) for x in range(len(tasks))]
    if args.GD:
        if args.proxy:
            bestPop = GDPMFPSO(tasks, options, params, writer)
        else:
            bestPop = GDMFPSO(tasks, options, params, writer)
    else:
        bestPop = MFPSO(tasks, options, params, writer)
    bestPolicy = formatPolicy(params, bestPop)
    # bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb)] for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])]
    # bestPolicy, idx = np.unique(bestPolicy, axis=0, return_index=True)
    # bestPolicy = bestPolicy.tolist()
    # skillFactor = skillFactor[idx]
    skillFactor = np.array([bestPop[i,j].skill_factor for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])])
    # policy_str = [f'{x}' for x in bestPolicy]
    # policy_str = '  '.join(policy_str)
    # writer[0].add_text('Best Policy', policy_str, 0)
    writer[0].add_text('Best Policy', bestPolicy, 0)
    # raise NotImplementedError
    return bestPolicy, skillFactor, groups
    
def formatPolicy(param, bestPop):
    Ub = param['Ub']
    Lb = param['Lb']
    args = param['args']
    n_op = args.num_op
    if isinstance(bestPop, np.ndarray):
        bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb).reshape(-1,n_op)] for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])]
        bestPolicy = np.array(bestPolicy).squeeze()
    else:
        bestPolicy = np.floor(bestPop.rnvec.T*(Ub-Lb)+Lb).reshape(1,-1)
    op_index = bestPolicy[:,:n_op]
    mag_index = bestPolicy[:,n_op:2*n_op]
    prob_index = bestPolicy[:,2*n_op:3*n_op]
    op_index = np.unique(op_index, axis=0)
    idx = [np.where((op_index[i,:]==op_index).all(-1))[0] for i in range(op_index.shape[0])]
    formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
    for i in idx:
        formattedPolicy['op_index'].append(op_index[i, :])
        formattedPolicy['magnitude_index'].append(np.unique(mag_index[i,:],axis=0))
        formattedPolicy['prob_index'].append(np.unique(prob_index[i,:],axis=0))

    return formattedPolicy












