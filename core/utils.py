import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   
import torch.distributions as distributions 
import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde
from scipy.integrate import quad

def get_deepfeat(model_name, model, img):
    extracted_feat = []
    def hook_fn(model, input, output):
        nonlocal extracted_feat
        extracted_feat = output.detach()

    if 'resnet18' in model_name['type']:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(img)
        
    feat = extracted_feat.view(extracted_feat.size(0),-1)
    hook_handle.remove()

    return feat

def kde_kl_divergence(x, y, xbins=1000, epsilon=1e-10):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    if len(x) == 0 or len(y) == 0:
        return np.inf
    # 计算 KDE
    kde_x = gaussian_kde(x)
    kde_y = gaussian_kde(y)
    
    # 定义积分范围
    xmin = min(np.min(x), np.min(y))
    xmax = max(np.max(x), np.max(y))
    x_range = np.linspace(xmin, xmax, xbins)
    
    # 计算 KDE 的概率密度
    dens_x = kde_x(x_range)
    dens_y = kde_y(x_range)
    dens_x = np.maximum(dens_x, epsilon)
    dens_y = np.maximum(dens_y, epsilon)
    # 计算 KL 散度
    delta_x = np.abs(x_range[1] - x_range[0])
    kl_div = np.sum(dens_x * np.log(dens_x / dens_y)) * delta_x
    
    return kl_div

def kl_divergence_multivariate(x, y, xbins=1000, noise_level=1e-6):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    x = x+np.random.normal(0, noise_level, x.shape)
    y = y+np.random.normal(0, noise_level, y.shape)
    
    n_features = x.shape[1]
    
    # 初始化 KL 散度
    kl_div = 0.0
    
    for i in range(n_features):
        kl_div += kde_kl_divergence(x[:, i], y[:, i], xbins=xbins)
    
    return kl_div

def KL_loss(p, q):
    # 检查输入矩阵维度
    loss = kl_divergence_multivariate(p, q)
    return loss

def KL_loss_intergroup(p, q, idx, w):
    loss = [w[idx][j]*KL_loss(p,q) for j in range(w.shape[0])]
    return loss

def KL_loss_all(p, q):
    loss = KL_loss(p,q)
    return loss

def loss_hinge_dis(dis_fake, dis_real):
    loss_dist = torch.mean(F.relu(1. - dis_real))
    loss_dist += torch.mean(F.relu(1. + dis_fake))
    return loss_dist

def loss_domain(pred_real, y_real):
    return F.cross_entropy(pred_real, y_real)
def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

class AddNoise:
    def __init__(self, min=0.0, max=0.1):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        noise = (torch.rand(tensor.shape)*(self.max-self.min)).to(tensor.device)
        return torch.clamp(tensor + noise, 0., 1.)

import torch

class PearsonCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # 防止除以零的小常数

    def forward(self, pred, target):
        # 确保输入维度一致
        assert pred.shape == target.shape, "预测值和目标值形状需一致"

        # 中心化数据（减去均值）
        pred_centered = pred - torch.mean(pred)
        target_centered = target - torch.mean(target)

        # 计算协方差和标准差
        covariance = torch.sum(pred_centered * target_centered, dim=1)
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) + self.eps)
        target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=1) + self.eps)

        # 计算皮尔逊相关系数
        pearson_corr = covariance / (pred_std * target_std)

        # 损失 = 1 - 相关系数（最小化损失等价于最大化相关性）
        loss = 1 - pearson_corr.mean()
        return loss