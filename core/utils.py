import numpy as np
import torch

from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy import linalg
import numpy as np
from scipy import linalg

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

    return feat.cpu().numpy().squeeze()

def kde_kl_divergence(x, y, xbins=1000, epsilon=1e-10):
    """
    计算两个样本之间的 KL 散度，使用 KDE 方法。
    
    参数:
    x (np.ndarray): 第一个样本数据
    y (np.ndarray): 第二个样本数据
    bandwidth (float): KDE 的带宽参数
    xbins (int): 用于积分的区间数量
    
    返回:
    float: KL 散度值
    """
    # 确保输入是 numpy 数组
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
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
    delta_x = x_range[1] - x_range[0]
    kl_div = np.sum(dens_x * np.log(dens_x / dens_y)) * delta_x
    
    return kl_div

def kl_divergence_multivariate(x, y, xbins=1000):
    """
    计算两组多维样本之间的 KL 散度，使用 KDE 方法。
    
    参数:
    x (np.ndarray): 第一组样本数据，形状为 (n_samples, n_features)
    y (np.ndarray): 第二组样本数据，形状为 (n_samples, n_features)
    xbins (int): 用于积分的区间数量
    
    返回:
    float: KL 散度值
    """
    # 确保输入是 numpy 数组
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
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

def KL_loss_all(p, q, idx, w):
    loss = [w[idx][j]*KL_loss(p,q) for j in range(w.shape[0])]
    return loss