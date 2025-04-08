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
    # loss_dist = torch.mean(dis_real)
    # loss_dist += torch.mean(-dis_fake)
    loss_dist = torch.mean(F.relu(1. + dis_real))
    loss_dist += torch.mean(F.relu(1. - dis_fake))
    return loss_dist

def loss_domain(pred_real, y_real):
    return F.cross_entropy(pred_real, y_real.squeeze())
def loss_hinge_gen(dis_fake):
    # loss = torch.mean(-dis_fake)
    loss = torch.mean(F.relu(1 - dis_fake))
    return loss

class AddNoise:
    def __init__(self, min=0.0, max=0.1):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        noise = (torch.rand(tensor.shape)*(self.max-self.min)).to(tensor.device)
        return torch.clamp(tensor + noise, 0., 1.)
    
class LayerMonitor:
    def __init__(self, model, writer, log_interval=1):
        self.writer = writer
        self.log_interval = log_interval
        self.step = 0
        
        # 自动注册所有层的hook
        self.handles = []
        for name, layer in model.named_modules():
            if not isinstance(layer, (nn.ModuleList, nn.Sequential)):
                handle = layer.register_forward_hook(
                    self._create_hook(name)
                )
                self.handles.append(handle)
    
    def _create_hook(self, layer_name):
        def hook(module, input, output):
            # 每隔指定步数记录
            if len(output) == 2:
                return output
            if self.step % self.log_interval == 0:
                with torch.no_grad():
                    # 标量统计
                    self.writer.add_scalar(
                        f"activations/{layer_name}/mean", 
                        output.mean(), self.step
                    )
                    self.writer.add_scalar(
                        f"activations/{layer_name}/std",
                        output.std(), self.step
                    )
                    
                    # 直方图
                    self.writer.add_histogram(
                        f"activations_hist/{layer_name}",
                        output.cpu(),
                        self.step
                    )
                    
                    # 梯度统计（如果存在）
                    if output.requires_grad:
                        output.register_hook(
                            lambda grad: self._grad_hook(grad, layer_name)
                        )
            return output
        return hook
    
    def _grad_hook(self, grad, layer_name):
        self.writer.add_scalar(
            f"gradients/{layer_name}/mean",
            grad.mean(), self.step
        )
        self.writer.add_histogram(
            f"gradients_hist/{layer_name}",
            grad.cpu(),
            self.step
        )
    
    def update_step(self):
        self.step += 1
        
    def remove(self):
        """移除所有hook"""
        for handle in self.handles:
            handle.remove()