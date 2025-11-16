import torch
import torch.nn as nn
import math

from functools import partial
from copy import deepcopy
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.cluster import KMeans  # 用于初始化中心点
from core.layer import *

class RBFNetwork(nn.Module):
    def __init__(self, in_features, num_centers, out_features, gamma=1.0):
        super(RBFNetwork, self).__init__()
        self.num_centers = num_centers
        layers = []
        layers.append(RBFLayer(in_features, num_centers, gamma))
        # layers.append(RBFLayer(num_centers, num_centers, gamma*0.1))
        # num_centers = num_centers//2
        layers.append(nn.Linear(num_centers, out_features))  # 线性输出层
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x)
        return output

    def init_centers_kmeans(self, data):
        """使用K-means聚类初始化中心点"""
        for layer in self.layers:
            if isinstance(layer, RBFLayer):
                kmeans = KMeans(n_clusters=self.num_centers)
                kmeans.fit(data)
                layer.centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
def init_sequential(seq_model):
    for layer in seq_model:
        if isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        elif isinstance(layer, nn.Conv1d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0.1)  # 卷积层偏置小正数初始化
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            init.ones_(layer.weight)  # gamma 参数
            init.zeros_(layer.bias)   # beta 参数

class Proxy(nn.Module):
    def __init__(self, in_channels, activation=None, norm=None):
        super(Proxy, self).__init__()
        layers = []
        for _ in range(2):
            layers.append(generate_linear(in_channels, in_channels//2, activation, norm))
            in_channels = in_channels//2
        layers.append(generate_linear(in_channels, 1, norm='tanh'))
        self.layers = nn.Sequential(*layers)
        self.counter = 0

        init_sequential(self.layers)

    def forward(self, x):
        return self.layers(x)
    
    def add_counter(self):
        self.counter += 1
    
    def get_counter(self):
        return self.counter
    
class Generator(nn.Module):
    def __init__(self, t_num, in_channels, f_num, activation=None, norm=None):
        # 是否要加norm层？总层数的设置？需要跟BigGAN或者StyleGAN一样对类别编码然后与输入拼接吗
        super(Generator, self).__init__()
        self.t_num = t_num
        encoder_norm = None
        decoder_norm = norm
        # Encoder
        self.encoder = nn.Sequential(
            generate_linear(in_channels+t_num, f_num, activation, encoder_norm)
        )
        # Bottleneck
        self.res_block = nn.Sequential(
            ResBlockLinear(f_num, f_num, activation, encoder_norm)
        )
        # Decoder
        self.decoder = nn.Sequential(
            generate_linear(f_num, in_channels, activation, decoder_norm)
        )
        self.embedding = nn.Embedding(t_num, f_num)
    def forward(self, x, y_fake):
        x = torch.cat([x, y_fake], dim=1)
        x = self.encoder(x)
        x1 = self.res_block(x)
        x2 = self.decoder(x1)
        return x2
    
class Discriminator(nn.Module):
    def __init__(self, t_num, in_channels, f_num, activation=None, norm=None):
        super(Discriminator, self).__init__()
        hidden_norm = 'None'
        # 得是layernorm
        real_fake_norm = norm
        domain_cls_norm = norm
        # Hidden
        layers = []
        # layers.append(generate_linear(in_channels, f_num, activation, hidden_norm))
        # layers.append(ResBlockLinear(f_num, f_num, activation, hidden_norm))
        # self.hidden = nn.Sequential(*layers)
        # layers = []
        # for _ in range(1):
        #     layers.append(ResBlockLinear(f_num, f_num, activation, real_fake_norm))
        # layers.append(generate_linear(f_num, f_num, activation, real_fake_norm))
        # self.real_fake = nn.Sequential(*layers)
        # layers = []
        # for _ in range(1):
        #     layers.append(ResBlockLinear(f_num, f_num, activation, domain_cls_norm))
        # layers.append(generate_linear(f_num, t_num, activation, domain_cls_norm))
        # self.domain_cls = nn.Sequential(*layers)

        layers.append(generate_linear(in_channels, f_num, activation, hidden_norm))
        for _ in range(1):
            # layers.append(generate_linear_specnorm(f_num, f_num//2, activation))
            layers.append(ResBlockLinear(f_num, f_num, activation, hidden_norm))
            # f_num = f_num//2
        self.hidden = nn.Sequential(*layers)
        # Output Dis
        layers = []
        f_num1 = f_num
        for _ in range(1):
            # layers.append(generate_linear_specnorm(f_num, f_num, activation))
            layers.append(ResBlockLinear(f_num1, f_num1, activation, real_fake_norm))
        # layers.append(generate_linear(f_num1, f_num1, activation, real_fake_norm))
        self.real_fake = nn.Sequential(*layers)
        layers = []
        f_num2 = f_num
        for _ in range(1):
            layers.append(ResBlockLinear(f_num2, f_num2, activation, domain_cls_norm))
        # self.domain_cls = nn.Sequential(*layers)
        layers.append(generate_linear(f_num2, t_num, activation, domain_cls_norm))
        self.domain_cls = nn.Sequential(*layers)
        self.embedding = nn.Embedding(t_num, f_num2)

    def forward(self, x, y=None):
        h = self.hidden(x)
        dist = self.real_fake(h)
        cls = self.domain_cls(h)
        # cls = self.domain_cls(self.embedding(y) + h)
        # if self.training:
        #     cls = self.cls(self.embedding(y) + cls)
        # else:
        #     cls = self.cls(cls)
        return dist, cls
    
class G_D(nn.Module):
    def __init__(
            self, t_num, in_channels, G_f_num, D_f_num, 
            G_activation='relu', D_activation='relu', 
            G_norm='instance', D_norm='spectral'
            ):
        super(G_D, self).__init__()
        self.G = Generator(t_num, in_channels, G_f_num, G_activation, G_norm)
        self.D = Discriminator(t_num, in_channels, D_f_num, D_activation, D_norm)
        self.apply(self.weights_init)
        self.counter_ = 0
        self.t_num = t_num
        self.handles = []

    def weights_init(self, m):
        if hasattr(m, 'weight'):
            act_dict = {
                nn.ReLU: ('kaiming', 'relu'),
                nn.LeakyReLU: ('kaiming', 'leaky_relu'),
                nn.Tanh: ('xavier', 'tanh'),
                nn.Sigmoid: ('xavier', 'sigmoid'),
                None: ('wgan', None)  # 默认WGAN初始化
            }

            is_spectral = hasattr(m, 'weight_orig') and hasattr(m, 'weight_u')

            act_type = None
            try:
                for child in m.children():
                    if isinstance(child, tuple(act_dict.keys())):
                        act_type = type(child)
                        break
            except:
                return

            if is_spectral:
                weight = m.weight_orig
                init_type, nonlinearity = act_dict.get(act_type, (None, None))
            else:
                weight = m.weight
                init_type, nonlinearity = act_dict.get(act_type, (None, None))
            if weight is not None:
            # 执行初始化
                if init_type == 'kaiming':
                    if nonlinearity == 'leaky_relu':
                        nn.init.kaiming_normal_(weight, a=0.2, mode='fan_in')
                    else:
                        nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity=nonlinearity)
                elif init_type == 'xavier':
                    gain = nn.init.calculate_gain(nonlinearity)
                    nn.init.xavier_normal_(weight, gain=1.0)
                else:  # WGAN默认初始化
                    nn.init.normal_(weight, 0.0, 0.02)
            
            # 初始化偏置项（如果存在）
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.01)  # 避免dead neurons

    def forward(self, x, y=None, y_fake=None, train_G=False):
        if y_fake is None:
            D_dist, D_cls = self.D(x, y)
            return D_dist, D_cls
        else:
            y_fake_oh = torch.zeros(x.shape[0], self.t_num).to(x.device)
            y_fake_oh.scatter_(1, y_fake, 1)
            with torch.set_grad_enabled(train_G):
                G_out = self.G(x, y_fake_oh)
            D_dist, D_cls = self.D(G_out, y_fake)
            if train_G:
                return G_out, D_dist, D_cls
            else:
                return D_dist, D_cls

    def add_counter(self):
        self.counter_ += 1

    def get_G_feat(self):
        return self.G_feat
    
    def get_counter(self):
        return self.counter_
    
    def hook_fn(self, name):
        def hook(module, input, output):
            self.G_feat = output.detach()
        return hook
    
    def register_hook(self, layer_names='res_block'):
        self.remove_hook()
        for name, module in self.G.named_children():
                if name == layer_names:
                    handle = module.register_forward_hook(self.hook_fn(name))
                    self.handles.append(handle)
    def remove_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def __del__(self):
        self.remove_hook()

class DomainClassifier(nn.Module):
    def __init__(self, in_channels, t_num, activation=None, norm=None):
        super(DomainClassifier, self).__init__()
        layers = []
        for _ in range(2):
            layers.append(generate_linear(in_channels, in_channels//2, activation, norm))
            in_channels = in_channels//2
        layers.append(generate_linear(in_channels, t_num, activation, norm))
        self.layers = nn.Sequential(*layers)
        self.counter = 0
    def forward(self, x):
        x = self.layers(x)
        return x

    def add_counter(self):
        self.counter += 1
    
    def get_counter(self):
        return self.counter