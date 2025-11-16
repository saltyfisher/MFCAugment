import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

class RBFLayer(nn.Module):
    def __init__(self, in_features, num_centers, gamma=1.0):
        super(RBFLayer, self).__init__()
        self.in_features = in_features    # 输入维度
        self.num_centers = num_centers    # 中心点数量
        self.gamma = gamma                # 高斯函数宽度参数
        
        # 初始化中心点（可学习参数）
        self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
        # 初始化宽度（可学习或固定）
        self.log_gamma = nn.Parameter(torch.log(torch.Tensor([gamma])))  # 可学习gamma
        
        # 参数初始化
        nn.init.normal_(self.centers, 0, 1)  # 初始化为正态分布

    def forward(self, x):
        """
        输入: x (batch_size, in_features)
        输出: rbf_output (batch_size, num_centers)
        """
        # 计算输入x与所有中心点的欧氏距离
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, in_features)
        c_expanded = self.centers.unsqueeze(0)  # (1, num_centers, in_features)
        distances = torch.norm(x_expanded - c_expanded, dim=2)  # (batch_size, num_centers)
        
        # 应用高斯径向基函数
        gamma = torch.exp(self.log_gamma)  # 确保gamma为正
        rbf_output = torch.exp(-gamma * distances ** 2)
        return rbf_output
    
def activation_list(activation=None):
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'leakyrelu':
        activation = nn.LeakyReLU(1e-2)
    else:
        activation = nn.Identity()
    return activation

class InstanceNorm(nn.Module):
    def __init__(self, in_channels):
        super(InstanceNorm, self).__init__()
        self.in_channels = in_channels
    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)
        v = x.var(dim=1, keepdim=True)
        x = (x - m) / (v + 1e-6)
        return x
def norm_list(norm=None, in_channels=None):
    if norm == 'batch':
        norm = nn.BatchNorm1d(in_channels)
    elif norm == 'instance':
        norm = InstanceNorm(in_channels)
    elif norm == 'layer':
        norm = nn.LayerNorm(in_channels)
    else:
        norm = nn.Identity()
    return norm

def generate_linear(in_channels, out_channels, activation=None, norm=None):
    activation = activation_list(activation)
    if norm == 'spectral':
        linear = spectral_norm(nn.Linear(in_channels, out_channels))
        return nn.Sequential(
            linear,
            activation,
        )
    else:
        norm = norm_list(norm, out_channels)
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            norm,
            activation
            )

class ResBlockLinear(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, norm=None):
        super(ResBlockLinear, self).__init__()
        layers = []

        layers.append(generate_linear(in_channels, out_channels, activation, norm=norm))
        layers.append(generate_linear(out_channels, out_channels, norm=norm))
        self.layers = nn.Sequential(*layers)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.layers(x))
