import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   
import torch.distributions as distributions 
import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde
from scipy.integrate import quad

def get_deepfeat(args, model, img):
    model_name = args.model
    extracted_feat = []
    batch_size = args.batch_size

    class FeatureHook:
        def __init__(self):
            self.features = None
        
        def __call__(self, module, input, output):
            self.features = output.detach().clone()

    hook_fn = FeatureHook()

    # 根据模型类型注册不同的hook
    model_type = model_name['type'] if isinstance(model_name, dict) else model_name
    
    if 'resnet' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'wideresnet' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'preactresnet' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'vgg' in model_type:
        hook_handle = model.classifier[-1].register_forward_hook(hook_fn)
    elif 'inception' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'efficient' in model_type:
        hook_handle = model.classifier.register_forward_hook(hook_fn)
    elif 'googlenet' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'mobile' in model_type:
        hook_handle = model.classifier[-1].register_forward_hook(hook_fn)
    elif 'shufflenet' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'resnext' in model_type:
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    else:
        # 默认情况，尝试使用avgpool
        hook_handle = model.avgpool.register_forward_hook(hook_fn)

    # if 'resnet' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # elif 'wideresnet' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # elif 'preactresnet' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # elif 'vgg' in model_type:
    #     hook_handle = model.module.classifier[-1].register_forward_hook(hook_fn)
    # elif 'inception' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # elif 'efficient' in model_type:
    #     hook_handle = model.module.classifier.register_forward_hook(hook_fn)
    # elif 'googlenet' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # elif 'mobile' in model_type:
    #     hook_handle = model.module.classifier[-1].register_forward_hook(hook_fn)
    # elif 'shufflenet' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # elif 'resnext' in model_type:
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    # else:
    #     # 默认情况，尝试使用avgpool
    #     hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    
    imgs = torch.split(img, batch_size, dim=0)
    feat = []
    out = []
    for x in imgs:
        with torch.no_grad():
            y = model(x)
            y = F.softmax(y, dim=1)
        out.append(y.view(x.size(0),-1))
        extracted_feat = hook_fn.features
        extracted_feat = extracted_feat.squeeze()
        if len(extracted_feat.shape) < 2:
            extracted_feat = extracted_feat.unsqueeze(0)
        feat.append(extracted_feat.squeeze())

        del y, extracted_feat

    feat = torch.cat(feat, dim=0)
    out = torch.cat(out, dim=0)
    hook_handle.remove()

    return feat, out

def get_clsprob(model_name, model, img):
    extracted_feat = []
    def hook_fn(model, input, output):
        nonlocal extracted_feat
        extracted_feat = output.detach()

    if 'resnet18' in model_name['type']:
        hook_handle = model.fc.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(img)
        
    feat = extracted_feat.view(extracted_feat.size(0),-1)
    hook_handle.remove()

    return feat

import numpy as np

def compute_cost_matrix(X_s, X_t, metric='sqeuclidean'):
    """
    计算源点和目标点之间的成本矩阵 M。
    M_ij = cost(X_s_i, X_t_j)

    参数:
        X_s (np.ndarray): 源向量集合, 形状 (n, d)
        X_t (np.ndarray): 目标向量集合, 形状 (m, d)
        metric (str): 使用的距离度量, 'sqeuclidean' (平方欧氏距离) 或 'euclidean'

    返回:
        np.ndarray: 成本矩阵, 形状 (n, m)
    """
    # print("📏 计算成本矩阵...")
    n_source = X_s.shape[0]
    n_target = X_t.shape[0]
    M = np.zeros((n_source, n_target))

    if metric == 'sqeuclidean':
        # 更高效的广播计算平方欧氏距离
        sum_sq_X_s = np.sum(X_s**2, axis=1, keepdims=True)
        sum_sq_X_t = np.sum(X_t**2, axis=1, keepdims=True)
        M = sum_sq_X_s + sum_sq_X_t.T - 2 * X_s @ X_t.T
        M = np.maximum(M, 0) # 确保非负，处理数值误差
        # 使用循环以便与之前的版本行为一致，虽然效率稍低
        # for i in range(n_source):
        #     for j in range(n_target):
        #         M[i, j] = np.sum((X_s[i, :] - X_t[j, :])**2)
    elif metric == 'euclidean':
        for i in range(n_source):
            for j in range(n_target):
                M[i, j] = np.sqrt(np.sum((X_s[i, :] - X_t[j, :])**2))
    else:
        raise ValueError(f"未知的度量: {metric}")
    
    # print("成本矩阵 M (部分):\n", M[:min(3, n_source), :min(3, n_target)])
    return M

def estimate_point_distribution(points, bandwidth_factor=0.5, min_sigma=1e-3):
    """
    根据点集自身估计每个点的概率分布（权重）。
    权重较大的点通常是那些周围有较多其他点的点（即高密度区域的点）。
    使用高斯核的和来估计每个点的“密度”，然后归一化。

    参数:
        points (np.ndarray): 点集, 形状 (N, d)
        bandwidth_factor (float): 用于确定高斯核带宽的因子。
                                  带宽 sigma = median_pairwise_distance * bandwidth_factor.
        min_sigma (float): 带宽的最小值，防止 sigma 过小或为0。

    返回:
        np.ndarray: 每个点的权重, 形状 (N,), 总和为1。
    """
    # print(f"📊 估算 {points.shape[0]} 个点的分布 (方法: 高斯核密度和)...")
    N = points.shape[0]
    if N == 0:
        return np.array([])
    if N == 1:
        return np.array([1.0])

    # 1. 计算点对之间的平方欧氏距离
    # (x-y)^2 = x^2 - 2xy + y^2
    points_sq_sum = np.sum(points**2, axis=1) # 每个点各维度平方和
    # dist_sq_matrix[i,j] = ||points[i] - points[j]||^2
    dist_sq_matrix = points_sq_sum[:, np.newaxis] + points_sq_sum[np.newaxis, :] - 2 * (points @ points.T)
    dist_sq_matrix = np.maximum(dist_sq_matrix, 0) # 确保非负，处理数值误差

    # 2. 确定带宽 sigma
    sigma = min_sigma # 默认最小带宽
    if N > 1:
        pairwise_distances = np.sqrt(dist_sq_matrix[np.triu_indices(N, k=1)]) # 提取上三角部分的距离（不包括对角线）
        if len(pairwise_distances) > 0:
            median_dist = np.median(pairwise_distances)
            # 如果所有点重合，median_dist 可能为0
            calculated_sigma = median_dist * bandwidth_factor
            sigma = max(calculated_sigma, min_sigma)
        # 如果只有两个点且它们重合 (pairwise_distances为空或全0), sigma 保持 min_sigma
        
    # print(f"用于分布估计的高斯核带宽 sigma: {sigma:.4f}")

    # 3. 计算每个点的“密度”或“影响力”
    # density_i = sum_j exp(-||points_i - points_j||^2 / (2 * sigma^2))
    # 对角线上的 dist_sq_matrix[i,i] 是 0，所以 exp(0) = 1。
    # 这意味着每个点对自身的贡献是1。
    kernel_matrix = np.exp(-dist_sq_matrix / (2 * sigma**2 + 1e-10))
    raw_weights = np.sum(kernel_matrix, axis=1) # 每个点是其所在行（或列）的和

    # 4. 归一化权重
    sum_raw_weights = np.sum(raw_weights)
    if sum_raw_weights == 0: # 所有权重都是0（例如，如果sigma非常小且点相距很远）
        # print("⚠️ 所有原始权重为0，返回均匀分布。")
        return np.ones(N) / N
        
    weights = raw_weights / sum_raw_weights
    
    # print(f"估算得到的权重 (前几个): {weights[:min(5,N)]}")
    return weights

def Sinkhorn_dist(X_s, X_t, a=None, b=None, reg=0.1, num_iters=100, tol=1e-9, verbose=False,
                       estimate_marginals=False, marginals_bandwidth_factor=0.5):
    """
    使用 Sinkhorn 算法计算两个向量集合之间的最优传输距离。

    参数:
        X_s (np.ndarray): 源向量集合, 形状 (n, d)
        X_t (np.ndarray): 目标向量集合, 形状 (m, d)
        a (np.ndarray, optional): 源分布 (权重), 长度 n。
        b (np.ndarray, optional): 目标分布 (权重), 长度 m。
        reg (float): 熵正则化参数 (lambda)。
        num_iters (int): 最大迭代次数。
        tol (float): 收敛的容忍度。
        verbose (bool): 是否打印迭代过程中的信息。
        estimate_marginals (bool): 如果为True且a或b为None，则从数据估计边际分布。
                                    否则，如果a或b为None，使用均匀分布。
        marginals_bandwidth_factor (float): 用于估计边际分布时的高斯核带宽因子。

    返回:
        tuple: (P, dist)
            P (np.ndarray): 近似的最优传输计划, 形状 (n, m)
            dist (float): Sinkhorn 距离 (最优传输成本)
    """
    # print("\n🌀 开始 Sinkhorn 算法计算...")
    n = X_s.shape[0]
    m = X_t.shape[0]

    # 1. 处理/估算边际分布 a 和 b
    if a is None:
        if estimate_marginals and n > 0:
            # print("源分布 'a' 未提供，将从源点集 X_s 估计...")
            a = estimate_point_distribution(X_s, bandwidth_factor=marginals_bandwidth_factor)
        elif n > 0:
            a = np.ones(n) / n
            # print(f"源分布 'a' 未提供，使用均匀分布 (每个元素 {1/n:.4f})")
        else:
            a = np.array([]) # 空输入
    
    if b is None:
        if estimate_marginals and m > 0:
            # print("目标分布 'b' 未提供，将从目标点集 X_t 估计...")
            b = estimate_point_distribution(X_t, bandwidth_factor=marginals_bandwidth_factor)
        elif m > 0:
            b = np.ones(m) / m
            # print(f"目标分布 'b' 未提供，使用均匀分布 (每个元素 {1/m:.4f})")
        else:
            b = np.array([]) # 空输入

    if n == 0 or m == 0:
        # print("⚠️ 源点集或目标点集为空，无法计算 Sinkhorn 距离。")
        return np.array([]).reshape(n,m), 0.0

    if not (np.isclose(np.sum(a), 1.0, atol=1e-6) and np.isclose(np.sum(b), 1.0, atol=1e-6)):
        print(f"⚠️ 警告: 分布 'a' (sum={np.sum(a):.6f}) 或 'b' (sum={np.sum(b):.6f}) 的和不接近 1。")
        print("   确保估算的或提供的分布是有效的概率分布。")

    # 2. 计算成本矩阵 M
    M = compute_cost_matrix(X_s, X_t)

    # 3. 计算 Kernel 矩阵 K
    K = np.exp(-M / reg)
    # print(f"Kernel 矩阵 K (部分) (基于 exp(-M/reg={reg})):\n", K[:min(3,n), :min(3,m)])

    # 4. Sinkhorn 迭代
    # print(f"\n🔄 开始迭代 (最多 {num_iters} 次, 收敛容忍度 {tol:.1e})...")
    u = np.ones(n) / n 
    v = np.ones(m) / m 

    for i in range(num_iters):
        u_prev = u.copy()

        # 更新 v: v = b / (K^T u)
        KTu = K.T @ u
        v = b / (KTu + 1e-10) # 加一个小常数避免除以零

        # 更新 u: u = a / (K v)
        Kv = K @ v
        u = a / (Kv + 1e-10) # 加一个小常数避免除以零

        if verbose and (i % (num_iters // 10 if num_iters > 10 else 1) == 0 or i == num_iters - 1):
            # 检查边际约束的满足程度 (可选，但有助于调试)
            # P_curr = u[:, np.newaxis] * K * v[np.newaxis, :]
            # err_a = np.linalg.norm(P_curr.sum(axis=1) - a)
            # err_b = np.linalg.norm(P_curr.sum(axis=0) - b)
            # print(f"迭代 {i+1:03d}: u变化 {np.linalg.norm(u - u_prev):.2e}, 边际误差 a: {err_a:.2e}, b: {err_b:.2e}")
            print(f"迭代 {i+1:03d}: u变化 {np.linalg.norm(u - u_prev):.2e}")

        if np.linalg.norm(u - u_prev) < tol:
            # print(f"✅ 在第 {i+1} 次迭代收敛。")
            break
        # if i == num_iters - 1: 
        #     print(f"⚠️ 达到最大迭代次数 {num_iters}，可能未完全收敛。最后的u变化: {np.linalg.norm(u - u_prev):.2e}")

    # 5. 计算最优传输计划 P
    P = u[:, np.newaxis] * K * v[np.newaxis, :]
    # print("\n近似的最优传输计划 P (部分):\n", P[:min(3,n), :min(3,m)])

    # 6. 计算 Sinkhorn 距离
    sinkhorn_dist = np.sum(P * M)
    # print(f"\n🏆 计算得到的 Sinkhorn 距离: {sinkhorn_dist:.6f}")
    
    # 验证 P 的边际
    # print(f"P 的行和 (应接近 a): {P.sum(axis=1)[:min(5,n)]} ... (a: {a[:min(5,n)]} ...)")
    # print(f"P 的列和 (应接近 b): {P.sum(axis=0)[:min(5,m)]} ... (b: {b[:min(5,m)]} ...)")
    # print(f"总和 P.sum() (应接近 1): {P.sum():.6f}")
    sinkhorn_dist = np.round(sinkhorn_dist, 6)
    return sinkhorn_dist

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
    loss = np.round(loss, 4)
    return loss

def KL_loss_intergroup(p, q, idx):
    loss = [w[idx][j]*KL_loss(p,q) for j in range(w.shape[0])]
    return loss

def KL_loss_all(p, q):
    loss = KL_loss(p,q)
    return loss


def gaussian_kde_pdf(samples, eval_points, bandwidth):
    """
    高斯核密度估计 PDF 计算 (支持高维)
    samples: (N, D)
    eval_points: (M, D)
    bandwidth: float
    return: (M,) PDF 值
    """
    N, D = samples.shape
    diff = eval_points.unsqueeze(1) - samples.unsqueeze(0)  # (M, N, D)
    # 高斯核公式
    kernel_vals = torch.exp(-0.5 * diff.pow(2).sum(dim=2) / (bandwidth ** 2))
    const = 1.0 / ((2 * torch.pi) ** (D / 2) * (bandwidth ** D))
    pdf_vals = const * kernel_vals.mean(dim=1)
    return pdf_vals

def kl_divergence_kde(p_samples, q_samples, bandwidth=None):
    """
    高维KDE估计KL散度
    p_samples: (Np, D)
    q_samples: (Nq, D)
    bandwidth: float，可选。如果None则自动选取
    return: KL(P||Q)
    """
    device = p_samples.device
    Np, D = p_samples.shape
    
    # 自动带宽选择（Silverman's rule）
    if bandwidth is None:
        std_p = p_samples.std(dim=0).mean()
        std_q = q_samples.std(dim=0).mean()
        avg_std = (std_p + std_q) / 2
        bandwidth = 1.06 * avg_std * (Np ** (-1 / (D + 4)))
    
    # 用p_samples作为评估点
    eval_points = p_samples
    
    # KDE计算pdf
    p_pdf = gaussian_kde_pdf(p_samples, eval_points, bandwidth)
    q_pdf = gaussian_kde_pdf(q_samples, eval_points, bandwidth)
    
    eps = 1e-12
    kl_vals = torch.log((p_pdf + eps) / (q_pdf + eps))
    return kl_vals.mean().item()
def Jensen_loss(p, q):
    loss = 0.5 * KL_loss(p, q) + 0.5 * KL_loss(p, q)
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