import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   
import torch.distributions as distributions 
import numpy as np
import gco
import cv2

from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.ndimage import convolve1d

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
    
    if isinstance(model, nn.DataParallel):
        if 'resnet' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        elif 'wideresnet' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        elif 'preactresnet' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        elif 'vgg' in model_type:
            hook_handle = model.module.classifier[-1].register_forward_hook(hook_fn)
        elif 'inception' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        elif 'efficient' in model_type:
            hook_handle = model.module.classifier.register_forward_hook(hook_fn)
        elif 'googlenet' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        elif 'mobile' in model_type:
            hook_handle = model.module.classifier[-1].register_forward_hook(hook_fn)
        elif 'shufflenet' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        elif 'resnext' in model_type:
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
        else:
            # 默认情况，尝试使用avgpool
            hook_handle = model.module.avgpool.register_forward_hook(hook_fn)
    else:
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
    
    imgs = torch.split(img, batch_size, dim=0)
    feat = []
    out = []
    device = next(model.parameters()).device
    for x in imgs:
        with torch.no_grad():
            y = model(x)
            y = F.softmax(y, dim=1)
        out.append(y.view(x.size(0),-1))
        extracted_feat = hook_fn.features
        extracted_feat = extracted_feat.squeeze()
        if len(extracted_feat.shape) < 2:
            extracted_feat = extracted_feat.unsqueeze(0)
        extracted_feat = extracted_feat.to(device)
        # print(extracted_feat.shape)
        feat.append(extracted_feat)

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

import torch

def _scott_bandwidth_1d(samples):
    # Scott's rule: h = sigma * n^(-1/5)
    n = samples.shape[0]
    std = samples.std(unbiased=True)
    h = std * (n ** (-1.0 / 5.0))
    # 数值保护，避免 h=0
    return torch.clamp(h, min=1e-12)

def _silverman_bandwidth_1d(samples):
    # Silverman's rule: h = 1.06 * sigma * n^(-1/5)
    n = samples.shape[0]
    std = samples.std(unbiased=True)
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    return torch.clamp(h, min=1e-12)

def kde_1d_torch(samples, grid, bandwidth=None, rule='scott'):
    """
    1D KDE with Gaussian kernel on a fixed grid.
    samples: (N,) tensor
    grid: (M,) tensor
    bandwidth: float (tensor scalar) or None
    rule: 'scott' or 'silverman'
    return: (M,) normalized PDF on the grid
    """
    device = samples.device
    samples = samples.reshape(-1)
    grid = grid.reshape(-1)

    if bandwidth is None:
        if rule == 'silverman':
            bandwidth = _silverman_bandwidth_1d(samples)
        else:
            bandwidth = _scott_bandwidth_1d(samples)

    n = samples.shape[0]
    # 计算核值: K(u) = (1/sqrt(2π)) * exp(-0.5 u^2), KDE = (1/(n h)) * sum K((x - xi)/h)
    u = (grid.unsqueeze(1) - samples.unsqueeze(0)) / bandwidth  # (M, N)
    kernel_vals = torch.exp(-0.5 * u.pow(2))  # (M, N)
    pdf = (kernel_vals.sum(dim=1)) / (n * bandwidth * torch.sqrt(torch.tensor(2.0 * torch.pi, device=device)))

    # 数值防护，避免负值或过小值
    pdf = torch.clamp(pdf, min=1e-30)
    # 归一化到积分为1（等距网格近似）
    dx = (grid[1] - grid[0])
    pdf = pdf / (pdf.sum() * dx)
    return pdf

def kde_kl_divergence_1d_torch(x, y, xbins=1000, epsilon=1e-10, rule='scott', margin=0.0, noise_level=0.0):
    """
    1D KL(P||Q) with KDE on a fixed grid, mimicking scipy version.
    x, y: (Nx,), (Ny,) tensors (float)
    xbins: grid size
    epsilon: clamp minimum for densities
    rule: 'scott' or 'silverman'
    margin: extend the grid range by this fraction of range (e.g., 0.05 for ±5%)
    noise_level: add small Gaussian noise to inputs to stabilize
    """
    device = x.device
    x = x.reshape(-1)
    y = y.reshape(-1)

    if noise_level > 0.0:
        x = x + torch.randn_like(x) * noise_level
        y = y + torch.randn_like(y) * noise_level

    xmin = torch.min(torch.min(x), torch.min(y))
    xmax = torch.max(torch.max(x), torch.max(y))
    rng = xmax - xmin
    if margin > 0 and rng > 0:
        xmin = xmin - rng * margin
        xmax = xmax + rng * margin

    grid = torch.linspace(xmin, xmax, xbins, device=device)

    # 每个分布用自己的带宽（更贴近 scipy 的做法）
    pdf_x = kde_1d_torch(x, grid, bandwidth=None, rule=rule)
    pdf_y = kde_1d_torch(y, grid, bandwidth=None, rule=rule)

    pdf_x = torch.clamp(pdf_x, min=epsilon)
    pdf_y = torch.clamp(pdf_y, min=epsilon)

    dx = (grid[1] - grid[0])
    kl = torch.sum(pdf_x * torch.log(pdf_x / pdf_y)) * dx
    return kl

def kl_divergence_multivariate_torch(p, q, xbins=1000, epsilon=1e-10, rule='scott', margin=0.0, noise_level=1e-6):
    """
    多维KL(P||Q)，逐维做1D KDE并累加（与您现有的numpy/scipy实现一致）
    p: (Np, D) tensor
    q: (Nq, D) tensor
    returns: scalar float KL
    """
    assert p.shape[1] == q.shape[1], "p and q must have the same number of features"
    device = p.device
    D = p.shape[1]
    kl_total = torch.tensor(0.0, device=device)

    for i in range(D):
        xi = p[:, i]
        yi = q[:, i]
        kl_i = kde_kl_divergence_1d_torch(
            xi, yi,
            xbins=xbins,
            epsilon=epsilon,
            rule=rule,
            margin=margin,
            noise_level=noise_level
        )
        kl_total = kl_total + kl_i

    return kl_total.item()

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

def calculate_grid_pairwise_costs_convolved(img, sigma, pool_size):
    """
    Calculates horizontal, vertical, and diagonal pairwise costs for a grid graph.

    Args:
        img (np.ndarray): Input image (H, W, 3).
        sigma (float): Color sensitivity parameter.
        pool_size (int): Pooling window size for convolution.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - pairwise_h: Horizontal costs, shape (H, W-1)
            - pairwise_v: Vertical costs, shape (H-1, W)
            - pairwise_dr: Diagonal right costs, shape (H-1, W-1)
            - pairwise_dl: Diagonal left costs, shape (H-1, W-1)
    """
    h, w, _ = img.shape
    img_float = img.astype(np.float32)
    
    kernel = np.ones(pool_size) / pool_size
        
    # 水平方向：计算相邻像素差异
    horizontal_diff = np.mean(np.abs(img_float[:, :-1, :] - img_float[:, 1:, :]), axis=2)
    
    # 垂直方向：计算相邻像素差异
    vertical_diff = np.mean(np.abs(img_float[:-1, :, :] - img_float[1:, :, :]), axis=2)
    
    # 对角线方向：计算相邻像素差异
    # 右下对角线 (i,j) 和 (i+1,j+1)
    diag_right_diff = np.mean(np.abs(img_float[:-1, :-1, :] - img_float[1:, 1:, :]), axis=2)
    
    # 左下对角线 (i,j+1) 和 (i+1,j)
    diag_left_diff = np.mean(np.abs(img_float[:-1, 1:, :] - img_float[1:, :-1, :]), axis=2)
    
    # 对水平方向差异使用水平convolve1d（沿axis=1，即列方向）
    pairwise_h = convolve1d(horizontal_diff, kernel, axis=1, mode='constant', cval=0.0)
    
    # 对垂直方向差异使用垂直convolve1d（沿axis=0，即行方向）
    pairwise_v = convolve1d(vertical_diff, kernel, axis=0, mode='constant', cval=0.0)
    
    # 对对角线方向差异使用convolve1d
    pairwise_dr = convolve1d(convolve1d(diag_right_diff, kernel, axis=1, mode='constant', cval=0.0), 
                             kernel, axis=0, mode='constant', cval=0.0)
    pairwise_dl = convolve1d(convolve1d(diag_left_diff, kernel, axis=1, mode='constant', cval=0.0), 
                             kernel, axis=0, mode='constant', cval=0.0)

    # 归一化到[0,1]范围
    # if pairwise_h.max() > 1:
    #     pairwise_h = pairwise_h / pairwise_h.max()
    # if pairwise_v.max() > 1:
    #     pairwise_v = pairwise_v / pairwise_v.max()
    # if pairwise_dr.max() > 1:
    #     pairwise_dr = pairwise_dr / pairwise_dr.max()
    # if pairwise_dl.max() > 1:
    #     pairwise_dl = pairwise_dl / pairwise_dl.max()

    return pairwise_h, pairwise_v, pairwise_dr, pairwise_dl

def trimap_generate(input,
                saliency,
                trimap_alpha=10,
                trimap_alpha_threshold=0.2,
                trimap_gen='graph',
                sigma1=None,
                sigma2=None,
                lam1=None,
                lam2=None,
                mp=None,):
    
    if trimap_gen == 'graph':
        large_val_pairwise = 10
        large_val_unary = 100
        h, w, c = input.shape
        unary = np.zeros((3, h, w))
        # saliency = cv2.GaussianBlur(saliency, (3, 3), 0)
        # 0 for background, 2 for foreground, 1 for unknown
        unary[0] = -1*np.log(1-saliency+1e-8)
        unary[2] = -1*np.log(saliency+1e-8)
        # unary[0] = saliency
        # unary[1] = 1-saliency
        unary[1] = trimap_alpha * (saliency - 0.5)**2

        pairwise_cost = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                pairwise_cost[i, j] = (i - j)**2 / (3 - 1)**2
        
        # pairwise_cost = np.ones((3,3), dtype=np.float32)
        # np.fill_diagonal(pairwise_cost, 0)
        # pairwise_cost = np.array([
        #     [0, lam1, lam2],
        #     [lam1, 0, lam2],
        #     [lam2, lam2, 0]
        # ])

        # input = input.transpose(1, 2, 0)
        unary = unary.transpose(1, 2, 0)
        # pairwise_h, pairwise_v = calculate_grid_pairwise_costs(input, sigma2)
        pairwise_h, pairwise_v, pairwise_dr, pairwise_dl = calculate_grid_pairwise_costs_convolved(input, sigma1, 3)
        pairwise_h = np.nan_to_num(pairwise_h, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_v = np.nan_to_num(pairwise_v, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_dr = np.nan_to_num(pairwise_dr, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_dl = np.nan_to_num(pairwise_dl, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_h = (pairwise_h * large_val_pairwise).astype(np.int32)
        pairwise_v = (pairwise_v * large_val_pairwise).astype(np.int32)
        pairwise_dr = (pairwise_dr * large_val_pairwise).astype(np.int32)
        pairwise_dl = (pairwise_dl * large_val_pairwise).astype(np.int32)

        # unary[:,:-1] += np.repeat(pairwise_h[:,:,np.newaxis],3,axis=2)
        # unary[:-1,:] += np.repeat(pairwise_v[:,:,np.newaxis],3,axis=2)
        unary = np.nan_to_num(unary, nan=0.0, posinf=large_val_unary, neginf=-large_val_unary)
        # unary = (unary * large_val_unary).astype(np.int32)
        mask = gco.cut_grid_graph(unary, pairwise_cost, pairwise_v, pairwise_h, pairwise_dr, pairwise_dl, algorithm='swap')

        # edges, weights = _calculate_pairwise_costs_vectorized(input, sigma2)
        # edges, weights = calculate_pairwise_costs_with_convolve1d(input)
        # unary = unary.reshape(h*w, 3)
        # unary = np.nan_to_num(unary, nan=0.0, posinf=large_val, neginf=-large_val)
        # unary = (unary * large_val).astype(np.int32)
        # weights = np.nan_to_num(weights, nan=0.0, posinf=0, neginf=0)
        # weights = (weights * large_val).astype(np.int32)
        # mask = gco.cut_general_graph(edges, weights, unary, pairwise_cost, algorithm='swap')

        mask = mask.reshape(h, w)
        mask[mask == 0] = 255
        mask[mask == 1] = 128
        mask[mask == 2] = 0
    elif trimap_gen == 'stats':
        mask = saliency.copy()
        _, mask1 = cv2.threshold((saliency*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_bg = np.quantile(saliency, trimap_alpha_threshold)
        t_fg = np.quantile(saliency, 1-trimap_alpha_threshold)
        mask[mask<=t_bg] = 0
        mask[mask>=t_fg] = 1
        mask[(mask>t_bg)&(mask<t_fg)] = 0.5
        mask = (mask * 255).astype(np.uint8)
    # if mp is None:
    #     mask = []
    #     for i in range(b):
    #         # 将数据移动到正确的设备上
    #         mask_i = gco.cut_grid_graph(unary[i], pairwise_h[i], pairwise_v[i], pairwise_cost)
    #         mask.append(mask_i)
    # else:
    #     # 使用joblib替代multiprocessing，并传递索引确保输出与输入数据索引对齐
    #     inputs = [(i, unary[i].detach(), 
    #               pairwise_h[i], 
    #               pairwise_v[i], 
    #               pairwise_cost) 
    #              for i in range(b)]
    #     results = Parallel(n_jobs=1, backend='loky')(delayed(gco.cut_grid_graph)(*inp) for inp in inputs)
        
    #     # 根据索引重新排序结果
    #     sorted_results = sorted(results, key=lambda x: x[0])
    #     mask = [result[1] for result in sorted_results]

    mask = mask.astype(np.uint8)
    return mask

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)