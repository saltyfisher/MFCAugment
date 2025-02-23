import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import entropy
from scipy import linalg
import numpy as np
from scipy import linalg

def check_matrix_dimensions(p_sigma, q_sigma):
    if p_sigma.shape != q_sigma.shape:
        raise ValueError("p_sigma 和 q_sigma 的维度必须相同")
    if p_sigma.shape[0] != p_sigma.shape[1]:
        raise ValueError("p_sigma 和 q_sigma 必须是方阵")

def ensure_positive_definite(sigma, min_value=1e-6):
    # 确保对角线元素不小于一个合理的最小值
    sigma = np.maximum(sigma, min_value * np.eye(sigma.shape[0]))
    
    # 检查矩阵是否为正定矩阵
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        # 如果不是正定矩阵，尝试通过添加一个小的正数到对角线来修复
        eigenvalues = np.linalg.eigvalsh(sigma)
        smallest_eigenvalue = np.min(eigenvalues)
        if smallest_eigenvalue <= 0:
            correction = abs(smallest_eigenvalue) + min_value
            sigma += correction * np.eye(sigma.shape[0])
    return sigma

def cholesky_decomposition(sigma):
    try:
        return linalg.cholesky(sigma, lower=True)
    except linalg.LinAlgError as e:
        raise ValueError(f"Cholesky 分解失败: {e}")

def log_determinant(sigma):
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise ValueError("行列式必须为正")
    return logdet

def KL_loss(p_mu, p_sigma, q_mu, q_sigma):
    # 检查输入矩阵维度
    check_matrix_dimensions(p_sigma, q_sigma)
    
    # 确保矩阵为正定矩阵
    p_sigma = ensure_positive_definite(p_sigma)
    q_sigma = ensure_positive_definite(q_sigma)
    
    n = p_sigma.shape[0]
    
    # 计算 Cholesky 分解
    L_q = cholesky_decomposition(q_sigma)
    L_p = cholesky_decomposition(p_sigma)
    
    # 计算 (p_mu - q_mu).T @ q_sigma_inv @ (p_mu - q_mu)
    diff = p_mu - q_mu
    l1 = diff.T @ linalg.solve_triangular(L_q, linalg.solve_triangular(L_q.T, diff, lower=True), lower=True)
    
    # 计算 log(det(q_sigma) / det(p_sigma))
    logdet_q = log_determinant(q_sigma)
    logdet_p = log_determinant(p_sigma)
    l2 = logdet_q - logdet_p
    
    # 计算 trace(q_sigma_inv @ p_sigma)
    l3 = np.trace(linalg.solve(q_sigma, p_sigma))
    
    loss = l1 + l2 + l3 - n
    return loss / 2

# 其他函数保持不变

def KL_loss_sum(gm1_mu, gm1_sigma, gm2_mu, gm2_sigma, gm1_alpha, gm2_alpha):
    loss = []
    m = gm1_mu.shape[0]
    for i in range(m):
       loss.append(gm1_alpha[i]*(np.log(gm1_alpha[i]/gm2_alpha[i]) + KL_loss(gm1_mu[i], gm1_sigma[i], gm2_mu[i], gm2_sigma[i])))
       #loss.append((1/gm1_alpha[i])*(KL_loss(gm1_mu[i], gm1_sigma[i], gm2_mu[i], gm2_sigma[i])))
       #loss.append(KL_loss(gm1_mu[i], gm1_sigma[i], gm2_mu[i], gm2_sigma[i]))
    return sum(loss)

def euclidian_dist(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def change_row(X, row1, row2):
    X[[row1, row2]] = X[[row2, row1]]
    
def change_column(X, column1, column2):
    X[:, [column1, column2]] = X[:, [column2, column1]]

def personcoef(x1, x2):
    x1_center = x1-np.mean(x1, axis=0)
    x2_center = x2-np.mean(x2, axis=0)
    x1_normal = x1_center/np.std(x1_center, axis=0)
    x2_normal = x2_center/np.std(x2_center, axis=0)
    corr_coef = np.dot(x1_normal.T, x2_normal)/x1_normal.shape[0]
    return corr_coef 
    
# 贪心算法计算KL散度上界
def KL_loss_upbound(gm1_mu, gm1_sigma, gm2_mu, gm2_sigma, gm1_alpha, gm2_alpha):
    m = gm1_mu.shape[0]
    for i in range(m):
        dist = []
        for j in range(i, m):
            dist.append(euclidian_dist(gm1_mu[i], gm2_mu[j]))
        min_index = i + np.argmin(dist)
        change_row(gm2_mu, i, min_index)
        change_row(gm2_alpha, i, min_index)
        change_row(gm2_sigma, i, min_index)
    return KL_loss_sum(gm1_mu, gm1_sigma, gm2_mu, gm2_sigma, gm1_alpha, gm2_alpha)

# 轮转损失
def KL_loss_cycle(aug_mu, aug_sigma, ori_mu, ori_sigma):
    m = aug_mu.shape[0]
    loss = []
    for i in range(0, m- 1):
        loss.append(KL_loss(aug_mu[i], aug_sigma[i], ori_mu[i + 1], ori_sigma[i + 1]))
    loss.append(KL_loss(aug_mu[m - 1], aug_sigma[m - 1], ori_mu[0], ori_sigma[0]))
    return sum(loss)

def KL_loss_all(aug_mu, aug_sigma, ori_mu, ori_sigma, idx, w):
    loss = [w[idx][j]*KL_loss(aug_mu,aug_sigma,ori_mu[j],ori_sigma[j]) for j in range(ori_mu.shape[0])]
    return loss

# W距离
def W_loss(gm1_mu, gm1_sigma, gm2_mu, gm2_sigma):
    m = gm1_mu.shape[0]
    loss = []
    for i in range(0, m-1):
        loss.append(euclidian_dist(gm1_mu[i], gm2_mu[i+1]))
        loss.append(euclidian_dist(gm1_sigma[i], gm2_sigma[i+1]))
    loss.append(euclidian_dist(gm1_mu[m-1], gm2_mu[0]))
    loss.append(euclidian_dist(gm1_sigma[m-1], gm2_sigma[0]))
    return sum(loss)

def Person_coef(gm1_mu, gm1_sigma, gm2_mu, gm2_sigma):
    m = gm1_mu.shape[0]
    loss = []
    for i in range(0, m-1):
        loss.append(personcoef(gm1_mu[i], gm2_mu[i+1]))
    loss.append(personcoef(gm1_mu[m-1], gm2_mu[0]))
    return sum(loss)

def cos_loss(gm1_mu, gm1_sigma, gm2_mu, gm2_sigma):
    m = gm1_mu.shape[0]
    loss = []
    for i in range(0, m-1):
        loss.append(1-cosine(gm1_mu[i], gm2_mu[i+1]))
    loss.append(1-cosine(gm1_mu[m-1], gm2_mu[0]))
    return sum(loss)