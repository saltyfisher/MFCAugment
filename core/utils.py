import numpy as np

from scipy.spatial.distance import cosine
from scipy.stats import entropy

def KL_loss(p_mu, p_sigma, q_mu, q_sigma):
    p_sigma = p_sigma + 1e-10*np.eye(p_sigma.shape[0])
    q_sigma = q_sigma + 1e-10*np.eye(q_sigma.shape[0])
    n = p_sigma.shape[0]
    q_sigma_inv = np.linalg.inv(q_sigma)
    l1 = (p_mu-q_mu).T @ q_sigma_inv @ (p_mu-q_mu)
    l2 = np.log(np.linalg.det(q_sigma)/np.linalg.det(p_sigma))
    l3 = np.trace(q_sigma_inv @ p_sigma)
    loss = l1 + l2 + l3 - n
    return loss/2

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