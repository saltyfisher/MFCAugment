from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.cluster import KMeans
import numpy as np

def robust_cluster_sampling(X, n_clusters=10, min_samples=1):
    # Step 1: 聚类
    k = X.shape[0]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    
    # Step 2: 计算基础采样数（每个簇至少min_samples）
    if n_clusters * min_samples > k:
        raise ValueError(f"k must be >= {n_clusters * min_samples} to guarantee min_samples per cluster")
    
    # 分配基础样本
    base_samples = np.full(n_clusters, min_samples, dtype=int)
    remaining = k - n_clusters * min_samples
    
    # Step 3: 按比例分配剩余名额
    proportions = cluster_counts / cluster_counts.sum()
    extra_samples = (proportions * remaining).astype(int)
    
    # 处理余数
    remainder = remaining - extra_samples.sum()
    if remainder > 0:
        largest_clusters = np.argsort(-cluster_counts)[:remainder]
        extra_samples[largest_clusters] += 1
    
    # 合并总采样数
    total_samples = base_samples + extra_samples
    
    # Step 4: 校验并执行抽样
    selected_indices = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        n_available = len(cluster_indices)
        n_request = total_samples[cluster]
        
        # 动态调整逻辑
        if n_request > n_available:
            print(f"Warning: Cluster {cluster} has only {n_available} samples, using all")
            n_request = n_available
        
        if n_request > 0:
            chosen = np.random.choice(cluster_indices, size=n_request, replace=False)
            selected_indices.extend(chosen)
    
    # 最终校验
    if len(selected_indices) < k:
        print(f"Warning: Only {len(selected_indices)} samples collected due to cluster limits")
    
    return X[np.random.permutation(selected_indices)]

def cluster_data(feat_list, label_list, n_clusters):
    skf = StratifiedShuffleSplit(n_splits=max(label_list)+1,test_size=1-1/n_clusters)
    groups = []
    for i, (train_index, test_index) in enumerate(skf.split(feat_list, label_list)):
        groups.append(train_index)
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    return groups, centers