import numpy as np
import os
import time
import torch
import joblib
import pickle
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import rankdata


logging.getLogger('hyperopt').setLevel(logging.WARNING)

def getdatafeat(args, resize_size, data_list, model):
    from core.utils import get_deepfeat

    # model = torch.nn.DataParallel(model)
    st = time.time()
    if args.resize:
        batch = 64
    else:
        batch = 1
    
    all_feat_list = []
    all_cls_list = []

    for i in range(0,len(data_list),batch):
        batch_data = data_list[i:i+batch]

        # 创建当前批次的输入
        input_list = [(d).unsqueeze(0) for d in batch_data]
        batch_input = torch.cat(input_list).to(args.device)
        
        # 获取特征
        feat, cls = get_deepfeat(args, model, batch_input)
        # 立即释放输入张量内存
        del input_list, batch_input
        
        all_feat_list.append(feat)
        all_cls_list.append(cls)

        if args.gpu:
            torch.cuda.empty_cache()
    # print('getdatafeat time:', time.time()-st)
    return all_feat_list, all_cls_list

class SingleTask(object):
    def __init__(self,dim:int,Lb,Ub,encode:list[int],fnc):
        self.dims = dim
        self.Lb = Lb
        self.Ub = Ub
        self.encode = encode
        self.evalfnc = fnc

    def evaluate(self, x, params):
        params.update({'Lb':self.Lb,'Ub':self.Ub})
        return self.evalfnc(x, params)


def load_policy_eval_dependencies():
    from core.augmentations import MyAugment
    from core.utils import KL_loss, MMD_loss, kl_divergence_multivariate_torch

    return MyAugment, KL_loss, MMD_loss, kl_divergence_multivariate_torch


def resolve_bayes_eval_groups(params):
    if params.get('eval_groups') is not None:
        return params['eval_groups']
    if params.get('eval_group') is not None:
        return params['eval_group']
    return params['groups']

def evalFuncBayes(policy, params):
    MyAugment, KL_loss, MMD_loss, kl_divergence_multivariate_torch = load_policy_eval_dependencies()

    data_list = params['data_list']
    feat_list = params['feat_list']
    groups = resolve_bayes_eval_groups(params)
    pca = params['pca']
    group_id = params['task_id']
    args = params['args']
    model = params['model']
    resize_size = params['resize_size']
    aug = MyAugment(policy,num_ops=params['n_op'])
    aug_data = []
    data = [data_list[i] for i in groups[group_id]]
    for d in data:
        d = (d*255).to(torch.uint8)
        d = aug(d) / 255
        aug_data.append(d)
    aug_feat, _ = getdatafeat(args,resize_size,aug_data,model)
    if args.gpu:
        aug_feat = torch.cat(aug_feat).detach()
    else:
        aug_feat = torch.cat(aug_feat).detach().cpu().numpy()
    aug_feat = pca.transform(aug_feat)
    eval_metric = getattr(args, 'mfc_eval_metric', 'kl')
    if eval_metric == 'mmd':
        loss1 = MMD_loss(feat_list, aug_feat)
        loss2 = MMD_loss(feat_list[groups[group_id]], aug_feat)
    elif args.gpu:
        loss1 = kl_divergence_multivariate_torch(feat_list, aug_feat)
        loss2 = kl_divergence_multivariate_torch(feat_list[groups[group_id]], aug_feat)
    else:
        loss1 = KL_loss(feat_list, aug_feat)
        loss2 = KL_loss(feat_list[groups[group_id]], aug_feat)
    return loss1 - args.l * loss2

def evalFuncProxy(policy, params):
    group_id = params['task_id']
    proxies = params['proxies']

    p = proxies[group_id]
    p.eval()
    with torch.no_grad():
        policy = torch.FloatTensor(policy).to(params['args'].device).unsqueeze(0)
        loss = p(policy)
        loss = loss.item()
    return loss

def generate_proxy_data(params,sample_num=500, sampled_data=None):
    # 生成训练数据
    print('Collecting data')
    dataset = params['args'].dataset
    input_channels = len(params['Lb'])
    if sampled_data is not None:
        sampled_policies = sampled_data['sampled_policies']
        labels = sampled_data['labels']
        task_id = sampled_data['task_id']
    else:
        sampled_policies = [np.random.rand(sample_num, input_channels) for _ in range(len(params['groups']))]
        # sampled_policies = [pyDOE3.lhs(input_channels, sample_num) for _ in range(len(params['groups']))]
        labels = []
        task_id = []
        # 计算适应值函数值
        with joblib.Parallel(n_jobs=10,backend='threading') as parallel:
            for i, policies in enumerate(sampled_policies):
                params['task_id'] = i
                results = parallel(joblib.delayed(evalFunc)(policies[j], params) for j in range(sample_num))
                labels.append(results)
                task_id.append([i]*sample_num)
        result = {'sampled_policies':np.concatenate(sampled_policies), 'labels':np.concatenate(labels), 'task_id':np.concatenate(task_id), 'groups':params['groups'], 'centers':params['centers']}
        with open(f'./training_data_{dataset}_{sample_num}.pkl', 'wb') as f:
            pickle.dump(result, f)
    return sampled_policies, labels, task_id

def cluster_data(feat_list, label_list, n_clusters):
    skf = StratifiedShuffleSplit(n_splits=n_clusters,test_size=1-1/n_clusters)
    groups = []
    for i, (train_index, test_index) in enumerate(skf.split(feat_list, label_list)):
        groups.append(train_index)
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    return groups, centers


def normalize_sampling_weights(scores):
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, 0.0, None)
    total_score = np.sum(scores)
    if total_score <= 0.0:
        return np.full(scores.shape, 1.0 / len(scores))
    return scores / total_score


def uncertainty_scores(probabilities, label_list, uncertainty):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    entropy = np.sum(-probabilities * np.log(probabilities), axis=1)
    if uncertainty == 'entropy':
        return entropy

    true_class_probabilities = probabilities[np.arange(probabilities.shape[0]), label_list]
    negative_log_likelihood = -np.log(true_class_probabilities)
    if uncertainty == 'nll':
        return negative_log_likelihood
    if uncertainty == 'product':
        return negative_log_likelihood * entropy
    raise ValueError(f'unsupported uncertainty metric: {uncertainty}')


def sample_weight_centers(weights, center_count, diff_c):
    draw_count = center_count if diff_c else 1
    center_indices = np.random.choice(len(weights), draw_count, p=weights)
    if not diff_c:
        center_indices = np.repeat(center_indices, center_count)
    return np.asarray(center_indices)

def rank_histogram(rank_values, bins=10):
    counts, _ = np.histogram(rank_values, bins=bins, range=(0.0, 1.0))
    return counts.astype(int).tolist()


def pairwise_iou(groups):
    ratios = []
    for i in range(len(groups)):
        left = set(np.asarray(groups[i]).tolist())
        for j in range(i + 1, len(groups)):
            right = set(np.asarray(groups[j]).tolist())
            union_size = len(left | right)
            ratios.append(0.0 if union_size == 0 else len(left & right) / union_size)
    return ratios


def log_weighted_cluster_diagnostics(
    probability_ratios,
    rank_means,
    rank_histograms,
    center_ranks,
    uncertainty,
    subset_sigma,
):
    logger = logging.getLogger('MFC')
    ratio_text = ', '.join(f'{ratio:.3g}' for ratio in probability_ratios)
    rank_text = ', '.join(f'{rank_mean:.3f}' for rank_mean in rank_means)
    center_text = ', '.join(f'{center_rank:.3f}' for center_rank in center_ranks)
    logger.info(
        'MFC subset sampling diagnostics | uncertainty=%s | subset_sigma=%.3g | '
        'probability max/min=[%s] | center ranks=[%s] | sampled mean ranks=[%s] | rank histograms=%s',
        uncertainty,
        subset_sigma,
        ratio_text,
        center_text,
        rank_text,
        rank_histograms,
    )


def log_group_overlap_diagnostics(groups, true_groups):
    logger = logging.getLogger('MFC')
    sampled_iou = pairwise_iou(groups)
    assigned_iou = pairwise_iou(true_groups)
    logger.info(
        'MFC subset overlap diagnostics | sampled pairwise IoU=%s | assigned pairwise IoU=%s',
        [round(value, 4) for value in sampled_iou],
        [round(value, 4) for value in assigned_iou],
    )


def cluster_data_weighted(feat_list, label_list, n_clusters, diff_c, uncertainty='entropy', subset_sigma=0.15):
    sample_num = int(feat_list.shape[0]/n_clusters)
    # sample_num = int(feat_list.shape[0] * 0.8)
    sample_counts = n_clusters
    label_list = np.array(label_list)
    if isinstance(feat_list, torch.Tensor):
        feat_list = feat_list.cpu().numpy()

    scores = uncertainty_scores(feat_list, label_list, uncertainty)
    weights = normalize_sampling_weights(scores)
    rank_u = rankdata(scores, method='average') / len(scores)
    center_indices = sample_weight_centers(weights, sample_counts, diff_c)
    mu = rank_u[center_indices]
    groups = []
    probability_ratios = []
    rank_means = []
    rank_histograms = []
    for i in range(sample_counts):
        w = np.exp(- (rank_u - mu[i]) ** 2 / (2 * subset_sigma ** 2))
        w = w / np.sum(w)
        probability_ratios.append(float(np.max(w) / np.min(w)))
        # _, idx = np.unique(label_list, return_index=True)
        # weights = softmax(1 - softmax(feat_list, axis=1), axis=1)
        # weights = [softmax(weights[idx[i], i]) for i in range(len(idx))]
        # _, counts = np.unique(label_list, return_counts=True)
        # counts = counts/np.sum(counts)
        # sample_num = counts*sample_num
        # groups.append(np.concatenate([np.random.choice(idx[i], int(sample_num[i]), p=weights[i]) for i in range(len(idx))]))
        group = np.random.choice(np.arange(label_list.shape[0]), int(sample_num), p=w)
        groups.append(group)
        rank_means.append(float(np.mean(rank_u[group])))
        rank_histograms.append(rank_histogram(rank_u[group]))
    log_weighted_cluster_diagnostics(
        probability_ratios,
        rank_means,
        rank_histograms,
        mu,
        uncertainty,
        subset_sigma,
    )
    centers = [np.mean(feat_list[groups[i]], axis=0) for i in range(n_clusters)]
    # 按与聚类中心的距离再次分组
    sampled_indices = np.unique(np.concatenate(groups))
    sampled_features = feat_list[sampled_indices]
    distances = np.zeros((sampled_features.shape[0], n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(sampled_features - centers[i], axis=1)

    # 仅将已采样样本分配到距离最近的聚类中心，未采样样本不进入 true_groups。
    nearest_cluster = np.argmin(distances, axis=1)
    true_groups = [sampled_indices[nearest_cluster == i] for i in range(n_clusters)]
    log_group_overlap_diagnostics(groups, true_groups)
    # groups_weights = [np.sum(weights[groups[i]]) for i in range(len(groups))]
    # idx = np.argmax(groups_weights)
    # return groups[idx], centers

    # intersection = reduce(np.intersect1d, groups)
    # union = reduce(np.union1d, groups)
    # ratio = len(intersection)/len(union)
    # return groups, centers, intersection, ratio

    return groups, centers, true_groups


def combine_feature_batches(feature_batches, class_batches, use_gpu):
    if use_gpu:
        return torch.cat(feature_batches, dim=0), torch.cat(class_batches, dim=0)
    return (
        torch.cat(feature_batches, dim=0).cpu().numpy(),
        torch.cat(class_batches, dim=0).cpu().numpy(),
    )


def get_pca_component_count(dataset, feature_dim):
    ratio = 0.05 if 'breakhis' in dataset else 0.01
    return int(ratio * feature_dim)


def build_feature_reducer(args, feature_dim):
    n_components = get_pca_component_count(args.dataset, feature_dim)
    if args.gpu:
        from torch_pca import PCA as PCA_torch

        return PCA_torch(n_components=n_components)

    from sklearn.decomposition import PCA

    return PCA(n_components=n_components)


def reduce_features(args, feat_list):
    pca = build_feature_reducer(args, feat_list[0].shape[0])
    return pca.fit_transform(feat_list), pca


def to_numpy_features(feat_list):
    if isinstance(feat_list, torch.Tensor):
        return feat_list.detach().cpu().numpy()
    return np.asarray(feat_list)


def allocate_representative_counts(sample_size):
    center_count = int(round(sample_size * 0.5))
    middle_count = int(round(sample_size * 0.3))
    boundary_count = sample_size - center_count - middle_count
    return center_count, middle_count, boundary_count


def select_unique_ranked_indices(candidate_indices, selected, quota):
    output = []
    for idx in candidate_indices:
        idx = int(idx)
        if idx in selected:
            continue
        selected.add(idx)
        output.append(idx)
        if len(output) == quota:
            break
    return output


def representative_group_indices(feat_list, group_indices, sample_ratio):
    group_indices = np.asarray(group_indices)
    if not 0.0 < sample_ratio < 1.0:
        raise ValueError('sample_ratio must be greater than 0 and less than 1')

    sample_size = max(1, int(np.ceil(len(group_indices) * sample_ratio)))
    if len(group_indices) <= sample_size:
        return group_indices.copy()

    features = to_numpy_features(feat_list)[group_indices]
    center = features.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(features - center, axis=1)
    sorted_positions = np.argsort(distances)

    center_count, middle_count, boundary_count = allocate_representative_counts(sample_size)
    middle_rank = (len(sorted_positions) - 1) / 2.0
    middle_positions = sorted_positions[np.argsort(np.abs(np.arange(len(sorted_positions)) - middle_rank))]

    selected_positions = set()
    chosen_positions = []
    chosen_positions.extend(select_unique_ranked_indices(sorted_positions, selected_positions, center_count))
    chosen_positions.extend(select_unique_ranked_indices(middle_positions, selected_positions, middle_count))
    chosen_positions.extend(select_unique_ranked_indices(sorted_positions[::-1], selected_positions, boundary_count))

    if len(chosen_positions) < sample_size:
        chosen_positions.extend(
            select_unique_ranked_indices(
                sorted_positions,
                selected_positions,
                sample_size - len(chosen_positions),
            )
        )

    return group_indices[np.array(chosen_positions)]


def uniform_group_indices(group_indices, sample_ratio, seed):
    group_indices = np.asarray(group_indices)
    if not 0.0 < sample_ratio < 1.0:
        raise ValueError('sample_ratio must be greater than 0 and less than 1')

    sample_size = max(1, int(np.ceil(len(group_indices) * sample_ratio)))
    if len(group_indices) <= sample_size:
        return group_indices.copy()

    rng = np.random.default_rng(seed)
    selected_positions = rng.permutation(len(group_indices))[:sample_size]
    return group_indices[selected_positions]


def build_representative_groups(feat_list, groups, sample_ratio):
    return [representative_group_indices(feat_list, group, sample_ratio) for group in groups]


def build_uniform_eval_groups(groups, sample_ratio, seed):
    return [
        uniform_group_indices(group, sample_ratio, seed + group_id)
        for group_id, group in enumerate(groups)
    ]


def build_eval_groups(feat_list, groups, sample_ratio, sampling, seed):
    if sampling == 'representative':
        return build_representative_groups(feat_list, groups, sample_ratio)
    if sampling == 'uniform':
        return build_uniform_eval_groups(groups, sample_ratio, seed)
    raise ValueError(f'unsupported eval sampling mode: {sampling}')


def log_eval_sampling_diagnostics(groups, eval_groups, sample_ratio, sampling, seed):
    full_sizes = [len(group) for group in groups]
    eval_sizes = [len(group) for group in eval_groups]
    logger = logging.getLogger('MFC')
    logger.info(
        'MFC eval sampling diagnostics | mode=%s | ratio=%.3g | seed=%s | full sizes=%s | eval sizes=%s',
        sampling,
        sample_ratio,
        seed,
        full_sizes,
        eval_sizes,
    )


def build_search_bounds(total_op_num, num_ops, mag_bin, prob_bin, use_prob):
    n_dims = 3 if use_prob else 2
    var_dim = num_ops * n_dims
    lb = np.array([0] * var_dim)
    ub = [total_op_num - 1] * num_ops + [mag_bin - 1] * num_ops
    if use_prob:
        ub += [prob_bin - 1] * num_ops
    return lb, np.array(ub), n_dims, var_dim


def build_search_tasks(groups, num_ops, n_dims, lb, ub, use_bayes):
    if not use_bayes:
        raise ValueError('legacy non-Bayes MFC search was removed; enable --bayes')
    var_dim = num_ops * n_dims
    return [SingleTask(var_dim, lb, ub, [0] * var_dim, evalFuncBayes) for _ in range(len(groups))]


def build_mfc_params(
    model,
    data_list,
    feat_list,
    groups,
    centers,
    pca,
    lb,
    ub,
    args,
    resize_size,
    num_ops,
    mag_bin,
    prob_bin,
):
    eval_sample_ratio = getattr(args, 'mfc_eval_sample_ratio', 0.2)
    eval_sampling = getattr(args, 'mfc_eval_sampling', 'representative')
    eval_sample_seed = getattr(args, 'mfc_eval_sample_seed', 0)
    eval_groups = build_eval_groups(feat_list, groups, eval_sample_ratio, eval_sampling, eval_sample_seed)
    log_eval_sampling_diagnostics(groups, eval_groups, eval_sample_ratio, eval_sampling, eval_sample_seed)
    return {
        'feat_extractor': model,
        'data_list': data_list,
        'feat_list': feat_list,
        'batch_size': args.batch_size,
        'groups': groups,
        'full_groups': groups,
        'eval_groups': eval_groups,
        'eval_group': eval_groups,
        'centers': centers,
        'pca': pca,
        'Lb': lb,
        'Ub': ub,
        'w': 0,
        'n_op': num_ops,
        'mag_bin': mag_bin,
        'prob_bin': prob_bin,
        'args': args,
        'resize_size': resize_size,
        'model': model,
    }


def default_search_options():
    return {'popsize': 30, 'maxgen': 2, 'rmp': 0.3, 'reps': 2, 'proxy_update': 10}


def create_policy_writers(args, task_count, summary_writer_cls):
    currtime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    log_name = f'MFCAugment-{currtime}-{args.num_ops}'
    return [
        summary_writer_cls(log_dir=os.path.join(args.log_path, args.save_name, log_name, f'task{x}'))
        for x in range(task_count)
    ]


def close_policy_writers(writers):
    for writer in writers:
        writer.close()


def run_policy_search(args, tasks, options, params, writer):
    if args.multitask:
        from EA.MFSBX import MFSBX

        if args.generative:
            if args.proxy:
                from EA.GDPMFPSO import GDPMFPSO

                return GDPMFPSO(tasks, options, params, writer), None

            from EA.GDMFPSO import GDMFPSO

            return GDMFPSO(tasks, options, params, writer), None

        best_pop, skill_factor, best_ind = MFSBX(tasks, options, params, writer)
        return best_pop, skill_factor

    if args.bayes:
        best_topk = resolve_bayes_topk_count(args.bayes_topk, args.bayes_max_eval)
        if getattr(args, 'policy_pool_source', 'search') == 'random':
            best_policy = random_policy_pool_for_tasks(
                task_count=len(tasks),
                args=args,
                policy_count=best_topk,
            )
            return best_policy, None
        best_policy = bayesian_optimization_tasks_parallel(
            tasks,
            args,
            params,
            rep=args.bayes_rep,
            topk=best_topk,
            max_evals=args.bayes_max_eval,
        )
        return best_policy, None

    from EA.SBX import SBX

    best_pop, skill_factor, best_ind = SBX(tasks, options, params, writer)
    return best_pop, skill_factor


def resolve_best_policy(args, params, search_result, skill_factor):
    if args.bayes:
        return search_result
    if args.group:
        return formatPolicy(params, search_result, skill_factor)
    return formatPolicy(params, search_result)


def log_best_policy(writer, best_policy):
    for i, policy in enumerate(best_policy):
        writer[0].add_text('Best Policy', str(policy), i)


def MFCAugment(model, resize_size, data_list, label_list, args, n_clusters, mag_bin=31, prob_bin=10, num_ops=2, max_samples=100):
    from core.augmentations import augmentation_space
    from tensorboardX import SummaryWriter

    total_op_num = len(augmentation_space())
    feat_batches, cls_batches = getdatafeat(args, resize_size, data_list, model)
    feat_list, cls_list = combine_feature_batches(feat_batches, cls_batches, use_gpu=args.gpu)
    groups, centers, true_groups = cluster_data_weighted(
        cls_list,
        label_list,
        n_clusters,
        diff_c=args.diff_c,
        uncertainty=args.uncertainty,
        subset_sigma=args.subset_sigma,
    )
    centers = []
    feat_list, pca = reduce_features(args, feat_list)
    lb, ub, n_dims, var_dim = build_search_bounds(total_op_num, num_ops, mag_bin, prob_bin, args.use_prob)
    tasks = build_search_tasks(groups, num_ops, n_dims, lb, ub, args.bayes)
    params = build_mfc_params(
        model,
        data_list,
        feat_list,
        groups,
        centers,
        pca,
        lb,
        ub,
        args,
        resize_size,
        num_ops,
        mag_bin,
        prob_bin,
    )
    options = default_search_options()
    writer = create_policy_writers(args, len(tasks), SummaryWriter)
    try:
        search_result, skill_factor = run_policy_search(args, tasks, options, params, writer)
        bestPolicy = resolve_best_policy(args, params, search_result, skill_factor)
        log_best_policy(writer, bestPolicy)
        return bestPolicy, groups, true_groups
    finally:
        close_policy_writers(writer)
    
def formatPolicy(param, bestPop, skillFactor=None,verbose=False):
    Ub = param['Ub']
    Lb = param['Lb']
    args = param['args']
    n_op = param['n_op']
    formattedPolicyOut = []
    if isinstance(bestPop, np.ndarray):
        bestPolicy = [[np.floor(bestPop[i,j].pbest*(Ub-Lb)+Lb)] for i in range(bestPop.shape[0]) for j in range(bestPop.shape[1])]
        bestPolicy = np.array(bestPolicy)
        if len(bestPolicy.shape) > 2:
            bestPolicy = bestPolicy.squeeze()
        if len(bestPolicy.shape) < 2:
            bestPolicy = bestPolicy.reshape(1,-1)
    else:
        bestPolicy = np.floor(bestPop.rnvec.T*(Ub-Lb)+Lb).reshape(1,-1)
    if skillFactor!=None:
        skillFactor = np.array(skillFactor).reshape(1,-1)
        for k in range(skillFactor.max()+1):
            if skillFactor.size == 1:
                idx = 0
            else:
                idx = np.where(skillFactor==k)[1]
            op_index = bestPolicy[idx,:n_op]
            mag_index = bestPolicy[idx,n_op:2*n_op]
            if args.use_prob:
                prob_index = bestPolicy[idx,2*n_op:3*n_op]
            uni_op_index = np.unique(op_index, axis=0)
            idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
            formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
            formattedPolicy['op_index'] = uni_op_index
            for i in idx:
                formattedPolicy['magnitude_index'].append(np.unique(mag_index[i,:],axis=0))
                if args.use_prob:
                    formattedPolicy['prob_index'].append(np.unique(prob_index[i,:],axis=0))
            formattedPolicyOut.append(formattedPolicy)
    else:
        if verbose:
            op_index = bestPolicy[:,:n_op]
            uni_op_index = op_index.copy()
            idx = np.arange(bestPolicy.shape[0])
        else:
            op_index = bestPolicy[:,:n_op]
            uni_op_index = np.unique(op_index, axis=0)
            idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
        formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        formattedPolicy['op_index'] = uni_op_index
        for i in idx:            
            if args.use_prob:
                if verbose:
                    formattedPolicy['prob_index'].append(bestPolicy[i,n_op:2*n_op].reshape(1,-1))
                    formattedPolicy['magnitude_index'].append(bestPolicy[i,2*n_op:3*n_op].reshape(1,-1))
                else:
                    formattedPolicy['prob_index'].append(np.unique(bestPolicy[i,n_op:2*n_op],axis=0))
                    formattedPolicy['magnitude_index'].append(np.unique(bestPolicy[i,2*n_op:3*n_op],axis=0))
            else:
                if verbose:
                    formattedPolicy['magnitude_index'].append(bestPolicy[i,n_op:2*n_op].reshape(1,-1))
                else:
                    formattedPolicy['magnitude_index'].append(np.unique(bestPolicy[i,n_op:2*n_op],axis=0))
        formattedPolicyOut.append(formattedPolicy)

    return formattedPolicyOut

def policy_decoder(augment, use_prob, n_op):
    formattedPolicy = {'op_index':[],'prob_index':[],'magnitude_index':[]}
    op_idx = []
    op_level = []
    op_prob = []
    for i in range(n_op):
        op_idx.append(augment['policy_%d' % i])
        op_level.append(augment['level_%d' % i])
        if use_prob:
            op_prob.append(augment['prob_%d' % i])
    formattedPolicy['op_index'] = np.array([op_idx])
    formattedPolicy['magnitude_index'] = np.array([op_level])
    if use_prob:
        formattedPolicy['prob_index'] = np.array([op_prob])

    return formattedPolicy


def build_task_params(params, task_id):
    task_params = params.copy()
    task_params['task_id'] = task_id
    return task_params


def reevaluate_top_policies_with_full_groups(trial_history, params, topk):
    if topk <= 0 or not trial_history:
        return []

    eval_params = params.copy()
    full_groups = params.get('full_groups', params['groups'])
    eval_params['eval_groups'] = full_groups
    eval_params['eval_group'] = full_groups

    candidates = []
    for trial in trial_history[:topk]:
        full_loss = evalFuncBayes(trial['policy'], eval_params)
        candidates.append({'policy': trial['policy'], 'loss': full_loss})

    return sorted(candidates, key=lambda x: x['loss'])


def resolve_bayes_topk_count(topk_ratio_value, max_evals):
    return max(1, int(np.ceil(topk_ratio_value * max_evals)))


def random_policy_trial(rng, args, total_op_num):
    policy = {
        'op_index': rng.integers(0, total_op_num, size=(1, args.num_ops)),
        'magnitude_index': rng.integers(0, args.mag_bin - 1, size=(1, args.num_ops)),
        'prob_index': [],
    }
    if args.use_prob:
        policy['prob_index'] = rng.integers(0, args.prob_bin - 1, size=(1, args.num_ops))
    return {'policy': policy, 'loss': 0.0}


def random_policy_pool_for_tasks(task_count, args, policy_count):
    from core.augmentations_fastaa import augment_list

    rng = np.random.default_rng(getattr(args, 'policy_pool_seed', 0))
    total_op_num = len(augment_list())
    return [
        merge_trial_policies(
            [random_policy_trial(rng, args, total_op_num) for _ in range(policy_count)],
            args.use_prob,
        )
        for _ in range(task_count)
    ]


def select_final_trial_history(trial_history, args, params, topk):
    if topk >= len(trial_history):
        return trial_history
    if getattr(args, 'reevaluate_full_groups', True):
        return reevaluate_top_policies_with_full_groups(trial_history, params, topk)
    if topk <= 0:
        return []
    return trial_history[:topk]


def merge_trial_policies(trial_history, use_prob):
    if not trial_history:
        return {'op_index': np.empty((0, 0), dtype=int), 'prob_index': [], 'magnitude_index': []}

    op_rows = np.vstack([np.asarray(trial['policy']['op_index']).reshape(1, -1) for trial in trial_history])
    magnitude_rows = np.vstack([
        np.asarray(trial['policy']['magnitude_index']).reshape(1, -1)
        for trial in trial_history
    ])
    unique_ops, inverse = np.unique(op_rows, axis=0, return_inverse=True)
    final_policies = {'op_index': unique_ops, 'prob_index': [], 'magnitude_index': []}

    if use_prob:
        probability_rows = np.vstack([
            np.asarray(trial['policy']['prob_index']).reshape(1, -1)
            for trial in trial_history
        ])

    for op_group in range(len(unique_ops)):
        matches = inverse == op_group
        final_policies['magnitude_index'].append(np.unique(magnitude_rows[matches], axis=0))
        if use_prob:
            final_policies['prob_index'].append(np.unique(probability_rows[matches], axis=0))

    return final_policies

def bayesian_optimization_tasks(tasks, args, params, max_evals=200):
    from hyperopt import Trials, fmin, hp, tpe
    from core.augmentations_fastaa import augment_list

    """
    使用HyperOpt的贝叶斯优化来求解任务
    
    Parameters:
    tasks : list
        任务列表，每个任务应具有evaluate方法
    params : dict
        参数字典，传递给任务评估函数
    max_evals : int
        最大评估次数
    
    Returns:
    best_policy : list
        最优策略列表
    skill_factors : list
        技能因子列表
    best_ind : list
        最佳个体列表
    """
    
    # 存储每个任务的结果
    task_results = []

    # 对每个任务分别进行贝叶斯优化
    for task_idx, task in enumerate(tasks):
        trial_history = []
        print(f"Optimizing task {task_idx+1}/{len(tasks)}")
        params['task_id'] = task_idx
        # 定义搜索空间
        space = {}
        for i in range(args.num_ops):
            space['policy_%d' % i] = hp.choice('policy_%d' % i, list(range(0, len(augment_list()))))
            if args.use_prob:                
                space['prob_%d' % i] = hp.choice('prob_%d' % i, list(range(0, args.prob_bin-1)))
            space['level_%d' % i] = hp.choice('level_%d' % i, list(range(0, args.mag_bin-1)))
        
        # 定义目标函数
        def objective(x):
            # 将字典转换为数组
            policy = policy_decoder(x, args.use_prob, args.num_ops)
            # 评估策略
            loss = task.evaluate(policy, params)
            trial = {'policy':policy,'loss':loss}
            trial_history.append(trial)
            return loss
        
        x = {
            'policy_0': 1,
            'level_0': 1,
            'policy_1': 1,
            'level_1': 1
        }

        objective(x)
        # 执行贝叶斯优化
        trials = Trials()
        best = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=max_evals,
                   trials=trials,
                   show_progressbar=True)
        trial_history = sorted(trial_history, key=lambda x: x['loss'], reverse=False)
        trial_history = trial_history[:50]
        merged_policies = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        for r in trial_history:
            p = r['policy']
            merged_policies['op_index'].append(p['op_index'])
            merged_policies['magnitude_index'].append(p['magnitude_index'])
            if args.use_prob:
                merged_policies['prob_index'].append(p['prob_index'])
        final_policies = {'op_index':[],'prob_index':[],'magnitude_index':[]}
        op_index = merged_policies['op_index']
        uni_op_index = np.unique(op_index, axis=0)
        idx = [np.where((uni_op_index[i,:]==op_index).all(-1))[0] for i in range(uni_op_index.shape[0])]
        final_policies['op_index'] = uni_op_index
        mag_idx = np.array(merged_policies['magnitude_index']).squeeze()
        prob_idx = np.array(merged_policies['prob_index']).squeeze()
        for i in idx:            
            if args.use_prob:
                final_policies['prob_index'].append(np.unique(prob_idx[i,:],axis=0))
                final_policies['magnitude_index'].append(np.unique(mag_idx[i,:],axis=0))
            else:
                final_policies['magnitude_index'].append(np.unique(mag_idx[i,:],axis=0))
        # 获取最优策略
        task_results.append(final_policies)
        print(f"Task {task_idx+1} best loss: {trials.best_trial['result']['loss']}")
    
    return task_results

def bayesian_optimization_tasks_parallel(tasks, args, params, rep=1, topk=100, max_evals=200):
    """
    使用HyperOpt的贝叶斯优化来求解任务
    
    Parameters:
    tasks : list
        任务列表，每个任务应具有evaluate方法
    params : dict
        参数字典，传递给任务评估函数
    max_evals : int
        最大评估次数
    
    Returns:
    best_policy : list
        最优策略列表
    skill_factors : list
        技能因子列表
    best_ind : list
        最佳个体列表
    """
    
    # 存储每个任务的结果
    final_results = []
    for r in range(rep):
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    bayesian_optimization_single_task,
                    task_idx,
                    args,
                    task,
                    build_task_params(params, task_idx),
                    rep,
                    topk,
                    max_evals,
                ): task_idx
                for task_idx, task in enumerate(tasks)
            }
            
            # 收集结果
            results = [None] * len(tasks)
            task_times = [None] * len(tasks)
            task_best_loss = [None] * len(tasks)
            for future in as_completed(future_to_task):
                result, task_idx, elapsed_time, best_loss = future.result()
                results[task_idx] = result
                task_times[task_idx] = elapsed_time
                task_best_loss[task_idx] = best_loss
            final_results.extend(results)
            print("\n=== Task Execution Summary ===")
            for i in range(len(tasks)):
                print(f'Rep {r+1} Task {i+1}: Time = {task_times[i]:.2f}s, Best Loss = {task_best_loss[i]:.4f}')   
    return final_results

def bayesian_optimization_single_task(task_idx, args, task, params, rep=1, topk=100, max_evals=200):
    from hyperopt import Trials, fmin, hp, tpe
    from core.augmentations_fastaa import augment_list

    """
    对单个任务使用贝叶斯优化
    
    Parameters:
    task : SingleTask
        单个任务对象
    params : dict
        参数字典
    max_evals : int
        最大评估次数
    
    Returns:
    best_policy : array
        最优策略
    best_loss : float
        最佳损失值
    """
    
    st = time.time()
    # print(f"Optimizing task {task_idx+1}")
    # 定义搜索空间
    space = {}
    for i in range(args.num_ops):
        space['policy_%d' % i] = hp.choice('policy_%d' % i, list(range(0, len(augment_list()))))
        if args.use_prob:                
            space['prob_%d' % i] = hp.choice('prob_%d' % i, list(range(0, args.prob_bin-1)))
        space['level_%d' % i] = hp.choice('level_%d' % i, list(range(0, args.mag_bin-1)))
    
    # 定义目标函数
    trial_history = []
    def objective(x):
        # 将字典转换为数组
        policy = policy_decoder(x, args.use_prob, args.num_ops)
        # 评估策略
        loss = task.evaluate(policy, params)
        trial = {'policy':policy,'loss':loss}
        trial_history.append(trial)
        return loss
    
    # x = {
    #     'policy_0': 1,
    #     'level_0': 1,
    #     'policy_1': 1,
    #     'level_1': 1
    # }

    # objective(x)
    # 执行贝叶斯优化
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                show_progressbar=False,
                verbose=False)
    trial_history = sorted(trial_history, key=lambda x: x['loss'], reverse=False)
    final_trial_history = select_final_trial_history(trial_history, args, params, topk)
    final_policies = merge_trial_policies(final_trial_history, args.use_prob)
    best_loss = final_trial_history[0]['loss'] if final_trial_history else trial_history[0]['loss']
    elapsed_time = time.time() - st
    return final_policies, task_idx, elapsed_time, best_loss









