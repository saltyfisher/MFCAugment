import numpy as np
import torch
import os
import logging
import subprocess
import time
import torchvision
import argparse
import models
from dataclasses import dataclass
from torchvision import transforms
# from sklearnex import patch_sklearn
from pathlib import Path
# from sklearn.metrics import f1_score,precision_score,recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from networks import num_class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common import get_logger
import csv

logger = get_logger('MFC')
logger.setLevel(logging.DEBUG)


@dataclass
class OutputConfig:
    save_path: Path
    log_path: Path
    save_name: str


def sample_ratio(value):
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError('sample ratio must be greater than 0 and less than 1')
    return ratio


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError('value must be greater than 0')
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be greater than 0')
    return parsed


def topk_ratio(value):
    ratio = float(value)
    if not 0.0 < ratio <= 1.0:
        raise argparse.ArgumentTypeError('top-k ratio must be greater than 0 and less than or equal to 1')
    return ratio


def resolve_bayes_topk_count(topk_ratio_value, max_evals):
    return max(1, int(np.ceil(topk_ratio_value * max_evals)))


def parse_bool_value(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean value: {value}')


def parse_uncertainty_value(value):
    value = str(value)
    if value not in {'entropy', 'nll', 'product'}:
        raise argparse.ArgumentTypeError(f'invalid uncertainty metric: {value}')
    return value


def parse_eval_sampling_value(value):
    value = str(value)
    if value not in {'representative', 'uniform'}:
        raise argparse.ArgumentTypeError(f'invalid eval sampling mode: {value}')
    return value


def parse_policy_pool_source(value):
    value = str(value)
    if value not in {'search', 'random'}:
        raise argparse.ArgumentTypeError(f'invalid policy pool source: {value}')
    return value


def parse_mfc_eval_metric(value):
    value = str(value).lower()
    if value not in {'kl', 'mmd'}:
        raise argparse.ArgumentTypeError(f'invalid MFC eval metric: {value}')
    return value


SUPPORTED_DATASETS = ('chestct', 'breakhis', 'corona', 'cifar-fs', 'miniimagenet')
BREAKHIS_MAGNIFICATIONS = ('40', '100', '200', '400')
DEFAULT_BREAKHIS_MAGNIFICATION = '40'


def parse_dataset_value(value):
    value = str(value).lower()
    if value not in SUPPORTED_DATASETS:
        raise argparse.ArgumentTypeError(f'invalid dataset: {value}')
    return value


def parse_magnification_value(value):
    if value is None:
        return None
    value = str(value)
    if value.lower() in {'none', 'null'}:
        return None
    if value not in BREAKHIS_MAGNIFICATIONS:
        raise argparse.ArgumentTypeError(f'invalid BreakHis magnification: {value}')
    return value


def as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def resolve_dataset_magnification_pairs(datasets, magnifications):
    datasets = as_list(datasets)
    magnifications = as_list(magnifications)
    non_empty_magnifications = [value for value in magnifications if value is not None]
    has_breakhis = any(dataset == 'breakhis' for dataset in datasets)
    if non_empty_magnifications and not has_breakhis:
        raise ValueError('magnification can only be used when dataset includes breakhis')

    breakhis_magnifications = (
        non_empty_magnifications
        if non_empty_magnifications
        else [DEFAULT_BREAKHIS_MAGNIFICATION]
    )
    pairs = []
    seen = set()
    for dataset in datasets:
        if dataset == 'breakhis':
            candidate_pairs = [('breakhis', magnification) for magnification in breakhis_magnifications]
        else:
            candidate_pairs = [(dataset, None)]
        for pair in candidate_pairs:
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def expand_dataset_magnification_args(args):
    variants = []
    for dataset, magnification in resolve_dataset_magnification_pairs(args.dataset, args.magnification):
        variant_args = argparse.Namespace(**vars(args))
        variant_args.dataset = dataset
        variant_args.magnification = magnification
        variants.append(variant_args)
    return variants


SUPPORTED_PARAMETER_TESTS = (
    'mfc_eval_sample_ratio',
    'mfc_eval_sampling',
    'mfc_eval_metric',
    'policy_pool_source',
    'group',
    'diff_c',
    'l',
    'mfc_refresh_interval',
    'uncertainty',
    'subset_sigma',
)
PARAMETER_TEST_DEFAULT_VALUES = {
    'mfc_eval_sample_ratio': [0.1, 0.2, 0.5],
    'mfc_eval_sampling': ['representative', 'uniform'],
    'mfc_eval_metric': ['kl', 'mmd'],
    'policy_pool_source': ['search', 'random'],
    'group': [False, True],
    'diff_c': [False, True],
    'l': [1, 2, 5],
    'mfc_refresh_interval': [20, 40, 80],
    'uncertainty': ['entropy', 'nll', 'product'],
    'subset_sigma': [0.1, 0.15, 0.2],
}
PARAMETER_TEST_CONVERTERS = {
    'mfc_eval_sample_ratio': sample_ratio,
    'mfc_eval_sampling': parse_eval_sampling_value,
    'mfc_eval_metric': parse_mfc_eval_metric,
    'policy_pool_source': parse_policy_pool_source,
    'group': parse_bool_value,
    'diff_c': parse_bool_value,
    'l': int,
    'mfc_refresh_interval': positive_int,
    'uncertainty': parse_uncertainty_value,
    'subset_sigma': positive_float,
}


def parse_parameter_test_values(param_name, raw_values):
    if param_name not in SUPPORTED_PARAMETER_TESTS:
        raise ValueError(f'unsupported parameter test: {param_name}')

    values = PARAMETER_TEST_DEFAULT_VALUES[param_name] if raw_values is None else raw_values
    converter = PARAMETER_TEST_CONVERTERS[param_name]
    return [converter(value) for value in values]


def build_parser():
    parser = argparse.ArgumentParser(description='Medical Image Classification with UncertaintyMixup')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassification/',
                        help='数据集目录路径')
    parser.add_argument('--model', type=str, default='resnet18', help='模型选择')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=180, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--strategy', type=str, default='', help='训练策略')
    parser.add_argument('--dataset', type=parse_dataset_value, nargs='+', default=['chestct'], help='数据集类型')
    parser.add_argument('--magnification', type=parse_magnification_value, nargs='+', default=[None],
                        choices=list(BREAKHIS_MAGNIFICATIONS) + [None],
                        help='BreakHis数据集的放大倍数，未指定时默认使用40X')
    parser.add_argument('--test_split', type=float, default=0.2, help='BreakHis数据集的测试集比例')
    parser.add_argument('--num_trials', type=int, default=10, help='独立实验次数')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--save_mixed_results', action='store_true', help='是否保存混合结果可视化图像')
    parser.add_argument('--gpu', action='store_true', help='是否在显存上进行计算')
    parser.add_argument('--pretrain', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--test_all', action='store_true', help='是否测试所有方法')
    parser.add_argument('--resize', action='store_false', help='搜索增广策略时是否缩放')
    parser.add_argument('--mfc', action='store_true', help='是否优化增广策略')
    parser.add_argument('--online', action='store_true', help='是否在线优化增广策略')
    parser.add_argument('--proxy', action='store_true', help='是否使用代理')
    parser.add_argument('--GD', action='store_true', help='是否使用生成式模型')
    parser.add_argument('--num_ops', type=int, default=2, help='增广策略中的操作数')
    parser.add_argument('--use_prob', action='store_true', help='增广策略中是否需要激活概率')
    parser.add_argument('--multitask', action='store_true', help='是否启用多任务算法')
    parser.add_argument('--generative', action='store_true', help='是否在多任务算法中启用生成式模型')
    parser.add_argument('--bayes', action='store_true', help='是否在多任务算法中启用贝叶斯优化')
    parser.add_argument('--bayes_max_eval', type=int, default=100, help='贝叶斯优化最大迭代次数')
    parser.add_argument('--bayes_topk', type=topk_ratio, default=0.1,
                        help='贝叶斯优化候选策略复评比例，取值范围为(0, 1]')
    parser.add_argument('--bayes_rep', type=int, default=2, help='贝叶斯优化重复次数')
    parser.add_argument('--policy_pool_source', type=parse_policy_pool_source, default='search',
                        help='策略池来源：search 使用目标函数搜索，random 使用同规模均匀随机策略池')
    parser.add_argument('--policy_pool_seed', type=int, default=0,
                        help='均匀随机策略池的随机种子')
    parser.add_argument('--reevaluate_full_groups', type=parse_bool_value, default=True,
                        help='Bayes搜索结束后是否对top-k策略做全样本复评')
    parser.add_argument('--no_reevaluate_full_groups', dest='reevaluate_full_groups', action='store_false',
                        help='关闭Bayes top-k策略的全样本复评')
    parser.add_argument('--mfc_eval_sample_ratio', type=sample_ratio, default=0.2,
                        help='Bayes搜索阶段每个子集使用的代表样本比例，取值范围为(0, 1)')
    parser.add_argument('--mfc_eval_sampling', type=parse_eval_sampling_value, default='uniform',
                        help='Bayes搜索阶段评估子集采样方式')
    parser.add_argument('--mfc_eval_metric', type=parse_mfc_eval_metric, default='kl',
                        help='Bayes搜索阶段增广数据分布评估函数：kl或mmd')
    parser.add_argument('--mfc_eval_sample_seed', type=int, default=0,
                        help='Bayes搜索阶段均匀评估子采样随机种子')
    parser.add_argument('--mfc_refresh_interval', type=positive_int, default=40,
                        help='在线MFC策略刷新周期；默认40个epoch')
    parser.add_argument('--group', action='store_true', help='每个数据子集是否单独适配增广策略')
    parser.add_argument('--diff_c', action='store_true', help='每个数据子集的中心点是否不同')
    parser.add_argument('--uncertainty', type=parse_uncertainty_value, default='entropy',
                        help='构造难度子集使用的不确定性度量')
    parser.add_argument('--subset_sigma', type=positive_float, default=0.15,
                        help='秩空间子集采样高斯带宽')
    parser.add_argument('--l', type=int, default=1, help='目标函数权重')
    parser.add_argument('--mag_bin', type=int, default=31, help='变换操作强度离散个数')
    parser.add_argument('--prob_bin', type=int, default=10, help='变换概率离散个数')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--param_test', type=str, default='', choices=SUPPORTED_PARAMETER_TESTS,
                        help='测试单个参数变化对最终结果的影响')
    parser.add_argument('--param_values', nargs='+', default=None,
                        help='单参数测试的候选值；省略时使用内置默认候选值')
    return parser


def get_cluster_count(dataset):
    if 'lym' in dataset:
        return 2
    if 'breakhis' in dataset:
        return 8
    if dataset in {'cifar-fs', 'miniimagenet'}:
        return 10
    if 'chestct' in dataset:
        return 4
    return 4


def should_refresh_policy(args, epoch, epoch_start):
    if not args.online:
        return epoch == epoch_start
    if args.testing:
        return epoch == epoch_start
    return epoch % args.mfc_refresh_interval == 0


def build_idx_to_group(groups, require_disjoint=True):
    idx_to_group = {}
    for group_idx, group_members in enumerate(groups):
        for idx in group_members:
            idx = int(idx)
            if require_disjoint and idx in idx_to_group:
                raise ValueError(f'sample index {idx} appears in multiple groups')
            idx_to_group[idx] = group_idx
    return idx_to_group


def build_policy_transforms(policy_subset, resize_size, args):
    from core.augmentations import MyAugment

    if args.resize:
        return [
            torchvision.transforms.Compose([
                transforms.Resize(resize_size),
                MyAugment(p, mag_bin=args.mag_bin, prob_bin=args.prob_bin, num_ops=args.num_ops),
                transforms.ToTensor(),
            ])
            for p in policy_subset
        ]
    return [
        torchvision.transforms.Compose([
            MyAugment(p, mag_bin=args.mag_bin, prob_bin=args.prob_bin, num_ops=args.num_ops),
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
        for p in policy_subset
    ]


def refresh_mfc_policy(model, resize_size, data_list, label_list, args):
    from core.MFCAugment import MFCAugment

    cluster_num = get_cluster_count(args.dataset)
    policy_subset, groups, true_group = MFCAugment(
        model,
        resize_size,
        data_list,
        label_list,
        args,
        n_clusters=cluster_num,
        num_ops=args.num_ops,
    )
    if policy_subset == []:
        return [], {}, []
    assignment_groups = true_group if args.group else groups
    return (
        build_policy_transforms(policy_subset, resize_size, args),
        build_idx_to_group(assignment_groups, require_disjoint=args.group),
        policy_subset,
    )


def format_name_value(value):
    if isinstance(value, float):
        value = f'{value:g}'
    value = str(value)
    return value.replace('-', 'm').replace('.', 'p').replace('/', '-').replace('\\', '-')


def add_non_default_part(parts, args, name, default, prefix=None):
    value = getattr(args, name, default)
    if value != default:
        parts.append(f'{prefix or name}{format_name_value(value)}')


def build_save_name(args):
    parts = [format_name_value(args.dataset), format_name_value(args.test_split)]
    if args.dataset == 'breakhis' and args.magnification is not None:
        parts.append(f'{format_name_value(args.magnification)}X')
    if args.mfc:
        parts.append('mfc')
    elif args.strategy != '':
        parts.append(format_name_value(args.strategy))
    parts.append(format_name_value(args.model))

    if args.online:
        parts.append('online')
    if args.testing:
        parts.append('testing')
    if args.pretrain:
        parts.append('pretrain')
    if not args.resize:
        parts.append('noresize')

    if args.mfc:
        add_non_default_part(parts, args, 'num_ops', 2, 'ops')
        if args.use_prob:
            parts.append('useprob')
        add_non_default_part(parts, args, 'mag_bin', 31, 'magbin')
        add_non_default_part(parts, args, 'prob_bin', 10, 'probbin')
        add_non_default_part(parts, args, 'l', 1, 'l')
        if args.multitask:
            parts.append('multitask')
        if args.generative:
            parts.append('generative')
        if args.bayes:
            parts.append('bayes')
            parts.extend([
                f'eval{format_name_value(args.bayes_max_eval)}',
                f'topk{format_name_value(args.bayes_topk)}',
                f'rep{format_name_value(args.bayes_rep)}',
                f'ratio{format_name_value(args.mfc_eval_sample_ratio)}',
            ])
            add_non_default_part(parts, args, 'mfc_eval_metric', 'kl', 'metric')
            if args.policy_pool_source != 'search':
                parts.append(f'pool{args.policy_pool_source}')
                add_non_default_part(parts, args, 'policy_pool_seed', 0, 'poolseed')
            if not getattr(args, 'reevaluate_full_groups', True):
                parts.append('nofullreeval')
        add_non_default_part(parts, args, 'mfc_refresh_interval', 40, 'refresh')
        if args.group:
            parts.append('group')
        if args.diff_c:
            parts.append('diffc')

    add_non_default_part(parts, args, 'batch_size', 32, 'bs')
    add_non_default_part(parts, args, 'num_epochs', 180, 'ep')
    add_non_default_part(parts, args, 'lr', 0.001, 'lr')
    add_non_default_part(parts, args, 'dropout_rate', 0.5, 'drop')
    if args.optimizer != 'adam':
        parts.append(format_name_value(args.optimizer))
    add_non_default_part(parts, args, 'momentum', 0.9, 'mom')
    add_non_default_part(parts, args, 'weight_decay', 1e-4, 'wd')

    if args.proxy:
        parts.append('proxy')
    if args.GD:
        parts.append('GD')
    return '_'.join(parts)


def prepare_output_config(args, params_root=Path('./params_save'), logs_root=Path('./logs')):
    save_path = params_root
    log_path = logs_root
    if args.mfc:
        save_path = save_path.joinpath('mfc')
        log_path = log_path.joinpath('mfc')
        args.log_path = log_path
        args.save_path = save_path
    if args.proxy:
        args.proxy_log_path = log_path.joinpath('proxy')
        args.proxy_save_path = save_path.joinpath('proxy')
    if args.GD:
        args.GD_log_path = log_path.joinpath('GD')
        args.GD_save_path = save_path.joinpath('GD')

    save_name = build_save_name(args)
    args.save_name = save_name
    return OutputConfig(save_path=save_path, log_path=log_path, save_name=save_name)


def resolve_device(args):
    if not args.gpu:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu was requested, but CUDA is not available')
    return torch.device(f'cuda:{args.device}')


def create_model(args):
    num_classes = num_class(args.dataset)
    model = models.__dict__[args.model](num_classes=num_classes)
    device = resolve_device(args)
    model.to(device)
    return model, num_classes


def create_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        print(f"Using SGD optimizer with learning rate {args.lr} and momentum {args.momentum}")
        return optimizer

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Using Adam optimizer with learning rate {args.lr}")
    return optimizer


def print_trial_banner(trial_index, total_trials):
    print(f"\n{'='*50}")
    print(f"开始第 {trial_index + 1}/{total_trials} 次独立实验")
    print(f"{'='*50}")


def run_trial(args, itrs, dataroot=None):
    if dataroot is None:
        dataroot = args.data_dir
    print_trial_banner(itrs, args.num_trials)
    model, num_classes = create_model(args)
    optimizer = create_optimizer(model, args)
    output = prepare_output_config(args)
    print(output.save_name + f'_itrs{itrs + 1}')
    os.makedirs(output.save_path, exist_ok=True)
    return train_val(
        model,
        optimizer,
        num_classes,
        args,
        itrs,
        dataroot,
        output.save_path,
        output.log_path,
        output.save_name,
        model,
    )


def build_stats_rows(all_trials_results):
    metric_names = list(all_trials_results[0].keys())
    stats = {}
    for metric in metric_names:
        values = [result[metric] for result in all_trials_results]
        stats[metric] = {
            'mean': f'{np.mean(values):.4f}',
            'std': f'{np.std(values):.4f}',
            'max': f'{np.max(values):.4f}',
            'min': f'{np.min(values):.4f}',
        }
    return stats


def print_stats_rows(stats):
    for metric, values in stats.items():
        print(f"{metric}:")
        print(f"  均值: {values['mean']}")
        print(f"  方差: {values['std']}")
        print(f"  最大值: {values['max']}")
        print(f"  最小值: {values['min']}")


def build_stats_csv_path(args):
    return Path('result') / f'{build_save_name(args)}.csv'


def update_best_metrics(metrics, best_metrics, best_f1):
    if metrics['f1'] > best_f1:
        return metrics, metrics['f1']
    return best_metrics, best_f1


def format_test_epoch_metrics(epoch, max_epoch, metrics, best_metrics):
    best_accuracy = best_metrics['accuracy'] if best_metrics is not None else metrics['accuracy']
    return (
        f'Epoch [{epoch}/{max_epoch}] - Test Loss: {metrics["loss"]:.4f}, '
        f'Test Acc: {metrics["accuracy"]:.4f}, Best Test Acc: {best_accuracy:.4f}'
    )


def write_stats_csv(csv_filename, stats):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['metric', 'mean', 'std', 'max', 'min']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for metric, values in stats.items():
            writer.writerow({
                'metric': metric,
                'mean': values['mean'],
                'std': values['std'],
                'max': values['max'],
                'min': values['min'],
            })


def summarize_trials(args, all_trials_results):
    if not all_trials_results:
        return

    print(f"\n{'='*50}")
    print("所有实验结束，统计结果如下:")
    print(f"{'='*50}")

    stats = build_stats_rows(all_trials_results)
    print_stats_rows(stats)

    csv_filename = build_stats_csv_path(args)
    write_stats_csv(csv_filename, stats)
    print(f"\n统计结果已保存到 {csv_filename}")

def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot

def run_epoch(model, loader, loss_fn, optimizer, policy, groups, args):
    cnt = 0
    steps = 0
    device = next(model.parameters()).device
    train_loss = 0.0
    all_labels = []
    all_preds = []
    skipped_augmentations = 0
    augmentation_candidates = 0
    for batch in loader:
        if groups == []:
            data, label = batch[:2]
        else:
            augmented_batch = []
            for data, label, data_idx in zip(*batch):
                if args.group:
                    p_idx = groups.get(int(data_idx))
                    if p_idx is None:
                        skipped_augmentations += 1
                        augmented_batch.append(data)
                        continue
                else:
                    p_idx = np.random.randint(0, len(policy))
                augmentation_candidates += 1
                augmented_batch.append(policy[p_idx](ToPILImage()(data)))
            data = torch.stack(augmented_batch)
            label = batch[1]

        steps += 1
        data = data.to(device) 
        label = label.to(device)
        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)
        if optimizer:
            loss.backward()
        if optimizer:
            optimizer.step()

        _, preds = preds.max(1)
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        cnt += len(data)

        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / cnt
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {}
    metrics['loss'] = train_loss
    metrics['accuracy'] = accuracy
    metrics['recall'] = recall
    metrics['precision'] = precision
    metrics['f1'] = f1
    metrics['unaugmented_ratio'] = (
        skipped_augmentations / (skipped_augmentations + augmentation_candidates)
        if skipped_augmentations + augmentation_candidates > 0
        else 0.0
    )
    if skipped_augmentations:
        logger.info(
            'MFC augmentation skipped for %d/%d samples without group assignment (ratio %.4f)',
            skipped_augmentations,
            skipped_augmentations + augmentation_candidates,
            metrics['unaugmented_ratio'],
        )

    return metrics

def train_val(model, optimizer, num_classes, args, itrs, dataroot, save_path=None, log_path=None, save_name=None, pretrain_model=None):
    from data import get_data

    
    criterion = nn.CrossEntropyLoss()
    
    max_epoch = args.num_epochs
    epoch_start = 1
    rs = {'train':[],'test':[]}
    best_f1 = -np.inf
    best_metrics = None

    traintest_dataset,test_dataset,resize_size,transform_train = get_data(
        strategy=args.strategy,
        dataset=args.dataset,
        magnification=args.magnification,
        dataroot=dataroot,
        test_split=args.test_split
        )

    traintestloader = DataLoader(traintest_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    base_transformer = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor()])
    data_list, label_list = traintest_dataset.get_all_files(base_transformer)

    policy = []
    policy_subset = []
    optimal_policy = []
    idx_to_group = []
    for epoch in range(epoch_start, max_epoch+1):
        model.train()    
        st = time.time() 
        metrics = run_epoch(model, traintestloader, criterion, optimizer, optimal_policy, idx_to_group, args)
        # print('time elapsed: %.2f' % (time.time()-st))
        rs['train'].append(metrics)
        loss = metrics['loss']
        accuracy = metrics['accuracy']
        print(f'Epoch [{epoch}/{max_epoch}] - Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}')
        model.eval()
        # scheduler.step()
        
        if (save_path is not None):
            with torch.no_grad():
                metrics = run_epoch(model, testloader
                , criterion, None, [], [], [])
                rs['test'].append(metrics)
            best_metrics, best_f1 = update_best_metrics(rs['test'][-1], best_metrics, best_f1)
            print(format_test_epoch_metrics(epoch, max_epoch, rs['test'][-1], best_metrics))
            if save_path:
                # logger.info('save model@%d to %s' % (epoch, save_path))
                
                torch.save({
                'epoch': epoch,
                'log': {
                    'train': [rs['train'][i] for i in range(len(rs['train']))],
                    'test': [rs['test'][i] for i in range(len(rs['test']))],
                },
                # 'optimizer': optimizer.state_dict(),
                # 'model': model.state_dict(),
                'BestPolicy':policy
            }, str(save_path.joinpath(save_name+'.pth')))   
        if args.mfc and should_refresh_policy(args, epoch, epoch_start):
            refresh_start = time.time()
            optimal_policy, idx_to_group, policy_subset = refresh_mfc_policy(
                model,
                resize_size,
                data_list,
                label_list,
                args,
            )
            refresh_elapsed = time.time() - refresh_start
            logger.info('MFC policy refresh at epoch %d took %.2fs', epoch, refresh_elapsed)
            if policy_subset == []:
                continue
            policy.append(policy_subset)

    # 输出本次训练的最优结果
    if best_metrics is not None:
        print(f"最佳测试结果 (第 {itrs + 1} 次实验):")
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return model, best_metrics
    
def run_python_file(args):
    # command = [python_executable, file_name] + list(args)
    # result = subprocess.run(args, capture_output=True, text=True)
    result = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout:
        print(line, end='')

    result.wait()
    return result.returncode


def copy_args_with_parameter(args, param_name, value):
    variant_args = argparse.Namespace(**vars(args))
    setattr(variant_args, param_name, value)
    variant_args.param_test = ''
    variant_args.param_values = None
    return variant_args


def build_parameter_sensitivity_csv_path(args, param_name):
    return Path('result') / f'{build_save_name(args)}_sensitivity_{param_name}.csv'


def write_parameter_sensitivity_csv(args, param_name, records):
    csv_filename = build_parameter_sensitivity_csv_path(args, param_name)
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['parameter', 'value', 'metric', 'mean', 'std', 'max', 'min']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for metric, values in record['stats'].items():
                writer.writerow({
                    'parameter': param_name,
                    'value': record['value'],
                    'metric': metric,
                    'mean': values['mean'],
                    'std': values['std'],
                    'max': values['max'],
                    'min': values['min'],
                })
    print(f"\n参数敏感性测试结果已保存到 {csv_filename}")
    return csv_filename


def run_trials(args):
    os.makedirs('result', exist_ok=True)

    all_trials_results = []

    for itrs in range(0, args.num_trials):
        model, best_metrics = run_trial(args, itrs)
        if best_metrics is not None:
            all_trials_results.append(best_metrics)

    summarize_trials(args, all_trials_results)
    return all_trials_results


def run_parameter_sensitivity(args):
    os.makedirs('result', exist_ok=True)
    param_name = args.param_test
    values = parse_parameter_test_values(param_name, args.param_values)
    records = []

    for value in values:
        variant_args = copy_args_with_parameter(args, param_name, value)
        print(f"\n{'='*50}")
        print(f"参数敏感性测试: {param_name} = {value}")
        print(f"{'='*50}")
        trial_results = run_trials(variant_args)
        stats = build_stats_rows(trial_results) if trial_results else {}
        records.append({
            'parameter': param_name,
            'value': value,
            'results': trial_results,
            'stats': stats,
        })

    write_parameter_sensitivity_csv(args, param_name, records)
    return records


def run_dataset_magnification_variants(args):
    records = []
    for variant_args in expand_dataset_magnification_args(args):
        label = variant_args.dataset
        if variant_args.magnification is not None:
            label = f'{label}:{variant_args.magnification}'
        print(f"\n{'='*50}")
        print(f"数据集实验: {label}")
        print(f"{'='*50}")
        result = (
            run_parameter_sensitivity(variant_args)
            if variant_args.param_test
            else run_trials(variant_args)
        )
        records.append({
            'dataset': variant_args.dataset,
            'magnification': variant_args.magnification,
            'result': result,
        })
    return records


def main(args):
    expanded_args = expand_dataset_magnification_args(args)
    if len(expanded_args) > 1:
        return run_dataset_magnification_variants(args)
    args = expanded_args[0]
    if args.param_test:
        return run_parameter_sensitivity(args)
    return run_trials(args)


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # patch_sklearn()
    parser = build_parser()
    cli_args = parser.parse_args()
    try:
        main(cli_args)
    except ValueError as exc:
        parser.error(str(exc))
