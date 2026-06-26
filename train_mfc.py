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
    parser.add_argument('--dataset', type=str, default='chestct', help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None, choices=['40', '100', '200', '400', None],
                        help='BreakHis数据集的放大倍数，None表示使用所有倍数')
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
    parser.add_argument('--bayes_max_eval', type=int, default=200, help='贝叶斯优化最大迭代次数')
    parser.add_argument('--bayes_topk', type=int, default=10, help='贝叶斯优化返回的策略数')
    parser.add_argument('--bayes_rep', type=int, default=2, help='贝叶斯优化重复次数')
    parser.add_argument('--mfc_eval_sample_ratio', type=sample_ratio, default=0.2,
                        help='Bayes搜索阶段每个子集使用的代表样本比例，取值范围为(0, 1)')
    parser.add_argument('--group', action='store_true', help='每个数据子集是否单独适配增广策略')
    parser.add_argument('--diff_c', action='store_true', help='每个数据子集的中心点是否不同')
    parser.add_argument('--l', type=int, default=1, help='目标函数权重')
    parser.add_argument('--mag_bin', type=int, default=31, help='变换操作强度离散个数')
    parser.add_argument('--prob_bin', type=int, default=10, help='变换概率离散个数')
    parser.add_argument('--testing', action='store_true')
    return parser


def get_cluster_count(dataset):
    if 'lym' in dataset:
        return 2
    if 'breakhis' in dataset:
        return 8
    if 'chestct' in dataset:
        return 4
    return 4


def should_refresh_policy(args, epoch, epoch_start):
    if not args.online:
        return epoch == epoch_start
    if args.testing:
        return epoch == epoch_start
    return epoch % 40 == 0


def build_idx_to_group(groups):
    idx_to_group = {}
    for group_idx, group_members in enumerate(groups):
        for idx in group_members:
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
        build_idx_to_group(assignment_groups),
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
    args.device = device
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
    for batch in loader:
        if groups == []:
            data, label = batch[:2]
        else:
            augmented_batch = []
            for data, label, data_idx in zip(*batch):
                if args.group:
                    p_idx = groups.get(int(data_idx), 0)
                else:
                    p_idx = np.random.randint(0, len(policy))
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
            optimal_policy, idx_to_group, policy_subset = refresh_mfc_policy(
                model,
                resize_size,
                data_list,
                label_list,
                args,
            )
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

def main(args):
    os.makedirs('result', exist_ok=True)

    all_trials_results = []

    for itrs in range(0, args.num_trials):
        model, best_metrics = run_trial(args, itrs)
        if best_metrics is not None:
            all_trials_results.append(best_metrics)

    summarize_trials(args, all_trials_results)
    return all_trials_results


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # patch_sklearn()
    main(build_parser().parse_args())
