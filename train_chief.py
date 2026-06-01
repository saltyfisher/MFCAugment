import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
import logging
import sys
import csv
import pickle
from collections import OrderedDict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CHIEF.models.ctran import ctranspath
from data_mm import get_data
from networks import num_class


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CHIEF_Trainer')


class CHIEFFeatureExtractor:
    """CHIEF特征提取器"""
    
    def __init__(self, model_path='./CHIEF/model_weight/CHIEF_CTransPath.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 加载CHIEF模型
        self.model = ctranspath()
        self.model.head = nn.Identity()  # 移除分类头，只提取特征
        
        # 加载预训练权重
        if os.path.exists(model_path):
            td = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(td['model'], strict=True)
            logger.info(f"Loaded CHIEF model from {model_path}")
        else:
            logger.warning(f"Model path {model_path} not found, using random initialization")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 定义图像预处理（CHIEF需要224x224输入）
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.chief_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 确保图像为224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def extract_features(self, images):
        """
        从图像中提取特征
        
        Args:
            images: PIL图像列表或单个PIL图像
            
        Returns:
            特征张量 [batch_size, 768]
        """
        if not isinstance(images, list):
            images = [images]
        
        features_list = []
        with torch.no_grad():
            for img in images:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 使用CHIEF特定的预处理
                img_tensor = self.chief_transform(img).unsqueeze(0).to(self.device)
                
                # 提取特征
                feature = self.model(img_tensor)
                features_list.append(feature.cpu())
        
        # 拼接所有特征
        features = torch.cat(features_list, dim=0)
        return features


class FeatureClassifier(nn.Module):
    """基于CHIEF特征的简单分类器"""
    
    def __init__(self, input_dim=768, num_classes=8):
        super(FeatureClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_data.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            running_loss += loss.item() * batch_data.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_val(model, optimizer, num_classes, args, itrs, extractor, dataroot, save_path=None, log_path=None, save_name=None):
    """
    训练和验证函数，按照train_mfc_mm.py的方式
    
    Args:
        model: 分类器模型
        optimizer: 优化器
        num_classes: 类别数
        args: 参数
        itrs: 当前试验次数
        extractor: CHIEF特征提取器
        dataroot: 数据根目录
        save_path: 模型保存路径
        log_path: 日志路径
        save_name: 保存文件名
        
    Returns:
        model: 训练后的模型
        best_metrics: 最佳测试指标
    """
    
    criterion = nn.CrossEntropyLoss()
    
    max_epoch = args.num_epochs
    epoch_start = 1
    rs = {'train':[],'test':[]}
    best_f1 = 0
    best_metrics = None

    # 按照train_mfc_mm.py的方式加载数据
    logger.info(f"Loading dataset from {dataroot}")
    traintest_dataset, test_dataset, resize_size, train_transform = get_data(
        strategy=args.strategy,
        dataset=args.dataset,
        magnification=args.magnification,
        dataroot=dataroot,
        random_state=args.seed,
        test_split=args.test_split
    )

    # 获取训练集的所有图像和标签（使用loader获取原始PIL图像）
    logger.info("Extracting training features...")
    train_images = [traintest_dataset.dataset.loader(traintest_dataset.dataset.samples[i][0]) for i in traintest_dataset.indices]
    train_labels = [traintest_dataset.dataset.targets[i] for i in traintest_dataset.indices]
    
    # 提取训练集特征
    train_features = extractor.extract_features(train_images)
    train_labels_tensor = torch.tensor(train_labels)
    
    # 获取测试集的所有图像和标签（使用loader获取原始PIL图像）
    logger.info("Extracting test features...")
    test_images = [test_dataset.dataset.loader(test_dataset.dataset.samples[i][0]) for i in test_dataset.indices]
    test_labels = [test_dataset.dataset.targets[i] for i in test_dataset.indices]
    
    # 提取测试集特征
    test_features = extractor.extract_features(test_images)
    test_labels_tensor = torch.tensor(test_labels)
    
    # 创建特征数据集
    class FeatureDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    train_feature_dataset = FeatureDataset(train_features, train_labels_tensor)
    test_feature_dataset = FeatureDataset(test_features, test_labels_tensor)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_feature_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_feature_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 学习率调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    for epoch in range(epoch_start, max_epoch+1):
        model.train()
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, extractor.device)
        rs['train'].append(train_metrics)
        loss = train_metrics['loss']
        accuracy = train_metrics['accuracy']
        print(f'Epoch [{epoch}/{max_epoch}] - Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}')
        
        model.eval()
        
        # 验证
        if epoch % 1 == 0 or epoch == max_epoch:
            with torch.no_grad():
                test_metrics = evaluate(model, test_loader, criterion, extractor.device)
                rs['test'].append(test_metrics)
                print(f'Epoch [{epoch}/{max_epoch}] - Test Loss: {test_metrics["loss"]:.4f}, Test Acc: {test_metrics["accuracy"]:.4f}')
            
            if rs['test'][-1]['f1'] > best_f1:
                best_f1 = rs['test'][-1]['f1']
                best_metrics = rs['test'][-1]
            
            # 保存模型
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'log': {
                        'train': [rs['train'][i] for i in range(len(rs['train']))],
                        'test': [rs['test'][i] for i in range(len(rs['test']))],
                    },
                    'model': model.state_dict(),
                }, save_name + f'_itrs{itrs+1}.pth')
        
        # 更新学习率
        # scheduler.step()

    # 输出本次训练的最优结果
    if best_metrics is not None:
        print(f"最佳测试结果 (第 {itrs + 1} 次实验):")
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return model, best_metrics


def main(args):
    """主函数 - 单次试验"""
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化特征提取器
    extractor = CHIEFFeatureExtractor(
        model_path=args.model_path,
        device=args.device
    )
    
    # 初始化分类器
    num_classes = num_class(args.dataset)
    model = FeatureClassifier(
        input_dim=768,
        num_classes=num_classes,
    ).to(extractor.device)
    
    logger.info(f"Classifier architecture:\n{model}")
    logger.info(f"Number of classes: {num_classes}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练
    model, best_metrics = train_val(
        model, optimizer, num_classes, args, 0, extractor,
        args.data_root, args.save_path, args.log_path, args.save_name
    )
    
    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHIEF Feature-based Classification for BreakHis')
    
    # 数据相关参数
    parser.add_argument('--data_root', type=str, default='/workspace/MedicalImageClassficationData/', 
                       help='Path to dataset root directory')
    parser.add_argument('--strategy', type=str, default='',
                       help='Augmentation strategy (empty, randaugment, trivialaugment, etc.)')
    parser.add_argument('--dataset', type=str, default='breakhis',
                       help='Dataset name (e.g., breakhis)')
    parser.add_argument('--magnification', type=str, default='40',
                       help='Magnification level (e.g., 40, 100, 200, 400)')
    parser.add_argument('--model_path', type=str, 
                       default='./CHIEF/model_weight/CHIEF_CTransPath.pth',
                       help='Path to CHIEF model weights')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # 其他参数
    parser.add_argument('--test_split', type=float, default=0.3,
                       help='Test set split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of independent trials')
    
    args = parser.parse_args()

    # 创建结果保存目录
    stats_path = './result'
    model_params_path = './params_save'
    log_path = './logs'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(stats_path, exist_ok=True)
    os.makedirs(model_params_path, exist_ok=True)
    
    # 构建保存文件名
    save_name = f"chief_{args.dataset}_{args.magnification}"
    if args.strategy:
        save_name += f"_{args.strategy}"
    
    result_filename = save_name + '.txt'
    result_filename = os.path.join(stats_path, result_filename)
    
    # 保存统计结果到CSV文件
    csv_filename = save_name + '.csv'
    csv_filename = os.path.join(stats_path, csv_filename)
    
    args.save_name = save_name
    args.log_path = log_path
    args.save_path = model_params_path
    
    # 存储所有试验的结果
    all_trials_results = []
    
    for itrs in range(0, args.num_trials):
        print(f"\n{'='*50}")
        print(f"开始第 {itrs + 1}/{args.num_trials} 次独立实验")
        print(f"{'='*50}")
        
        # 设置不同的随机种子
        args.seed = 42 + itrs
        
        # 运行单次试验
        best_metrics = main(args)
        
        # 保存本次实验的最佳结果
        if best_metrics is not None:
            all_trials_results.append(best_metrics)
            with open(result_filename, 'a+') as f:
                f.write(str(best_metrics)+"\n")
    
    # 所有轮次结束后计算统计信息
    if all_trials_results:
        print(f"\n{'='*50}")
        print("所有实验结束，统计结果如下:")
        print(f"{'='*50}")
        
        # 获取所有的指标名称
        metric_names = list(all_trials_results[0].keys())
        
        # 计算每个指标的统计信息
        stats = {}
        for metric in metric_names:
            values = [result[metric] for result in all_trials_results]
            stats[metric] = {
                'mean': f'{np.mean(values):.4f}',
                'std': f'{np.std(values):.4f}',
                'max': f'{np.max(values):.4f}',
                'min': f'{np.min(values):.4f}'
            }
            print(f"{metric}:")
            print(f"  均值: {stats[metric]['mean']}")
            print(f"  方差: {stats[metric]['std']}")
            print(f"  最大值: {stats[metric]['max']}")
            print(f"  最小值: {stats[metric]['min']}")
        
        # 保存统计结果到CSV文件
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['metric', 'mean', 'std', 'max', 'min']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in metric_names:
                writer.writerow({
                    'metric': metric,
                    'mean': stats[metric]['mean'],
                    'std': stats[metric]['std'],
                    'max': stats[metric]['max'],
                    'min': stats[metric]['min']
                })
            for i, trial_result in enumerate(all_trials_results):
                acc = trial_result['accuracy']
                prec = trial_result['precision']
                rec = trial_result['recall']
                f1 = trial_result['f1']
                csvfile.write(f"{i+1},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")
        
        print(f"\n统计结果已保存到 {csv_filename}")
