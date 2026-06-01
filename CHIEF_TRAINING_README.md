# CHIEF特征提取与分类训练指南

## 概述

`train_chief.py` 是一个使用CHIEF（CTransPath）模型作为特征提取器，对BreakHis系列数据集进行图像分类的训练脚本。该脚本遵循与ResNet相同的分类器设计模式，但使用预训练的CHIEF模型提取768维特征向量。

## 功能特点

1. **CHIEF特征提取**：使用预训练的CTransPath模型提取高质量的医学图像特征
2. **简单分类器**：基于提取的特征，使用两层全连接网络进行分类
3. **支持多种放大倍数**：兼容BreakHis数据集的所有放大倍数（40X, 100X, 200X, 400X）
4. **完整的评估指标**：提供准确率、精确率、召回率和F1分数
5. **分层数据划分**：确保训练集和测试集的类别分布一致

## 使用方法

### 基本用法

```bash
python train_chief.py --data_root /path/to/BreakHis/dataset
```

### 完整参数示例

```bash
python train_chief.py \
    --data_root /workspace/data/BreakHis/40X \
    --model_path ./CHIEF/model_weight/CHIEF_CTransPath.pth \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.5 \
    --test_split 0.3 \
    --seed 42 \
    --device cuda \
    --num_workers 4 \
    --save_path chief_breakhis_40X_best.pth
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | (必需) | BreakHis数据集根目录路径 |
| `--model_path` | `./CHIEF/model_weight/CHIEF_CTransPath.pth` | CHIEF模型权重文件路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--weight_decay` | 0.0001 | 权重衰减系数 |
| `--dropout` | 0.5 | Dropout比率 |
| `--test_split` | 0.3 | 测试集比例 |
| `--seed` | 42 | 随机种子 |
| `--device` | cuda | 计算设备（cuda或cpu） |
| `--num_workers` | 4 | 数据加载线程数 |
| `--save_path` | chief_model_best.pth | 最佳模型保存路径 |

## 数据集结构

BreakHis数据集应按以下结构组织：

```
BreakHis/
├── 40X/
│   ├── class_name_1/
│   │   ├── image1.png
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_name_2/
│   │   └── ...
│   └── ...
├── 100X/
├── 200X/
└── 400X/
```

支持的图像格式：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.ppm`, `.pgm`（不区分大小写）

## 模型架构

### 特征提取器（CHIEF/CTransPath）
- 输入：224x224 RGB图像
- 输出：768维特征向量
- 预处理：Resize(224) → ToTensor → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 分类器
```
Linear(768 → 256) → ReLU → Dropout → Linear(256 → num_classes)
```

## 输出指标

每个epoch会输出以下指标：
- **Loss**: 交叉熵损失
- **Accuracy**: 分类准确率
- **Precision**: 加权精确率
- **Recall**: 加权召回率
- **F1**: 加权F1分数

## 注意事项

1. **显存需求**：特征提取阶段需要较大显存，建议在GPU上运行
2. **模型权重**：确保CHIEF模型权重文件存在于指定路径
3. **数据预处理**：所有图像会自动转换为RGB模式
4. **类别数量**：自动从数据集文件夹结构中推断类别数量

## 与ResNet对比

| 特性 | CHIEF | ResNet |
|------|-------|--------|
| 特征维度 | 768 | 512/2048（取决于版本） |
| 预训练 | CTransPath（病理图像） | ImageNet |
| 输入尺寸 | 224x224 | 可变（通常224x224或更大） |
| 适用场景 | 医学病理图像 | 通用图像分类 |

## 示例：不同放大倍数的训练

```bash
# 40X放大倍数
python train_chief.py --data_root /workspace/data/BreakHis/40X --save_path chief_40X.pth

# 100X放大倍数
python train_chief.py --data_root /workspace/data/BreakHis/100X --save_path chief_100X.pth

# 200X放大倍数
python train_chief.py --data_root /workspace/data/BreakHis/200X --save_path chief_200X.pth

# 400X放大倍数
python train_chief.py --data_root /workspace/data/BreakHis/400X --save_path chief_400X.pth
```

## 故障排除

1. **找不到模型权重**：检查`--model_path`参数是否正确
2. **显存不足**：减小`--batch_size`或使用CPU模式
3. **数据加载错误**：确认数据集结构和图像格式正确
4. **类别数量不匹配**：检查数据集文件夹命名是否正确
