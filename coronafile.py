import pandas as pd
import os
import shutil
from pathlib import Path
from PIL import Image

def process_corona_dataset():
    """
    根据Label、Label_1_Virus_category和Label_2_Virus_category列中的内容为样本赋予标签，
    并根据Dataset_type将样本分类为训练和测试集。
    """
    # 数据集路径
    base_path = '/workspace/MedicalImageClassficationData/CoronaHack-Chest X-Ray-Dataset'
    metadata_path = os.path.join(base_path, 'Chest_xray_Corona_Metadata.csv')
    images_path = os.path.join(base_path, 'Coronahack-Chest-XRay-Dataset')
    
    # 输出路径
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    # 创建输出目录
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # 读取元数据
    metadata = pd.read_csv(metadata_path)
    
    # 定义标签映射
    # 根据代码库中的实现，标签分配如下：
    # 0: Normal
    # 1: Pneumonia + Stress-Smoking & ARDS
    # 2: Pneumonia + Virus (not COVID-19 or SARS)
    # 3: Pneumonia + COVID-19
    # 4: Pneumonia + SARS
    # 5: Pneumonia + bacteria (not Streptococcus)
    # 6: Pneumonia + Streptococcus
    
    def get_label(row):
        if row['Label'] == 'Normal':
            return 0
        elif row['Label'] == 'Pnemonia':
            # 处理缺失值情况
            label_1 = row['Label_1_Virus_category'] if pd.notna(row['Label_1_Virus_category']) else ''
            label_2 = row['Label_2_Virus_category'] if pd.notna(row['Label_2_Virus_category']) else ''
            
            if label_1 == 'Stress-Smoking' and label_2 == 'ARDS':
                return 1
            elif label_1 == 'Virus':
                if label_2 == 'COVID-19':
                    return 3
                elif label_2 == 'SARS':
                    return 4
                else:
                    return 2
            elif label_1 == 'bacteria':
                if label_2 == 'Streptococcus':
                    return 6
                else:
                    return 5
            else:
                # 默认病毒性肺炎
                return 2
        return -1  # 未知标签
    
    # 为每行数据分配标签
    metadata['assigned_label'] = metadata.apply(get_label, axis=1)
    
    # 标签名称映射
    label_names = {
        0: 'Normal',
        1: 'Pneumonia_StressSmoking_ARDS',
        2: 'Pneumonia_Virus_Other',
        3: 'Pneumonia_Virus_COVID19',
        4: 'Pneumonia_Virus_SARS',
        5: 'Pneumonia_Bacteria_Other',
        6: 'Pneumonia_Bacteria_Streptococcus'
    }
    
    # 创建标签目录
    for label_id, label_name in label_names.items():
        os.makedirs(os.path.join(train_path, label_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, label_name), exist_ok=True)
    
    # 处理每个图像
    processed_count = 0
    for index, row in metadata.iterrows():
        image_name = row['X_ray_image_name']
        dataset_type = row['Dataset_type']
        label = row['assigned_label']
        
        # 跳过无效标签
        if label == -1:
            continue
            
        # 确定源文件路径（支持jpeg和png）
        src_path_jpeg = os.path.join(images_path, 'train', image_name)
        src_path_png = os.path.join(images_path, 'train', image_name.replace('.jpeg', '.png'))
        src_path_jpeg_test = os.path.join(images_path, 'test', image_name)
        src_path_png_test = os.path.join(images_path, 'test', image_name.replace('.jpeg', '.png'))
        
        src_path = None
        if os.path.exists(src_path_jpeg):
            src_path = src_path_jpeg
        elif os.path.exists(src_path_png):
            src_path = src_path_png
        elif os.path.exists(src_path_jpeg_test):
            src_path = src_path_jpeg_test
        elif os.path.exists(src_path_png_test):
            src_path = src_path_png_test
            
        # 如果找不到源文件，跳过
        if src_path is None or not os.path.exists(src_path):
            print(f"Warning: Image file not found: {image_name}")
            continue
            
        # 确定目标路径
        label_name = label_names[label]
        if dataset_type == 'TRAIN':
            dst_path = os.path.join(train_path, label_name, image_name)
        else:  # TEST or VALIDATION
            dst_path = os.path.join(test_path, label_name, image_name)
            
        # 复制文件
        try:
            shutil.copy2(src_path, dst_path)
            processed_count += 1
            
            # 每处理100个文件打印一次进度
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
    
    print(f"Processing complete. Total processed images: {processed_count}")
    
    # 打印数据集统计信息
    print("\nDataset Statistics:")
    print("===================")
    train_counts = {}
    test_counts = {}
    
    for label_id, label_name in label_names.items():
        train_dir = os.path.join(train_path, label_name)
        test_dir = os.path.join(test_path, label_name)
        
        train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpeg', '.png', '.jpg'))]) if os.path.exists(train_dir) else 0
        test_count = len([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpeg', '.png', '.jpg'))]) if os.path.exists(test_dir) else 0
        
        train_counts[label_name] = train_count
        test_counts[label_name] = test_count
        
        print(f"{label_name}:")
        print(f"  Train: {train_count}")
        print(f"  Test:  {test_count}")
        print(f"  Total: {train_count + test_count}")
        print()

if __name__ == "__main__":
    process_corona_dataset()
