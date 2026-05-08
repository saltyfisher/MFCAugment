import pandas as pd
import os
import shutil
from pathlib import Path

def process_padufes_dataset():
    """
    根据/workspace/MedicalImageClassficationData/PAD-UFES-20中的metadata构造PADUFES的数据读取和保存代码，
    根据diagnostic列中的内容为样本赋予标签。然后将所有样本保存到同一目录下，根据标签创建不同的子目录。
    """
    # 数据集路径
    base_path = '/workspace/MedicalImageClassficationData/PAD-UFES-20'
    metadata_path = os.path.join(base_path, 'metadata.csv')
    images_path = os.path.join(base_path, 'images')
    
    # 输出路径
    output_path = os.path.join(base_path, 'organized_dataset')
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 读取元数据
    metadata = pd.read_csv(metadata_path)
    
    # 根据diagnostic列中的内容为样本赋予标签
    # BCC: 基底细胞癌, SCC: 鳞状细胞癌, ACK: 日光性角化病, 
    # NEV: 色素痣, MEL: 黑色素瘤, SEK: 脂溢性角化病
    diagnostic_mapping = {
        'BCC': 'BCC',  # 基底细胞癌
        'SCC': 'SCC',  # 鳞状细胞癌
        'ACK': 'ACK',  # 日光性角化病
        'NEV': 'NEV',  # 色素痣
        'MEL': 'MEL',  # 黑色素瘤
        'SEK': 'SEK'   # 脂溢性角化病
    }
    
    # 为每个类别创建目录
    for class_name in diagnostic_mapping.values():
        class_dir = os.path.join(output_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 处理每个图像
    processed_count = 0
    class_counts = {class_name: 0 for class_name in diagnostic_mapping.values()}
    
    for _, row in metadata.iterrows():
        img_id = row['img_id']
        diagnostic = row['diagnostic']
        
        # 检查诊断类型是否在映射中
        if diagnostic not in diagnostic_mapping:
            print(f"Warning: Unknown diagnostic type '{diagnostic}' for image {img_id}")
            continue
            
        # 获取目标类别名称
        class_name = diagnostic_mapping[diagnostic]
        
        # 源文件路径
        src_file = os.path.join(images_path, img_id)
        
        # 检查源文件是否存在
        if not os.path.exists(src_file):
            print(f"Warning: Image file not found: {src_file}")
            continue
            
        # 目标文件路径
        dst_file = os.path.join(output_path, class_name, img_id)
        
        # 复制文件
        try:
            shutil.copy2(src_file, dst_file)
            processed_count += 1
            class_counts[class_name] += 1
            
            # 每处理100个文件打印一次进度
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error copying {src_file} to {dst_file}: {e}")
    
    print(f"Processing complete. Total processed images: {processed_count}")
    
    # 打印数据集统计信息
    print("\nDataset Statistics:")
    print("===================")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    return output_path

if __name__ == "__main__":
    output_dir = process_padufes_dataset()
    print(f"\nDataset organized in: {output_dir}")