import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from pathlib import Path
import pandas as pd
import seaborn as sns
from collections import Counter

metrics = ['precision','recall','f1']
datasets = ['breakhis840X','breakhis8100X','breakhis8200X','breakhis8400X','EndoscopicBladder','ChestCTScan','kvasir-dataset']
augs = ['','_randaugment','_trivialaugment']

# all_dirs = Path('./logs/mfc').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize_resize_itrs*')
# all_dirs = Path('./logs').glob(f'{datasets[0]}_resnet18_180epochs_450imsize_randaugment_resize_itrs*')
# all_dirs = Path('./logs').glob(f'{datasets[1]}_resnet18_180epochs_450imsize_resize_itrs*')
# all_dirs = Path('./params_save').glob(f'{datasets[1]}_resnet18_180epochs_450imsize_resize_itrs*')
dataset_id = 1
aug_id = 0
mfc = True
file_names = f'{datasets[dataset_id]}_resnet18'
if 'ChestCTScan' in file_names:
    file_names += '_gray_180epochs_320imsize'
else:
    file_names += '_180epochs_450imsize'
if mfc:
    result_root = './params_save/mfc'
    file_names += '_resize_online_itrs*'
else:
    result_root = './params_save'
    file_names += f'{augs[aug_id]}_resize_itrs*'
all_dirs = list(Path(result_root).glob(file_names))
# all_dirs = Path('./params_save').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize{augs[0]}_resize_itrs*')
# all_dirs = Path('./params_save/mfc').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize_resize_online_itrs*')
all_augmentations = [
    'Identity',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'Rotate',
    'Brightness',
    'Color',
    'Contrast',
    'Sharpness',
    'Posterize',
    'Solarize',
    'AutoContrast',
    'Equalize',
]
for s in range(4):
    augmentations = []
    for f in all_dirs:
        result = torch.load(str(f))
        # print(f"{f} s: {len(result['BestPolicy'])}")
        try:
            P = result['BestPolicy'][s][0]
        except:
            continue
        for i, p in enumerate(P['op_index']):
            for j, aug_id in enumerate(p):
                aug_id = all_augmentations[int(aug_id)]
                mag_id = int(P['magnitude_index'][i].squeeze()[j])
                augmentations.append((aug_id, mag_id))
                # if augmentations[all_augmentations[aug_id]] is None:
                #     augmentations[all_augmentations[aug_id]] = [P[0]['magnitude_index'][i].squeeze()[j]]
                # else:
                #     augmentations[all_augmentations[aug_id]].append(P[0]['magnitude_index'][i].squeeze()[j])
            pass
    counts_dict = Counter(augmentations)
    data_for_df = []
    for (aug_type, strength), count in counts_dict.items():
        data_for_df.append([aug_type, strength, count])
    df_counts = pd.DataFrame(data_for_df, columns=['Augmentation', 'Strength', 'Count'])
    df_counts['Augmentation'] = pd.Categorical(df_counts['Augmentation'], categories=all_augmentations, ordered=True)
    df_counts['Strength'] = pd.Categorical(df_counts['Strength'], categories=np.arange(31), ordered=True)
    heatmap_data = df_counts.pivot_table(index='Strength', columns='Augmentation', values='Count', fill_value=0, dropna=False)
    # heatmap_data.fillna(0, inplace=True)
    # heatmap_data = heatmap_data.reindex(columns=all_augmentations)

    # 反转Y轴，让强度1在底部
    heatmap_data = heatmap_data.iloc[::-1]

    print("用于绘图的最终矩阵数据:")
    print(heatmap_data)
    print("-" * 30)

    # --- 5. 绘制热力图 (与之前的方法完全相同) ---
    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(heatmap_data, 
                    annot=True, 
                    fmt='g', 
                    cmap='viridis', # 换一个颜色主题试试
                    linewidths=.5,
                    cbar_kws={'label': 'Selection Count (Frequency)'})
    ax.set_xlabel('Augmentation Type', fontsize=15)
    ax.set_ylabel('Strength Level', fontsize=15)
    plt.title(f'{datasets[dataset_id]}-Update {s}')
    plt.xticks(np.arange(len(all_augmentations))+0.5, np.asarray(heatmap_data.columns), rotation=45, ha='right', fontsize=14)
    plt.yticks(np.arange(31)+0.5, np.asarray(heatmap_data.index)+1, rotation=0,fontsize=14)
    plt.tight_layout()
    # --- 步骤 6: 显示或保存图片 ---
    # plt.savefig('augmentation_policy.png', dpi=300) # 如果需要保存图片，取消本行注释
    plt.savefig(f'./vis/{datasets[dataset_id]}_step{s}_augmentation.png', format='png', dpi=600, bbox_inches='tight')
# break