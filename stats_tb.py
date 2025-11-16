import torch
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from itertools import product, combinations
from pathlib import Path

metrics = ['precision','recall','f1']
datasets = ['breakhis840X','breakhis8100X','breakhis8200X','breakhis8400X','EndoscopicBladder','ChestCTScan','kvasir-dataset']
augs = ['','_randaugment','_trivialaugment']

# all_dirs = Path('./logs/mfc').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize_resize_itrs*')
# all_dirs = Path('./logs').glob(f'{datasets[0]}_resnet18_180epochs_450imsize_randaugment_resize_itrs*')
# all_dirs = Path('./logs').glob(f'{datasets[1]}_resnet18_180epochs_450imsize_resize_itrs*')
# all_dirs = Path('./params_save').glob(f'{datasets[1]}_resnet18_180epochs_450imsize_resize_itrs*')
dataset_id = 0
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
all_dirs = Path(result_root).glob(file_names)
# all_dirs = Path('./params_save').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize{augs[0]}_resize_itrs*')
# all_dirs = Path('./params_save/mfc').glob(f'{datasets[-1]}_resnet18_gray_180epochs_320imsize_resize_online_itrs*')

all_result = {'precision':[],'recall':[],'f1':[],'accuracy':[],'auc':[]}
for f in all_dirs:
    try:
        event_acc = torch.load(str(f))
        values = {'precision':[],'recall':[],'f1':[],'accuracy':[],'auc':[]}
        for key in all_result.keys():
            values[key] = [e[key] for e in event_acc['log']['test']]
        idx = np.argmax(values['f1'])
        for key in all_result.keys():
            all_result[key].append(values[key][idx])
        # event_acc = EventAccumulator(str(f.joinpath('cls','test')))
        # event_acc = EventAccumulator('./logs/breakhis840X_resnet18_180epochs_450imsize_randaugment_resize_itrs10/cls/test')
        # event_acc.Reload()
        # values = {'precision':[],'recall':[],'f1':[],'accuracy':[],'auc':[]}
        # for key in all_result.keys():
        #     scalar_events = event_acc.Scalars(key)
        #     values[key] = [e.value for e in scalar_events]
        # idx = np.argmax(values['f1'])
        # for key in all_result.keys():
        #     all_result[key].append(values[key][idx])
    except:
        print("error")
        continue
    # print(all_result)

print([f'{np.mean(all_result[k]):.4f}' for k in all_result.keys()])
print([f'{np.std(all_result[k]):.4f}' for k in all_result.keys()])