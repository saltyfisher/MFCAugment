import yaml
import itertools

dataset_name = ['breakhis240X','breakhis2100X','breakhis2200X','breakhis2400X',
                'breakhis840X','breakhis8100X','breakhis8200X','breakhis8400X',
                'ChestCTScan']
dataset_name = ['ChestCTScan']
model_names = ['resnet18']
epoch_list = [180]
img_size_list = [320]
augment_type = ['none','randaugment_raw','trivialaugment_raw','randaugment','trivialaugment']
# augment_type = ['none']

settings = itertools.product(dataset_name,model_names,epoch_list,img_size_list,augment_type)

for setting in settings:
    with open('./networks/confs/template.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['dataset'] = setting[0]
    config['epoch'] = setting[2]
    config['model']['type'] = setting[1]
    config['img_size'] = [setting[3],setting[3]]
    config['aug'] = setting[4] #randaugment,trivialaugment

    dataset = config['dataset']
    model = config['model']['type']
    epoch = config['epoch']
    img_size = config['img_size'][0]
    aug = '' if config['aug']=='none' else '_'+config['aug']

    with open(f'./networks/confs/{dataset}_{model}_{epoch}epochs_{img_size}imsize{aug}.yaml', 'w') as file:
        config = yaml.safe_dump(config, file)

