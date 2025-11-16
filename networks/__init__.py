# import torch

# from torch import nn
# from torch.nn import DataParallel
# import torch.backends.cudnn as cudnn
# # from torchvision import models

# from networks.resnet import ResNet
# from networks.shakeshake.shake_resnet import ShakeResNet
# from networks.wideresnet import WideResNet
# from networks.shakeshake.shake_resnext import ShakeResNeXt
# from networks.convnet import SeqConvNet
# from networks.mlp import MLP
# from torchvision.models import vgg16, resnet18, resnet34, resnet50, efficientnet_v2_s
# from common import apply_weightnorm



# example usage get_model(
def get_model(conf, bs, num_class=10, writer=None):
    name = conf['type']
    ad_creators = (None,None)

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=conf.get('dropout',0.0), num_classes=num_class, adaptive_dropouter_creator=ad_creators[0],adaptive_conv_dropouter_creator=ad_creators[1], groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False), virtual_bn=conf.get('virtual_bn', False))
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=conf.get('dropout',0.0), num_classes=num_class, adaptive_dropouter_creator=ad_creators[0],adaptive_conv_dropouter_creator=ad_creators[1], groupnorm=conf.get('groupnorm',False), examplewise_bn=conf.get('examplewise_bn', False), virtual_bn=conf.get('virtual_bn', False))
    elif name == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False))
    elif name == 'miniconvnet':
        model = SeqConvNet(num_class,adaptive_dropout_creator=ad_creators[0],batch_norm=False)
    elif name == 'mlp':
        model = MLP(num_class, (3,32,32), adaptive_dropouter_creator=ad_creators[0])
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)
    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)
    elif name == 'vgg16':
        model = vgg16(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_class)
    elif name == 'resnet18':
        model = resnet18(num_classes=num_class)
    elif name == 'resnet34':    
        model = resnet34(num_classes=num_class)        
    elif name == 'resnet50':
        model = resnet50(num_classes=num_class)
    elif name == 'efficientnet':
        model = efficientnet_v2_s(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    else:
        raise NameError('no model named, %s' % name)

    if conf.get('weight_norm', False):
        print('Using weight norm.')
        apply_weightnorm(model)

    #model = model.cuda()
    #model = DataParallel(model)
    cudnn.benchmark = True
    return model

# def get_model(conf, bs, num_class=10, writer=None):
#     name = conf['type']
#     ad_creators = (None,None)

#     if name == 'resnet18':
#         model = resnet18(weights=None)
#         model.fc = nn.Linear(model.fc.in_features, num_class)
#     elif name == 'vgg16':
#         model = vgg16(weights=None)
#         model.classifier = nn.Linear(model.classifier.in_features, num_class)
#     elif name == 'efficientnet_v2_s':
#         model = efficientnet_v2_s(weights=None)
#         model.classifier = nn.Linear(model.classifier.in_features, num_class)
    
#     #model = model.cuda()
#     #model = DataParallel(model)
#     cudnn.benchmark = True
#     return model

def num_class(dataset):
    return {
        'lymphoma':3,
        'breakhis':8,
        'lc25000':5,
        'rect':2,
        'chestct':4,
        'EndoscopicBladder':4,
        'Corona':7,
        'kvasir-dataset':8,
        'PAD-UFES-20':6
    }[dataset]

# def num_class(dataset):
#     return {
#         'cifar10': 10,
#         'noised_cifar10': 10,
#         'targetnoised_cifar10': 10,
#         'reduced_cifar10': 10,
#         'cifar10.1': 10,
#         'pre_transform_cifar10': 10,
#         'cifar100': 100,
#         'pre_transform_cifar100': 100,
#         'fiftyexample_cifar100': 100,
#         'tenclass_cifar100': 10,
#         'svhn': 10,
#         'svhncore': 10,
#         'reduced_svhn': 10,
#         'imagenet': 1000,
#         'smallwidth_imagenet': 1000,
#         'ohl_pipeline_imagenet': 1000,
#         'reduced_imagenet': 120,
#     }[dataset]
