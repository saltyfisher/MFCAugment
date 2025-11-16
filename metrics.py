import copy
import torch
import torchmetrics

from collections import defaultdict
from theconf import Config as C
from torch import nn
from networks import num_class


def accuracy_score(num_classes, output, target):
    """Computes the precision@k for the specified values of k"""
    metric = torchmetrics.Accuracy(task='multiclass',average='macro',num_classes=num_classes).to(output.device)
    res = metric(output,target)
    return res

def precision_score(num_classes, output, target):
    """Computes the precision@k for the specified values of k"""
    metric = torchmetrics.Precision(task='multiclass',average='macro',num_classes=num_classes).to(output.device)
    res = metric(output,target)
    return res

def recall_score(num_classes, output, target):
    """Computes the precision@k for the specified values of k"""
    metric = torchmetrics.Recall(task='multiclass',average='macro',num_classes=num_classes).to(output.device)
    res = metric(output,target)
    return res

def f1_score(num_classes, output, target):
    metric = torchmetrics.F1Score(task='multiclass',average='macro',num_classes=num_classes).to(output.device)
    res = metric(output,target)
    return res

def auc_score(num_classes, output, target, topk=(1,)):
    metric = torchmetrics.AUROC(task='multiclass',average='macro',num_classes=num_classes).to(output.device)
    res = metric(output,target)
    return res

def cross_entropy_smooth(input, target, size_average=True, label_smoothing=0.1):
    y = torch.eye(10).cuda()
    lb_oh = y[target]

    target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __contains__(self, item):
        return self.metrics.__contains__(item)

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            newone[key] = value / self.metrics['cnt']
        return newone

    def divide(self, divisor, **special_divisors):
        newone = Accumulator()
        for key, value in self.items():
            if key in special_divisors:
                newone[key] = value/special_divisors[key]
            else:
                newone[key] = value/divisor
        return newone


class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass
