import torch
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy

def top1_accuracy(preds, targets, num_classes, average):
    acc = multiclass_accuracy(preds, targets, num_classes=num_classes, average=average)
    return acc.item()

def f1_score(preds, targets, num_classes, average):
    f1 = multiclass_f1_score(preds, targets, num_classes=num_classes, average=average)
    return f1.item()