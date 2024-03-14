import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import albumentations as A
import torchvision.transforms as transforms
from timm.models.layers import to_2tuple

def build_target(label, num_class):
    target = label.clone()
    dice_target = F.one_hot(label.long(), num_class)
    return dice_target.permute(0, 3, 1, 2)

def dice_eff(pred, dice_target, epsilon=1e-6):
    d = 0
    batch_size = pred.shape[0]
    for i in range(batch_size):
        p_i = pred[i].reshape(-1)
        d_i = dice_target[i].reshape(-1)
        inter = torch.dot(p_i, d_i.float())
        set_sum = torch.sum(p_i) + torch.sum(d_i)
        if set_sum == 0:
            set_sum == 2 * inter
        d += (2 * inter + epsilon) / (set_sum + epsilon)
        
    return d / batch_size

def multi_class_dice_eff(pred, dice_target):
    d = 0
    for channel in range(pred.shape[1]):
        dice = dice_eff(pred[:, channel, ...], dice_target[:, channel, ...])
        d += dice        
    return d / pred.shape[1]

def dice_loss(pred, dice_target, multiclass=True):
    fn = multi_class_dice_eff if multiclass else dice_eff
    return 1 - fn(pred, dice_target)

def criterion_single(pred, label, num_class, dice=True):
    l1 = F.cross_entropy(pred, label.long())
    if dice:
        dice_target = build_target(label, num_class)
        l2 = dice_loss(pred, dice_target)
        return l1 + l2
    else:
        return l1

def calculate_metric(pred, label):
    pred[pred>0] = 1
    label[label>0] = 1
    if pred.sum() > 0 and label.sum() > 0:
        dice = metric.binary.dc(pred, label)
        return dice
    elif pred.sum() == 0 and label.sum() > 0:
        return 0
    else:
        return 1e-6
    
def expand_mask(pred, label):
    label = label.float()
    masks = []
    for p in pred:
        size = p.shape[-2:]
        stride = 224 // size[0]
        kernel_size = stride
        padding = 0
        m = F.max_pool2d(label, kernel_size, stride, padding)
        masks.append(m)
    return masks

def criterion(pred, label, num_class): # len(pred)==5, label.shape: B, 1, 224, 224
    masks = expand_mask(pred, label)
    
    
#     print(len(masks))
    l_0 = criterion_single(pred[0], masks[0], num_class=num_class)
    l_1 = criterion_single(pred[1], masks[1], num_class=num_class)
    l_2 = criterion_single(pred[2], masks[2], num_class=num_class)
    l_3 = criterion_single(pred[3], masks[3], num_class=num_class)
    l_4 = criterion_single(pred[4], masks[4], num_class=num_class)
    return (l_0 + l_1 + l_2 + l_3 + l_4) / 5