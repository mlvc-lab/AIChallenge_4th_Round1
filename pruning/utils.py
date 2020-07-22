import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time, os, math


def get_weight_threshold(model, rate):
    importance_all = None
    for name, item in model.module.named_parameters():
        if 'conv' in name and 'mask' not in name:
            weights = item.data.view(-1).cpu()
            #importance = weights.pow(2).numpy()
            importance = weights.abs().numpy()

            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance_all)[int(len(importance_all) * rate / 100)]
    return threshold


def weight_prune(model, threshold):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            key = name.replace('weight', 'mask')
            state[key].data.copy_(torch.gt(item.data.abs(), threshold).float())


def get_filter_importance(model):
    importance_all = None
    for name, item in model.module.named_parameters():
        if 'downsample' in name:
            continue
        if 'conv1' in name and 'layer' in name:
            filters = item.data.view(item.size(0), -1).cpu()
            weight_len = filters.size(1)
            importance = filters.pow(2).sum(dim=1).numpy() / weight_len
        
            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)
    
    return importance_all


def filter_prune(model, importance, rate):
    threshold = np.sort(importance)[int(len(importance) * rate / 100)]
    #threshold = np.percentile(importance, rate)
    filter_mask = np.greater(importance, threshold)

    idx = 0
    for name, item in maskmodel.module.named_parameters():
        if 'mask1' in name and 'layer' in name:
            if 'downsample' in name:
                continue
                '''for i in range(item.size(0)):
                    item.data[i,:,:,:] = 1 if filter_mask[pre_idx] else 0
                    pre_idx+=1
                '''
            else:  
                pre_idx = idx
                for i in range(item.size(0)):
                    item.data[i,:,:,:] = 1 if filter_mask[idx] else 0
                    idx+=1


def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0

    for name, item in model.module.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.size(0) * item.size(1) * item.size(2) * item.size(3)

    num_total = mask_length
    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / num_total) * 100
    return num_total, num_zero, sparsity

'''
def weightcopy(fullmodel, maskmodel):
    for (fname, fitem), (mname, mitem) in zip(fullmodel.module.named_parameters(), maskmodel.module.named_parameters()):
        if 'mask' not in mname:
            mitem.data = fitem.data.clone()
'''
'''
def gradcopy(fullmodel, maskmodel):
    for (fname, fitem), (mname, mitem) in zip(fullmodel.module.named_parameters(), maskmodel.module.named_parameters()):
        if 'mask' not in fname:
            fitem.grad = mitem.grad.clone()
'''