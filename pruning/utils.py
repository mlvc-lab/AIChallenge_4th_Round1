import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time, os, math


def get_weight_threshold(model, rate):
    state = model.state_dict()

    importance_all = None
    total_weights = 0
    for name, item in model.named_parameters():
        if 'mask' in name:
            key = name.replace('mask', 'weight')
            assert key in state.keys()

            weights = state[key].data.view(-1).cpu()
            #importance = weights.pow(2).numpy()
            importance = weights.abs().numpy()

            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

        if 'weight' in name:
            total_weights += item.numel()

    threshold = np.sort(importance_all)[int(total_weights * rate / 100)]
    return threshold


def weight_prune(model, threshold):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'weight' in name:
            key = name.replace('weight', 'mask')
            if key in state.keys():
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
    total_weights = 0

    for name, item in model.module.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if 'weight' in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity

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