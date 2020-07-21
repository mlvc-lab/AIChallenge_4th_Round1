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



def weight_prune(fullmodel, maskmodel, threshold):
    for (fname, fitem), (mname, mitem) in zip(fullmodel.module.named_parameters(), maskmodel.module.named_parameters()):
        if 'conv' in fname and 'mask' not in fname:
            #mask_data = torch.gt(fitem.data.pow(2), threshold).float()
            mask_data = torch.gt(fitem.data.abs(), threshold).float()
        if 'mask' in mname:
            mitem.data = mask_data


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

def filter_prune(maskmodel, importance, rate):
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

def number_of_zeros(model):
    mask_nonzeros = 0
    nonzeros = 0
    grad_nonzeros = 0
    mask_length = 0
    length = 0
    for name, item in model.module.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.size(0) * item.size(1) * item.size(2) * item.size(3)
        elif len(item.size()) == 4:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()
            nonzeros += np.count_nonzero(np_flatten)

            grad_flatten = item.grad.data.view(-1)
            np_grad_flatten = grad_flatten.cpu().numpy()
            grad_nonzeros += np.count_nonzero(np_grad_flatten)

            length += item.size(0) * item.size(1) * item.size(2) * item.size(3)

    print("Length\t\t|| FullNet : {},\tMask : {},\tGrad : {} ||".format(length, mask_length, length))
    print("Nonzeros\t|| FullNet : {},\tMask : {},\tGrad : {} ||".format(nonzeros, mask_nonzeros, grad_nonzeros))
    print("NPercent\t|| FullNet : {:.2f}%,\tMask : {:.2f}%,\tGrad : {:.2f}% ||".format(100*nonzeros/length, 100*mask_nonzeros/mask_length, 100*grad_nonzeros/length))
    
    return (mask_nonzeros / mask_length) * 100


def weightcopy(fullmodel, maskmodel):
    for (fname, fitem), (mname, mitem) in zip(fullmodel.module.named_parameters(), maskmodel.module.named_parameters()):
        if 'mask' not in mname:
            mitem.data = fitem.data.clone()

def gradcopy(fullmodel, maskmodel):
    for (fname, fitem), (mname, mitem) in zip(fullmodel.module.named_parameters(), maskmodel.module.named_parameters()):
        if 'mask' not in fname:
            fitem.grad = mitem.grad.clone()