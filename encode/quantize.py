import os

import numpy as np

import torch


def load_to_dict(filename):
    return torch.load(filename)


def save_to_pth(model_dict, filename):
    torch.save(model_dict, filename)


def convert_weight_int8(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'quantizer.nbits') in model_dict.keys():
            # load
            nbits = model_dict[key.replace('weight', 'quantizer.nbits')][0]
            q_n = -1. * 2. ** (nbits - 1)
            q_p = 2. ** (nbits - 1) - 1
            
            step = model_dict[key.replace('weight', 'quantizer.step')][0]

            # quantize
            new_dict[key] = (model_dict[key] / step).clamp(q_n, q_p).round().type(torch.int8)
            new_dict[key.replace('weight', 'step')] = model_dict[key.replace('weight', 'quantizer.step')]

        elif 'num_batches_tracked' in key or 'quantizer.do_init' in key or 'quantizer.nbits' in key or 'quantizer.step' in key or 'running_mean' in key or 'running_var' in key:
            continue

        elif 'weight' in key and key.replace('weight', 'running_mean') in model_dict.keys():
            weight = model_dict[key]
            bias = model_dict[key.replace('weight', 'bias')]
            running_mean = model_dict[key.replace('weight', 'running_mean')]
            running_var = model_dict[key.replace('weight', 'running_var')]

            eps = 1e-05
            new_weight = weight / (running_var + eps).sqrt()
            new_bias = bias - weight * running_mean / (running_var + eps).sqrt()
            
            new_dict[key] = new_weight
            new_dict[key.replace('weight', 'bias')] = new_bias

        else:
            if key not in new_dict.keys():
                new_dict[key] = model_dict[key]
        
    return new_dict


if __name__ == '__main__':
    filename = 'cutmix_0.5_upgrade_tuned.pth'
    # load checkpoint and covert tensor to numpy array
    model_dict = load_to_dict(filename)
    
    # quantize all weights to int8
    quant_dict = convert_weight_int8(model_dict)

    save_to_pth(quant_dict, filename.split('.pth')[0] + '_cnvt.pth')

    for key in quant_dict.keys(): print(key)