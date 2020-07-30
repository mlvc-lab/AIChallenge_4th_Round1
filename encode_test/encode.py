import os

import numpy as np

import torch


def load_to_dict(filename):
    model = torch.load(filename, map_location='cpu')

    model_dict = dict()
    for key in model.keys():
        model_dict[key] = model[key].numpy()
    return model_dict


def convert_weight_int8(model_dict):
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'quantizer.nbits') in model_dict.keys():
            # load
            nbits = model_dict[key.replace('weight', 'quantizer.nbits')][0]
            q_n = -1. * 2. ** (nbits - 1)
            q_p = 2. ** (nbits - 1) - 1

            step = model_dict[key.replace('weight', 'quantizer.step')][0]

            # quantize
            model_dict[key] = np.round(np.clip(model_dict[key] / step, q_n, q_p)).astype(np.int8)
    return model_dict


def convert_int8_uint8(model_dict):
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'quantizer.nbits') in model_dict.keys():
            # convert dtype
            model_dict[key] = model_dict[key].view(np.uint8) + 128
    return model_dict

if __name__ == '__main__':
    # load checkpoint and covert tensor to numpy array
    model_dict = load_to_dict('quant0.pth')
    
    # quantize all weights to int8
    model_dict = convert_weight_int8(model_dict)

    # convert int8 to uint8
    model_dict = convert_int8_uint8(model_dict)

    # save to npz
    np.savez('noncompressed.npz', **model_dict)
    np.savez_compressed('compressed.npz', **model_dict)