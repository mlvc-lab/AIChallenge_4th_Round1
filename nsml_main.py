import pathlib
import os
import argparse

import PIL
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import models
from utils import *

import quantization

import nsml
from nsml_utils import *

'''
1. How to run
    nsml run -d tr-4 -e nsml_main.py -a "-a resnet --layers 18 --load best.pth"

2. How to list checkpoints saved
    nsml model ls {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER}

3. submit
    nsml submit [-t] {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER} {CHECKPOINT_NAME}
'''

def main():
    # Argument Settings
    parser = argparse.ArgumentParser(description='Image Tagging Classification from Naver Shopping Reviews')
    # for model architecture
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        help='name of the model architecture (default: resnet)')
    parser.add_argument('--layers', default=56, type=int, metavar='N',
                        help='number of layers in ResNet (default: 56)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--depth-mult', default=1.0, type=float, metavar='DM',
                         help='depth multiplier network (rexnet)')
    parser.add_argument('--model-mult', default=0, type=int,
                        help="e.g. efficient type (0 : b0, 1 : b1, 2 : b2 ...)")
    # for gpu configuration
    parser.add_argument('-C', '--cuda', default=True, dest='cuda', action='store_true',
                        help='use cuda?')
    parser.add_argument('-g', '--gpuids', metavar='GPU', default=[0],
                        type=int, nargs='+',
                        help='GPU IDs for using (default: 0)')
    # for load and save
    parser.add_argument('--load', default=None, type=str, metavar='FILE.pth',
                        help='name of checkpoint for testing model (default: None)')
    # for nsml submission
    parser.add_argument('--run-type', default='evaluate', type=str, metavar='TYPE',
                        help='type of run the main function e.g. train or evaluate (default: train)')
    parser.add_argument('--mode', default='train', help='Mode')
    parser.add_argument('--pause', default=0, type=int)
    # for quantization
    parser.add_argument('-Q', '--quantize', dest='quantize', action='store_true',
                        help='If this is set, the model layers are changed to quantized layers.')
    parser.add_argument('--quantizer', default='lsq', type=str,
                        help='method of quantization to apply (default: lsq)')
    parser.add_argument('--quant-bitw', default=8, type=int, metavar='N',
                        help='number of bits for weights (default: 8)')
    parser.add_argument('--quant-bita', default=32, type=int, metavar='N',
                        help='number of bits for activations (default: 32)')
    parser.add_argument('--quant-cfg', default='base', type=str,
                        help='name of quantization configuration for each type of layers (default: base)')
    args = parser.parse_args()

    # set a model
    if args.quantize:
        model, image_size = quantization.models.__dict__[args.arch](data='things', num_layers=args.layers,
                                                                    width_mult=args.width_mult,
                                                                    depth_mult=args.depth_mult,
                                                                    model_mult=args.model_mult,
                                                                    qnn=quantizer.qnn if args.run_type == 'train' else quantizer.iqnn,
                                                                    bitw=args.quant_bitw, bita=args.quant_bita,
                                                                    qcfg=quantizer.__dict__[args.quant_cfg])
    else:
        model, image_size = models.__dict__[args.arch](data='things', num_layers=args.layers,
                                                       width_mult=args.width_mult,
                                                       depth_mult=args.depth_mult,
                                                       model_mult=args.model_mult)
    
    # set multi-gpu
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
    
    # load a checkpoint
    ckpt_file = pathlib.Path('checkpoint_nsml') / args.load
    strict = False if args.quantize else True
    if args.quantize:
        quantization.load_quant_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
    else:
        load_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)

    # bind the loaded model
    if args.quantize:
        bind_quant_model(model)
    else:
        bind_model(model)
    nsml.save('submission')

    if args.pause:
        nsml.paused(scope=locals())


if __name__ == '__main__':
    main()
