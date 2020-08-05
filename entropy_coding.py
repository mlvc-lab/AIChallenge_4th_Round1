import time
import pathlib
import argparse
from os.path import isfile

import deepCABAC

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import models
import config
from utils import *
from data import DataLoader
from data import valid_datasets as dataset_names

# for sacred logging
from sacred import Experiment
from sacred.observers import MongoObserver

# sacred experiment
ex = Experiment('AI-Challenge_Entropy-Coding')
ex.observers.append(MongoObserver.create(url=config.MONGO_URI,
                                         db_name=config.MONGO_DB))


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def configuration():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='Entropy Coding')
    parser.add_argument('dataset', metavar='DATA', default='cifar10',
                        choices=dataset_names,
                        help='dataset: ' +
                            ' | '.join(dataset_names) +
                            ' (default: cifar10)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet)')
    parser.add_argument('--layers', default=56, type=int, metavar='N',
                        help='number of layers in ResNet (default: 56)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                            'uniformly at each layer (default: 1.0)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='path of checkpoint for testing model (default: none)')
    # for evaluation
    parser.add_argument('-E', '--evaluate', dest='evaluate', action='store_true',
                        help='test model?')
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='use cuda?')
    parser.add_argument('-g', '--gpuids', metavar='GPU', default=[0],
                        type=int, nargs='+',
                        help='GPU IDs for using (default: 0)')
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    cfg = parser.parse_args()
    return cfg


@ex.config
def hyperparam():
    """
    sacred exmperiment hyperparams
    :return:
    """
    args = configuration()


@ex.main
def main(args):
    global arch_name
    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # set model name
    arch_name = set_arch_name(args)

    print('\n=> creating model \'{}\''.format(arch_name))
    model = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                       width_mult=args.width_mult)

    if model is None:
        print('==> unavailable model parameters!! exit...\n')
        exit()

    # checkpoint file
    ckpt_dir = pathlib.Path('checkpoint') / arch_name / args.dataset
    ckpt_file = ckpt_dir / args.ckpt

    # for evaluation
    if args.evaluate:
        if args.cuda:
            torch.cuda.set_device(args.gpuids[0])
            with torch.cuda.device(args.gpuids[0]):
                model = model.cuda()
                criterion = criterion.cuda()
            model = nn.DataParallel(model, device_ids=args.gpuids,
                                    output_device=args.gpuids[0])
            cudnn.benchmark = True

        # Data loading
        print('==> Load data..')
        start_time = time.time()
        train_loader, val_loader = DataLoader(args.batch_size, args.workers,
                                            args.dataset, args.datapath,
                                            args.cuda)
        elapsed_time = time.time() - start_time
        print('===> Data loading time: {:,}m {:.2f}s'.format(
            int(elapsed_time//60), elapsed_time%60))
        print('===> Data loaded..')

        if isfile(ckpt_file):
            print('==> Loading Checkpoint \'{}\''.format(args.ckpt))
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=args.gpuids[0], use_cuda=args.cuda)
            print('==> Loaded Checkpoint \'{}\''.format(args.ckpt))

            # evaluate on validation set
            print('\n===> [ Evaluation ]')
            start_time = time.time()
            acc1, acc5 = validate(args, val_loader, None, model, criterion)
            elapsed_time = time.time() - start_time
            acc1 = round(acc1.item(), 4)
            acc5 = round(acc5.item(), 4)
            ckpt_name = '{}-{}-{}'.format(arch_name, args.dataset, args.ckpt[:-4])
            save_eval([ckpt_name, acc1, acc5])
            print('====> {:.2f} seconds to evaluate this model\n'.format(
                elapsed_time))
            return acc1
        else:
            print('==> no checkpoint found \'{}\''.format(
                args.ckpt))
            exit()

    # load checkpoint for entropy coding
    if isfile(ckpt_file):
        print('==> Loading Checkpoint \'{}\''.format(args.ckpt))
        checkpoint = load_model(model, ckpt_file,
                                main_gpu=args.gpuids[0], use_cuda=args.cuda)
        print('==> Loaded Checkpoint \'{}\''.format(args.ckpt))
    else:
        print('==> no checkpoint found \'{}\''.format(
            opt.ckpt))
        exit()

    # set encoder
    encoder = deepCABAC.Encoder()

    interv = 0.1
    stepsize = 15
    _lambda = 0.

    # encoding..
    print('==> Encoding..')
    for name, param in model.state_dict().items():
        if '.weight' in name:
            encoder.encodeWeightsRD( weights, interv, stepsize, _lambda )
        else:
            encoder.encodeWeightsRD( weights, interv, stepsize + 4, _lambda )
    
    stream = encoder.finish()
    with open(ckpt_dir / 'encoded_weights.bin', 'wb') as f:
        f.write(stream)


def validate(args, val_loader, epoch, model, criterion):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # logging at sacred
    ex.log_scalar('test.loss', losses.avg, epoch)
    ex.log_scalar('test.top1', top1.avg.item(), epoch)
    ex.log_scalar('test.top5', top5.avg.item(), epoch)

    return top1.avg, top5.avg


if __name__ == '__main__':
    start_time = time.time()
    ex.run()
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))