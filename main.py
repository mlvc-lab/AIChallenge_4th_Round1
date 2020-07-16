import time
import pathlib
from os.path import isfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
import config
from utils import *
from data import DataLoader

# for ignore ImageNet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# for sacred logging
from sacred import Experiment
from sacred.observers import MongoObserver

# sacred experiment
ex = Experiment('AI-Challenge_Base')
ex.observers.append(MongoObserver.create(url=config.MONGO_URI,
                                         db_name=config.MONGO_DB))


@ex.config
def hyperparam():
    """
    sacred exmperiment hyperparams
    :return:
    """
    args = config.config()


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    scheduler = set_scheduler(optimizer, args)
    start_epoch = 0

    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True

    # checkpoint file
    ckpt_dir = pathlib.Path('checkpoint')
    ckpt_file = ckpt_dir / arch_name / args.dataset / args.ckpt

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

    # for evaluation
    if args.evaluate:
        if isfile(ckpt_file):
            print('==> Loading Checkpoint \'{}\''.format(args.ckpt))
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=args.gpuids[0], use_cuda=args.cuda)
            epoch = checkpoint['epoch']
            # logging at sacred
            ex.log_scalar('best_epoch', epoch)

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                args.ckpt, epoch))

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

    # train...
    best_acc1 = 0.0
    train_time = 0.0
    validate_time = 0.0
    for epoch in range(start_epoch, args.epochs):
        print('\n==> {}/{} training'.format(
                arch_name, args.dataset))
        print('==> Epoch: {}, lr = {}'.format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        print('===> [ Training ]')
        start_time = time.time()
        acc1_train, acc5_train = train(args, train_loader,
            epoch=epoch, model=model,
            criterion=criterion, optimizer=optimizer)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print('====> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # evaluate on validation set
        print('===> [ Validation ]')
        start_time = time.time()
        acc1_valid, acc5_valid = validate(args, val_loader, epoch, model, criterion)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print('====> {:.2f} seconds to validate this epoch\n'.format(
            elapsed_time))
        
        # learning rate schduling
        scheduler.step()

        acc1_train = round(acc1_train.item(), 4)
        acc5_train = round(acc5_train.item(), 4)
        acc1_valid = round(acc1_valid.item(), 4)
        acc5_valid = round(acc5_valid.item(), 4)

        # remember best Acc@1 and save checkpoint and summary csv file
        state = model.state_dict()
        summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

        is_best = acc1_valid > best_acc1
        best_acc1 = max(acc1_valid, best_acc1)
        if is_best:
            save_model(arch_name, args.dataset, state)
        save_summary(arch_name, args.dataset, summary)

    # calculate time 
    avg_train_time = train_time / (args.epochs - start_epoch)
    avg_valid_time = validate_time / (args.epochs - start_epoch)
    total_train_time = train_time + validate_time
    print('====> average training time each epoch: {:,}m {:.2f}s'.format(
        int(avg_train_time//60), avg_train_time%60))
    print('====> average validation time each epoch: {:,}m {:.2f}s'.format(
        int(avg_valid_time//60), avg_valid_time%60))
    print('====> training time: {}h {}m {:.2f}s'.format(
        int(train_time//3600), int((train_time%3600)//60), train_time%60))
    print('====> validation time: {}h {}m {:.2f}s'.format(
        int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
    print('====> total training time: {}h {}m {:.2f}s'.format(
        int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))

    return best_acc1


def train(args, train_loader, **kwargs):
    r"""Train model each epoch
    """
    epoch = kwargs.get('epoch')
    model = kwargs.get('model')
    criterion = kwargs.get('criterion')
    optimizer = kwargs.get('optimizer')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

        end = time.time()

    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # logging at sacred
    ex.log_scalar('train.loss', losses.avg, epoch)
    ex.log_scalar('train.top1', top1.avg.item(), epoch)
    ex.log_scalar('train.top5', top5.avg.item(), epoch)

    return top1.avg, top5.avg


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
