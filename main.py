import time
import random
import pathlib
from os.path import isfile

import numpy as np
import cv2

from adamp import AdamP, SGDP

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import models
import config
import pruning
import quantization
import distillation
from utils import *
from data import DataLoader

# for ignore ImageNet PIL EXIF UserWarning and ignore transparent images
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "(Palette )?images with Transparency", UserWarning)

# for sacred logging
from sacred import Experiment
from sacred.observers import MongoObserver

# sacred experiment
ex = Experiment('AI-Challenge')
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

    assert not (args.prune and args.quantize), "You should choose one --prune or --quantize"
    if not args.prune and not args.quantize:  # base
        if not args.distill:    
            model, image_size = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                           width_mult=args.width_mult,
                                                           depth_mult=args.depth_mult,
                                                           model_mult=args.model_mult)
        elif args.distill: # for distillation
            model, image_size = distillation.models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                                        width_mult=args.width_mult,
                                                                        depth_mult=args.depth_mult,
                                                                        model_mult=args.model_mult)
    elif args.prune:    # for pruning
        pruner = pruning.__dict__[args.pruner]
        model = pruning.models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                   width_mult=args.width_mult,
                                                   depth_mult=args.depth_mult,
                                                   model_mult=args.model_mult,
                                                   mnn=pruner.mnn)
    elif args.quantize: # for quantization
        quantizer = quantization.__dict__[args.quantizer]
        model, image_size = quantization.models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                                    width_mult=args.width_mult,
                                                                    depth_mult=args.depth_mult,
                                                                    model_mult=args.model_mult,
                                                                    qnn=quantizer.qnn if args.run_type == 'train' else quantizer.iqnn,
                                                                    bitw=args.quant_bitw, bita=args.quant_bita,
                                                                    qcfg=quantizer.__dict__[args.quant_cfg])
    if args.distill: # for distillation
        teacher_name = set_arch_tch_name(args)
        distiller = distillation.losses.__dict__[args.dist_type]()
        teacher, tch_image_size = distillation.models.__dict__[args.tch_arch](data=args.dataset, num_layers=args.tch_layers,
                                                                              width_mult=args.tch_width_mult,
                                                                              depth_mult=args.tch_depth_mult,
                                                                              model_mult=args.tch_model_mult)
        assert image_size == tch_image_size, "The image size of student and teach should be the same."
    
    assert model is not None, 'Unavailable model parameters!! exit...\n'
    # for distillation
    if args.distill:
        assert teacher is not None, 'Unavailable teacher model parameters!! exit...\n'

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimizer == 'AdamW':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(args.momentum, 0.999),
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'SGDP':
        optimizer = SGDP(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay,
                         nesterov=args.nesterov)
    elif args.optimizer == 'AdamP':
        optimizer = AdamP(model.parameters(), lr=args.lr,
                          betas=(args.momentum, 0.999),
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    scheduler = set_scheduler(optimizer, args)
    
    # set multi-gpu
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True
        
        # for distillation
        if args.distill:
            teacher = torch.nn.DataParallel(teacher, device_ids=args.gpuids,
                                            output_device=args.gpuids[0]).cuda()

    # Dataset loading
    print('==> Load data..')
    start_time = time.time()
    train_loader, val_loader = DataLoader(args.batch_size, args.workers,
                                          args.dataset, args.datapath, image_size,
                                          args.cuda)
    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time//60), elapsed_time%60))
    print('===> Data loaded..')

    # load a pre-trained model
    if args.load is not None:
        if args.transfer:   # for transfer learning
            ckpt_file = pathlib.Path('checkpoint') / arch_name / args.src_dataset / args.load
        else:
            ckpt_file = pathlib.Path('checkpoint') / arch_name / args.dataset / args.load
        assert isfile(ckpt_file), '==> no checkpoint found \"{}\"'.format(args.load)

        print('==> Loading Checkpoint \'{}\''.format(args.load))
        # check pruning or quantization or transfer
        strict = False if args.prune or args.quantize or args.transfer else True
        # load a checkpoint
        if args.run_type == 'evaluate' and args.quantize:
            quantization.load_quant_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
        else:
            checkpoint = load_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
        print('==> Loaded Checkpoint \'{}\''.format(args.load))

    # for transfer
    if args.transfer:
        if args.dataset == 'ciar10':
            target_class_number = 10
        elif args.dataset == 'ciar100':
            target_class_number = 100
        elif args.dataset == 'imagenet':
            target_class_number = 1000
        elif args.dataset == 'things':
            target_class_number = 41
        
        if args.arch == "efficientnet":
            in_channel = model.module._fc.in_features
            model.module._fc = nn.Linear(in_channel, target_class_number).cuda()

        elif args.arch == "rexnet":
            in_channel = model.module.output[1].in_channels
            model.module.output[1] = nn.Conv2d(in_channel, target_class_number, 1, bias=True).cuda()

        elif args.arch == "resnet":
            in_channel = model.module.fc.in_features
            model.module.fc = nn.Linear(in_channel, target_class_number).cuda()

        elif args.arch == "mobilenetv2":
            in_channel = model.module.classifier[1].in_features
            model.module.classifier[1] = nn.Linear(in_channel, target_class_number).cuda()
            
        elif args.arch == "mobilenetv3":
            in_channel = model.module.classifier[1].in_features
            model.module.classifier[1] = nn.Linear(in_channel, target_class_number).cuda()
                
        else:
            assert False, "wrong model name input"
        print('==> modified for transfer learning')

    # for distillation
    if args.distill:
        assert args.tch_load is not None
        tch_ckpt_file = pathlib.Path('checkpoint') / teacher_name / args.dataset / args.tch_load
        assert isfile(tch_ckpt_file), '==> no checkpoint found \"{}\"'.format(args.tch_load)

        print('==> Loading Teacher Checkpoint \'{}\''.format(args.tch_load))
        tch_checkpoint = load_model(teacher, tch_ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda)
        print('==> Loaded Teacher Checkpoint \'{}\''.format(args.tch_load))


    # for training
    if args.run_type == 'train':
        # init parameters
        start_epoch = 0
        global iterations
        iterations = 0
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
            if not args.distill:
                acc1_train, acc5_train = train(args, train_loader,
                    epoch=epoch, model=model,
                    criterion=criterion, optimizer=optimizer)
            else:   # for distillation
                acc1_train, acc5_train = train(args, train_loader,
                    epoch=epoch, model=model,
                    criterion=criterion, optimizer=optimizer,
                    teacher=teacher, distiller=distiller)
            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set
            print('===> [ Validation ]')
            start_time = time.time()
            acc1_valid, acc5_valid = validate(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate this epoch'.format(
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
                save_model(arch_name, args.dataset, state, args.save)
            save_summary(arch_name, args.dataset, args.save.split('.pth')[0], summary)

            # for pruning
            if args.prune:
                num_total, num_zero, sparsity = pruning.cal_sparsity(model)
                print('\n====> sparsity: {:.2f}% || num_zero/num_total: {}/{}'.format(sparsity, num_zero, num_total))
                # logging at sacred
                ex.log_scalar('sparsity', sparsity, epoch)

            # end of one epoch
            print()

        # calculate the total training time 
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
    
    elif args.run_type == 'evaluate':   # for evaluation
        # for evaluation on validation set
        print('\n===> [ Evaluation ]')
        
        # main evaluation
        start_time = time.time()
        acc1, acc5 = validate(args, val_loader, None, model, criterion)
        elapsed_time = time.time() - start_time
        print('====> {:.2f} seconds to evaluate this model\n'.format(
            elapsed_time))
        
        acc1 = round(acc1.item(), 4)
        acc5 = round(acc5.item(), 4)

        # save the result
        ckpt_name = '{}-{}-{}'.format(arch_name, args.dataset, args.load[:-4])
        save_eval([ckpt_name, acc1, acc5])
        
        return acc1
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'
    

def train(args, train_loader, epoch, model, criterion, optimizer, **kwargs):
    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))
    # for things dataset
    if args.dataset == 'things':
        f1 = ScoreMeter()

    # switch to train mode
    model.train()
    
    # for distillation
    if args.distill:
        teacher = kwargs.get('teacher')
        distiller = kwargs.get('distiller')
        # teacher.eval()
        teacher.eval()

    end = time.time()
    #with torch.autograd.set_detect_anomaly(True):
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            #if args.dataset == 'things':
            #    input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # for pruning
        if args.prune:
            if (globals()['iterations']+1) % args.prune_freq==0 and (epoch+1) <= args.milestones[1]:
                target_sparsity = args.prune_rate - args.prune_rate * (1 - (globals()['iterations'])/(args.milestones[1] * len(train_loader)))**3
                if args.prune_type == 'structured':
                    importance = pruning.get_filter_importance(model)
                    pruning.filter_prune(model, importance, target_sparsity * 100)
                elif args.prune_type == 'unstructured':
                    threshold = pruning.get_weight_threshold(model, target_sparsity * 100)
                    pruning.weight_prune(model, threshold)

        # for data augmentation
        if args.augmentation:
            if args.mixed_aug:  # random choice
                args.aug_type = np.random.choice(['cutmix', 'saliencymix', 'mixup', 'cutout'], 1, p=[0.5, 0.2, 0.1, 0.2])
                
            aug_prob = np.random.rand(1)
            if aug_prob < args.aug_prob:   # do augmentation
                if args.aug_type == 'cutmix':
                    # permutations
                    rand_index = torch.randperm(input.size()[0]).cuda()
                    # set random variables
                    lam = np.random.beta(args.aug_beta, args.aug_beta)
                    bbx1, bby1, bbx2, bby2 = cutmix(input.size(), lam)
                    # generate mixed input and target
                    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                    target_a = target
                    target_b = target[rand_index]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                elif args.aug_type == 'saliencymix':
                    # permutations
                    rand_index = torch.randperm(input.size()[0]).cuda()
                    # set random variables
                    lam = np.random.beta(args.aug_beta, args.aug_beta)
                    bbx1, bby1, bbx2, bby2 = saliencymix(input[rand_index[0]], lam)
                    # generate mixed input and target
                    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                    target_a = target
                    target_b = target[rand_index]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                elif args.aug_type == 'mixup':
                    input, target_a, target_b, lam = mixup_data(input, target, args.mixup_alpha, args.cuda)
                    input, target_a, target_b = map(Variable, (input, target_a, target_b))

                elif args.aug_type == 'cutout':
                    input = Cutout(args.cut_nholes, args.cut_length).__call__(input)
        
        # cal output (normal or distillation)
        if not args.distill:
            # compute output
            output = model(input)
        else:
            output, s_features = model(input, args.dist_type)
            t_output, t_features = teacher(input, args.dist_type)
        
        # cal ce_loss (normal or two targets)
        if args.augmentation:
            if aug_prob < args.aug_prob:
                if args.aug_type == 'cutmix':
                    ce_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                elif args.aug_type == 'saliencymix':
                    ce_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                elif args.aug_type == 'mixup':
                    ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                elif args.aug_type == 'cutout':
                    ce_loss = criterion(output, target)
            else:
                ce_loss = criterion(output, target)
        else:
            ce_loss = criterion(output, target)

        # for distillation to cal distill loss
        if args.distill:
            loss = distiller.add_dist_loss(ce_loss, s_output=output, t_output=t_output,
                                           s_features=s_features, t_features=t_features)
        else:
            loss = ce_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        # for things dataset
        if args.dataset == 'things':
            f1.update(output, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

        end = time.time()

        # end of one mini-batch
        globals()['iterations'] += 1

    # for things dataset
    if args.dataset == 'things':
        f1.cal_score()

    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    # for things dataset
    if args.dataset == 'things':
        print('====> marco mean of F1 scores: {:.6f}'
              .format(f1.score))

    # logging at sacred
    ex.log_scalar('train.loss', losses.avg, epoch)
    ex.log_scalar('train.top1', top1.avg.item(), epoch)
    ex.log_scalar('train.top5', top5.avg.item(), epoch)
    # for things dataset
    if args.dataset == 'things':
        ex.log_scalar('train.f1', f1.score, epoch)

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
    # for things dataset
    if args.dataset == 'things':
        f1 = ScoreMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                #if args.dataset == 'things':
                #    input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            # for things dataset
            if args.dataset == 'things':
                f1.update(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()

        # for things dataset
        if args.dataset == 'things':
            f1.cal_score()

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        # for things dataset
        if args.dataset == 'things':
            print('====> marco mean of F1 scores: {:.6f}'
                  .format(f1.score))

    # logging at sacred
    ex.log_scalar('test.loss', losses.avg, epoch)
    ex.log_scalar('test.top1', top1.avg.item(), epoch)
    ex.log_scalar('test.top5', top5.avg.item(), epoch)
    # for things dataset
    if args.dataset == 'things':
        ex.log_scalar('test.f1', f1.score, epoch)

    return top1.avg, top5.avg


# Added for Cut-Mix
def cutmix(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def saliencymix(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# Added for Mixup data augmentation
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((1, 1, h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[0, 0, y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * mask

        return img


if __name__ == '__main__':
    start_time = time.time()
    ex.run()
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))
