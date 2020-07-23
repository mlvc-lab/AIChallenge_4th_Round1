import argparse
import shutil
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import math
import sys
import random
#sys.path.append('/home/youmin/SelfDistillation/')
from qdmodel import *
from scipy.stats import norm
# from advertorch.attacks import *

import config
# for sacred logging
from sacred import Experiment
from sacred.observers import MongoObserver

# sacred experiment
ex = Experiment('AI-Challenge_QDistill')
ex.observers.append(MongoObserver.create(url=config.MONGO_URI,
                                         db_name=config.MONGO_DB))



parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test', default='', type=str, metavar='PATH',
                    help='path to trained model (default: none)')
parser.add_argument('--type', default='cifar100', type=str, help='choose dataset cifar10, cifar100, imagenet')
parser.add_argument('--teacher', default='wideresnet', type=str, help='choose pre-trained teacher')
parser.add_argument('--student', default='resnet', type=str, help='choose to be trained student')
# for teacher
# for all resnet
parser.add_argument('--depth', type=int, default=40, help='model depth for resnet, wideresnet, resnext')
# for wideresnet
parser.add_argument('--wfactor', type=int, default=4, help='wide factor for wideresnet')
# for densenet
parser.add_argument('--kfactor', type=int, default=0, help='growth rate for densenet')
# index of each training runs
parser.add_argument('--tn', type=str, default='', help='n-th training')
# for student
# for all resnet
parser.add_argument('--sdepth', type=int, default=56, help='model depth for resnet, wideresnet, resnext')
# for wideresnet
parser.add_argument('--swfactor', type=int, default=1, help='wide factor for wideresnet')
# for densenet
parser.add_argument('--skfactor', type=int, default=0, help='growth rate for densenet')
# index of each training runs
parser.add_argument('--stn', type=str, default='', help='n-th training')
# distillation method for training student
parser.add_argument('--distype', type=str, default='KD', help='self distillation type, empty means exit')
parser.add_argument('--nl', type=int, default=1, help='number of locals (default: 1)')
parser.add_argument('--gn', type=int, default=1, help='SD number of groups (default: 1)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
best_prec1 = 0

@ex.config
def hyperparam():
    """
    sacred exmperiment hyperparams
    :return:
    """
    # args = config.config()
    args = parser.parse_args()


@ex.main
def main(args):
    global best_prec1
    teacher_name = ''
    student_name = '_distilled_by_'
    class_num = 0
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # trained model test code
    if args.test != '':
        print("=> Testing trained weights ")
        checkpoint = torch.load(args.test)
        print("=> loaded test checkpoint: {} epochs, Top1 Accuracy: {}, Top5 Accuracy: {}".format(checkpoint['epoch'],
                                                                                                  checkpoint[
                                                                                                      'test_acc1'],
                                                                                                  checkpoint[
                                                                                                      'test_acc5']))
        return
    else:
        print("=> No Test ")

    # data loader setting
    if args.type == 'cifar10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR10(root='/dataset/CIFAR', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='/dataset/CIFAR', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        class_num = 10
    elif args.type == 'cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR100(root='/dataset/CIFAR', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='/dataset/CIFAR', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
        class_num = 100
    elif args.type == 'imagenet':
        traindir = os.path.join('/dataset/ImageNet/train')
        valdir = os.path.join('/dataset/ImageNet/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),
                                                             transforms.CenterCrop(224), transforms.ToTensor(),
                                                             normalize,])), batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
        class_num = 1000
    else:
        print("No dataset")

    # load the pre-trained teacher
    if args.teacher == 'resnet':
        print('ResNet CIFAR10, CIFAR100 : 20(0.27M) 32(0.46M), 44(0.66M), 56(0.85M), 110(1.7M)\n'
              'ImageNet 18(11.68M), 34(21.79M), 50(25.5M)')
        cifar_list = [20, 32, 44, 56, 110]
        # CIFAR10, CIFAR100
        if args.depth in cifar_list:
            assert (args.depth - 2) % 6 == 0
            n = int((args.depth - 2) / 6)
            teacher = ResNet_Cifar(BasicBlock, [n, n, n], num_classes=class_num)
        # ImageNet 18(11.68M), 34(21.79M), 50(25.5M)
        elif args.depth == 18:
            teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=class_num)
        elif args.depth == 34:
            teacher = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=class_num)
        elif args.depth == 50:
            teacher = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=class_num)
        else:
            print("Inappropriate ResNet Teacher model")
            return
        teacher_name = args.teacher+str(args.depth)
    elif args.teacher == 'wideresnet':
        print('WideResNet CIFAR10, CIFAR100 : 40_1(0.6M), 40_2(2.2M), 40_4(8.9M), 40_8(35.7M), 28_10(36.5M), 28_12(52.5M),'
            ' 22_8(17.2M), 22_10(26.8M), 16_8(11.0M), 16_10(17.1M)')
        assert (args.depth - 4) % 6 == 0
        n = int((args.depth - 4) / 6)
        teacher = Wide_ResNet_Cifar(BasicBlock, [n, n, n], wfactor=args.wfactor, num_classes=class_num)
        teacher_name = args.teacher+str(args.depth)+'_'+str(args.wfactor)
    elif args.model == 'densenetbc':
        print('DenseNetBC<k, depth> CIFAR10, CIFAR100 : <12,100>(0.8M), <24,250>(15.3M), <40,190>(25.6M)')
        assert (args.depth - 4) % 6 == 0
        n = int((args.depth - 4) / 6)
        teacher = DenseNet(growth_rate=args.kfactor, block_config=[n, n, n], num_init_features=2 * args.kfactor, num_classes=class_num)
        teacher_name = args.teacher + str(args.kfactor) + '_' + str(args.depth)
    elif args.teacher == 'mobilenetv2':
        teacher = MobileNetV2(num_classes=class_num)
        teacher_name = args.teacher
    elif args.model == 'mobilenet':
        teacher = MobileNet(num_classes=class_num)
        teacher_name = args.teacher
    else:
        print("No Teacher model")
        return

    # create student
    if args.student == 'resnet': # QUANTIZED STUDENT!
        print('ResNet CIFAR10, CIFAR100 : 20(0.27M) 32(0.46M), 44(0.66M), 56(0.85M), 110(1.7M)\n'
              'ImageNet 18(11.68M), 34(21.79M), 50(25.5M)')
        cifar_list = [20, 32, 44, 56, 110]
        # CIFAR10, CIFAR100
        if args.sdepth in cifar_list:
            assert (args.sdepth - 2) % 6 == 0
            n = int((args.sdepth - 2) / 6)
            student = QResNet_Cifar(QBasicBlock, [n, n, n], num_classes=class_num)
        # ImageNet 18(11.68M), 34(21.79M), 50(25.5M)
        elif args.sdepth == 18:
            student = QResNet(QBasicBlock, [2, 2, 2, 2], num_classes=class_num)
        elif args.sdepth == 34:
            student = QResNet(QBasicBlock, [3, 4, 6, 3], num_classes=class_num)
        elif args.sdepth == 50:
            student = QResNet(QBottleneck, [3, 4, 6, 3], num_classes=class_num)
        else:
            print("Inappropriate ResNet Student model")
            return
        student_name = args.student + str(args.sdepth) + student_name + teacher_name + '_' + str(args.tn) + 'th'
    elif args.student == 'wideresnet':
        print('WideResNet CIFAR10, CIFAR100 : 40_1(0.6M), 40_2(2.2M), 40_4(8.9M), 40_8(35.7M), 28_10(36.5M), 28_12(52.5M),'
            ' 22_8(17.2M), 22_10(26.8M), 16_8(11.0M), 16_10(17.1M)')
        assert (args.sdepth - 4) % 6 == 0
        n = int((args.sdepth - 4) / 6)
        student = Wide_ResNet_Cifar(BasicBlock, [n, n, n], wfactor=args.swfactor, num_classes=class_num)
        student_name = args.student + str(args.sdepth) + '_' + str(args.swfactor) + student_name + teacher_name + '_' + str(args.tn) + 'th'
    elif args.model == 'densenet':
        print('DenseNetBC<k, depth> CIFAR10, CIFAR100 : <12,100>(0.8M), <24,250>(15.3M), <40,190>(25.6M)')
        assert (args.sdepth - 4) % 6 == 0
        n = int((args.sdepth - 4) / 6)
        teacher = DenseNet(growth_rate=args.skfactor, block_config=[n, n, n], num_init_features=2 * args.skfactor, num_classes=class_num)
        teacher_name = args.teacher + str(args.skfactor) + '_' + str(args.sdepth)
    elif args.student == 'mobilenetv2':
        student = MobileNetV2(num_classes=class_num)
        student_name = args.student + student_name + teacher_name + '_' + str(args.tn) + 'th'
    elif args.model == 'mobilenet':
        student = MobileNet(num_classes=class_num)
        student_name = args.student + student_name + teacher_name + '_' + str(args.tn) + 'th'
    else:
        print("No Student model")
        return

    # print pre-trained teacher and to-be-trained student information
    t_num_parameters = round((sum(l.nelement() for l in teacher.parameters()) / 1e+6), 3)
    s_num_parameters = round((sum(l.nelement() for l in student.parameters()) / 1e+6), 3)
    print("teacher name : ", teacher_name)
    print("teacher parameters : ", t_num_parameters, "M")
    print("student name : ", student_name)
    print("student parameters : ", s_num_parameters, "M")
    teacher = torch.nn.DataParallel(teacher).cuda()
    load_teacher_progress = '/root/volume/AIchallenge/distill_qt/wr40_4'# + args.type + '/' + teacher_name + '/B/' + str(args.tn)
    teacher.load_state_dict(torch.load(load_teacher_progress + '/best_weight.pth'))
    student = torch.nn.DataParallel(student).cuda()

    # define optimizer or loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(student.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # make progress save directory
    save_progress = '/root/volume/AIchallenge/distill_qt/progress' + args.type + '/' + student_name + '/' + args.distype + '/' + str(args.stn)
    if not os.path.isdir(save_progress):
        os.makedirs(save_progress)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args,optimizer, epoch)

        tr_acc, tr_acc5, tr_fc_loss, tr_d_loss = distillation(args, train_loader, teacher, student, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1, prec5, te_fc_loss = test(args,val_loader, student, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1, 'train_fc_loss': tr_fc_loss, 'train_d_loss': tr_d_loss, 'test_fc_loss': te_fc_loss,
                         'train_acc1': tr_acc, 'train_acc5': tr_acc5, 'test_acc1': prec1, 'test_acc5': prec5}, is_best, save_progress)
        torch.save(student.state_dict(), save_progress + '/weight.pth')
        if is_best:
            torch.save(student.state_dict(), save_progress + '/best_weight.pth')

    print('Best accuracy (top-1):', best_prec1)


def attention(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_loss(t, s):
    return (attention(t) - attention(s)).pow(2).mean()


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def similarity_preserve_loss(t, s):
    bsz = s.size()[0]
    f_s = s.view(bsz, -1)
    f_t = t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    # G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss


def distillation(args, train_loader, teacher, student, criterion, optimizer, epoch, reg=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    dis_losses = AverageMeter()
    dis_losses2 = AverageMeter()
    # local_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    student.train()
    # teacher.eval()
    teacher.train()
    end = time.time()
    with torch.autograd.set_detect_anomaly(True):
        loss = 0
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            target = target.cuda()

            # distilling
            if args.distype == 'KD':
                alpha = 0.9
                T = 4
                t_output = teacher(input, type=args.distype)
                s_output = student(input, type=args.distype)
                # kd_loss = F.kl_div(F.log_softmax(s_output/T, dim=1), F.softmax(t_output/T, dim=1), size_average=False) * (T**2) / int(s_output.size()[0])
                kd_loss = F.kl_div(F.log_softmax(s_output / T, dim=1), F.softmax(t_output / T, dim=1), reduction='batchmean') * (T ** 2)
                ce_loss = criterion(s_output, target)
                loss = alpha * kd_loss + (1. - alpha) * ce_loss
                dis_loss = kd_loss
            elif args.distype == 'AT':
                beta = 1e+3
                att_loss = 0
                t_output, t_middle_output = teacher(input, type=args.distype)
                s_output, s_middle_output = student(input, type=args.distype)
                for k in range(len(t_middle_output)):
                    att_loss += attention_loss(t_middle_output[k].detach(), s_middle_output[k])
                ce_loss = criterion(s_output, target)
                loss = ce_loss + (beta / 2) * att_loss
                dis_loss = att_loss
            elif args.distype == 'SP':
                gamma = 3e+3
                t_output, t_middle_output = teacher(input, type=args.distype)
                s_output, s_middle_output = student(input, type=args.distype)
                sp_loss = similarity_preserve_loss(t_middle_output[2].detach(), s_middle_output[2])
                ce_loss = criterion(s_output, target)
                loss = ce_loss + gamma * sp_loss
                dis_loss = sp_loss


            # measure accuracy and record loss
            prec1, prec5 = accuracy(s_output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            ce_losses.update(ce_loss.item(), input.size(0))
            dis_losses.update(dis_loss.item(), input.size(0))
            # dis_losses2.update(dis_loss2.item(), input.size(0))
            # local_losses.update(local_d_loss.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                # print("kd_loss : ", float(kd_loss))
                # print("ce_loss : ", float(ce_loss))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # dis_loss_list = [dis_losses, dis_losses2]
        dis_loss_list = [dis_losses]

    # logging at sacred
    ex.log_scalar('train_ce.loss', ce_losses.avg, epoch)
    ex.log_scalar('train_dis.loss', dis_loss_list[0].avg, epoch)
    ex.log_scalar('train.top1', top1.avg, epoch)
    ex.log_scalar('train.top5', top5.avg, epoch)

    return top1.avg, top5.avg, ce_losses, dis_loss_list


def test(args,val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        # making adversarial samples
        # adversary = GradientSignAttack(predict=model, loss_fn=criterion)
        # input_adv = adversary.perturb(input)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # logging at sacred
    ex.log_scalar('test.loss', losses.avg, epoch)
    ex.log_scalar('test.top1', top1.avg, epoch)
    ex.log_scalar('test.top5', top5.avg, epoch)

    return top1.avg, top5.avg, losses


def save_checkpoint(state, is_best, save_path):
    save_dir = save_path
    torch.save(state, save_path + '/' + str(state['epoch']) + 'epoch_result.pth')
    if is_best:
        torch.save(state, save_dir + '/best_result.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args,optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.type.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.type == ('imagenet'):
        lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    ex.run()