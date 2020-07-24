import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KD:
    def __init__(self):
        return

    def criterion(loss, s_output, t_output, s_features, t_features):
        alpha = 0.9
        T = 4
        # kd_loss = F.kl_div(F.log_softmax(s_output/T, dim=1), F.softmax(t_output/T, dim=1), size_average=False) * (T**2) / int(s_output.size()[0])
        kd_loss = F.kl_div(F.log_softmax(s_output / T, dim=1), F.softmax(t_output / T, dim=1), reduction='batchmean') * (T ** 2)
        loss = alpha * kd_loss + (1. - alpha) * ce_loss
        return loss, kd_loss


class AT:
    def __init__(self):
        return

    def criterion(loss, s_output, t_output, s_features, t_features):
        beta = 1e+3
        att_loss = 0
        t_output, t_middle_output = teacher(input, type=args.distype)
        s_output, s_middle_output = student(input, type=args.distype)
        for k in range(len(t_middle_output)):
            att_loss += attention_loss(t_middle_output[k].detach(), s_middle_output[k])
        ce_loss = criterion(s_output, target)
        loss = ce_loss + (beta / 2) * att_loss
        dis_loss = att_loss
        return loss, dis_loss


class SP:
    def __init__(self):
        return

    def criterion(loss, s_output, t_output, s_features, t_features):
        gamma = 3e+3
        t_output, t_middle_output = teacher(input, type=args.distype)
        s_output, s_middle_output = student(input, type=args.distype)
        sp_loss = similarity_preserve_loss(t_middle_output[2].detach(), s_middle_output[2])
        ce_loss = criterion(s_output, target)
        loss = ce_loss + gamma * sp_loss
        dis_loss = sp_loss
        return loss, dis_loss
