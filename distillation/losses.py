import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



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
