import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class KD:
    def __init__(self):
        self.alpha = 0.9
        self.T = 4

    def add_dist_loss(self, ce_loss, **kwargs):
        # get arguments
        s_output = kwargs.get('s_output')
        t_output = kwargs.get('t_output')
        
        # calculate KD loss
        # kd_loss = F.kl_div(F.log_softmax(s_output/T, dim=1), F.softmax(t_output/T, dim=1), size_average=False) * (T**2) / int(s_output.size()[0])
        kd_loss = F.kl_div(F.log_softmax(s_output / self.T, dim=1), F.softmax(t_output / self.T, dim=1), reduction='batchmean') * (self.T ** 2)

        return self.alpha * kd_loss + (1. - self.alpha) * ce_loss


class AT:
    def __init__(self):
        self.beta = 1e+3

    def attention(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_loss(self, t_feature, s_feature):
        return (self.attention(t_feature) - self.attention(s_feature)).pow(2).mean()

    def add_dist_loss(self, ce_loss, **kwargs):
        # get arguments
        s_output = kwargs.get('s_output')
        t_output = kwargs.get('t_output')
        s_features = kwargs.get('s_features')
        t_features = kwargs.get('t_features')

        # calculate AT loss
        att_loss = 0
        for i in range(len(s_features)):
            att_loss += self.attention_loss(t_features[i].detach(), s_features[i])
        
        return ce_loss + (self.beta / 2) * att_loss


class SP:
    def __init__(self):
        self.gamma = 3e+3

    def similarity_preserve_loss(self, t_feature, s_feature):
        bsz = s_feature.size()[0]
        f_s = s_feature.view(bsz, -1)
        f_t = t_feature.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

    def add_dist_loss(self, ce_loss, **kwargs):
        # get arguments
        s_output = kwargs.get('s_output')
        t_output = kwargs.get('t_output')
        s_features = kwargs.get('s_features')
        t_features = kwargs.get('t_features')
        
        # calculate SP loss
        sp_loss = self.similarity_preserve_loss(t_features[-1].detach(), s_features[-1])
        
        return ce_loss + self.gamma * sp_loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
