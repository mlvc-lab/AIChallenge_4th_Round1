import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .quantizer import LSQ


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 nbits=32, symmetric=False, do_mask=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)

        assert nbits > 0 and nbits < 33
        self.nbits = nbits

        if symmetric:
            self.q_n = -1. * 2. ** (self.nbits - 1) + 1
            self.q_p = 2. ** (self.nbits - 1) - 1
        else:
            self.q_n = -1. * 2. ** (self.nbits - 1)
            self.q_p = 2. ** (self.nbits - 1) - 1

        if self.nbits != 32:
            self.quantizer = LSQ(self.nbits, self.q_n, self.q_p)

        if do_mask:
            self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.do_mask = do_mask

    def forward(self, x):
        if self.nbits == 32:
            quantized_weight = self.weight
        else:
            quantized_weight = self.quantizer(self.weight)

        if self.do_mask:
            quantized_weight = quantized_weight * self.mask
        
        return self._conv_forward(x, quantized_weight)


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=32, symmetric=False, do_mask=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        assert nbits > 0 and nbits < 33
        self.nbits = nbits

        if symmetric:
            self.q_n = -1. * 2. ** (self.nbits - 1) + 1
            self.q_p = 2. ** (self.nbits - 1) - 1
        else:
            self.q_n = -1. * 2. ** (self.nbits - 1)
            self.q_p = 2. ** (self.nbits - 1) - 1

        if self.nbits != 32:
            self.quantizer = LSQ(self.nbits, self.q_n, self.q_p)

        if do_mask:
            self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.do_mask = do_mask

    def forward(self, x):
        if self.nbits == 32:
            quantized_weight = self.weight
        else:
            quantized_weight = self.quantizer(self.weight)

        if self.do_mask:
            quantized_weight = quantized_weight * self.mask
        
        return F.linear(x, quantized_weight, self.bias)


class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, nbits=32):
        super(QuantReLU, self).__init__(inplace)

        assert nbits > 0 and nbits < 33
        self.nbits = nbits

        self.q_n = 0.
        self.q_p = 2. ** self.nbits - 1

        if self.nbits != 32:
            self.quantizer = LSQ(self.nbits, self.q_n, self.q_p)

    def forward(self, x):
        if self.nbits == 32:
            output = F.relu(x, inplace=self.inplace)
        else:
            output = self.quantizer(x)
        return output


class QuantIdentity(nn.Identity):
    def __init__(self, nbits=32, symmetric=True):
        super(QuantIdentity, self).__init__()
        
        assert nbits > 0 and nbits < 33
        self.nbits = nbits
        
        if symmetric:
            self.q_n = -1. * 2. ** (self.nbits - 1) + 1
            self.q_p = 2. ** (self.nbits - 1) - 1
        else:
            self.q_n = -1. * 2. ** (self.nbits - 1)
            self.q_p = 2. ** (self.nbits - 1) - 1

        if self.nbits != 32:
            self.quantizer = LSQ(self.nbits, self.q_n, self.q_p)

    def forward(self, x):
        if self.nbits == 32:
            output = x
        else:
            output = self.quantizer(x)
        return self.quantizer(x)
