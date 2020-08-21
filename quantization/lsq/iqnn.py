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
        self.step = Parameter(torch.Tensor(1))

    def forward(self, x):
        quantized_weight = self.weight * self.step
        return self._conv_forward(x, quantized_weight)


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=32, symmetric=False, do_mask=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.step = Parameter(torch.Tensor(1))

    def forward(self, x):
        quantized_weight = self.weight * self.step
        return F.linear(x, quantized_weight, self.bias)
