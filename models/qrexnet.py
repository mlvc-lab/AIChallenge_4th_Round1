"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

################# CODE CHANGE NOT FINISHED 

import torch
import torch.nn as nn
from math import ceil

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def _add_conv(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(Conv2dLSQ(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def _add_conv_swish(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(Swish())


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            _add_conv_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        _add_conv(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels,
                  active=False)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        _add_conv(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x

        return out


class ReXNetV1(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=1000,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 bn_momentum=0.9):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        self.depth = sum(layers[:]) * 3

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        _add_conv_swish(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        if use_se:
            use_ses = [False] * (layers[0] + layers[1]) + [True] * sum(layers[2:])
        else:
            use_ses = [False] * sum(layers[:])

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_se=se, se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        _add_conv_swish(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.output = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, classes, 1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x).squeeze()
        return x






class LSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, Qn, Qp):
        q_x = x / alpha
        x_q = q_x.clamp(Qn, Qp).round() * alpha
        
        ctx.save_for_backward(q_x, alpha)
        ctx.other = Qn, Qp
        return x_q

    @staticmethod
    def backward(ctx, grad_weight):
        q_x, alpha = ctx.saved_tensors
        Qn, Qp = ctx.other

        g = math.sqrt(q_x.numel() * Qp)
        
        indicate_small = (q_x <= Qn).float()
        indicate_big = (q_x >= Qp).float()
        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_x + q_x.round())) * grad_weight / g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, grad_alpha, None, None, None


class Conv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1, bias=True, nbits=32):
        if padding is None:
            padding = kernel_size // 2
        
        super(Conv2dLSQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        if nbits < 0 or nbits == 32:
            self.register_parameter('alpha', None)
            return
        
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(Conv2dLSQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}'.format(s_prefix, self.nbits)
    

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2. * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1)))
            #self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        
        w_q = LSQ.apply(self.weight, self.alpha, Qn, Qp)
        #print("Conv2d alpha : ", self.alpha)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ActLSQ(nn.Module):
    def __init__(self, nbits=32, func='ReLU'):
        super(ActLSQ, self).__init__()
        if nbits < 0 or nbits == 32:
            self.register_parameter('alpha', None)
            self.func = func
            return
        
        self.nbits = 32#nbits
        self.func = func
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(ActLSQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, func={}'.format(s_prefix, self.nbits, self.func)

    def forward(self, x):
        if self.alpha is None:
            if self.func == 'ReLU':
                return F.relu(x, inplace=True)
            elif self.func == 'Identity':
                return x
            else:
                assert False
        
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2. * x.abs().mean() / math.sqrt(2 ** (self.nbits - 1)))
            #self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * 0.25)
            self.init_state.fill_(1)
        
        if self.func == 'ReLU':
            Qn = 0
            Qp = 2 ** self.nbits - 1
        elif self.func == 'Identity':
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            assert False
        
        x_q = LSQ.apply(x, self.alpha, Qn, Qp)
        #print("Act alpha : ", self.alpha)
        return x_q