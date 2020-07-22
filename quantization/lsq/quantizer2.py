import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Inferer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, q_n, q_p):
        x_n = x / s
        x_q = x_n.clamp(q_n, q_p).round() * s
        
        ctx.save_for_backward(x_n)
        ctx.other = q_n, q_p
        return x_q

    @staticmethod
    def backward(ctx, grad_x):
        x_n = ctx.saved_tensors[0]
        q_n, q_p = ctx.other
        
        gs = math.sqrt(x_n.numel() * q_p)
        
        idx_s = (x_n <= q_n).float()
        idx_l = (x_n >= q_p).float()
        idx_m = torch.ones(size=idx_s.shape, device=idx_s.device) - idx_s - idx_l
        
        grad_s = ((idx_s * q_n + idx_l * q_p + idx_m * (-x_n + x_n.round())) * grad_x / gs).sum().unsqueeze(dim=0)
        grad_x = idx_m * grad_x
        return grad_x, grad_s, None, None


class LSQ(nn.Module):
    def __init__(self, nbits, q_n, q_p):
        super().__init__()
        self.nbits = Parameter(torch.Tensor(1), requires_grad=False)
        self.nbits.fill_(nbits)

        self.q_n = q_n
        self.q_p = q_p

        self.step = Parameter(torch.Tensor(1))
        self.register_buffer('do_init', torch.zeros(1))

    @property
    def step_abs(self):
        return self.step.abs()

    def init_step(self, x, *args, **kwargs):
        self.step.data.copy_(
            2. * x.abs().mean() / math.sqrt(self.q_p)
        )
        self.do_init.fill_(1)

    def forward(self, x):
        if self.training and self.do_init == 0:
            self.init_step(x)
        
        return Inferer.apply(x, self.step_abs, self.q_n, self.q_p)

    def extra_repr(self):
        return 'nbits={}, q_n={}, q_p={}'.format(int(self.nbits[0]), int(self.q_n), int(self.q_p))
