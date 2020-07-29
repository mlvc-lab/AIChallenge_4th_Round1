"""MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
(https://arxiv.org/abs/1704.04861)
for more details.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    """Original MobileNet"""
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNet, self).__init__()
        input_channel = 32
        last_channel = self.cfg[-1]
        self.width_mult = width_mult
        
        # building first layer
        input_channel = int(input_channel * self.width_mult)
        last_channel = int(last_channel * max(1.0, self.width_mult))
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        # building depth-wise separable convolutional layers
        self.layers = self._make_layers(in_planes=input_channel)
        # building last average pooling layer
        self.avg_pool = nn.AvgPool2d(7)
        # buiulding classifier
        self.linear = nn.Linear(last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layers(self, in_planes):
        layers = []
        alpha = self.width_mult
        for x in self.cfg:
            out_planes = int(x * alpha) if isinstance(x, int) else int(x[0] * alpha)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MobileNet_CIFAR(nn.Module):
    """MobileNet for CIFAR-10/100"""
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNet_CIFAR, self).__init__()
        input_channel = 32
        last_channel = self.cfg[-1]
        self.width_mult = width_mult
        
        # building first layer
        input_channel = int(input_channel * self.width_mult)
        last_channel = int(last_channel * max(1.0, self.width_mult))
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        # building depth-wise separable convolutional layers
        self.layers = self._make_layers(in_planes=input_channel)
        # building last average pooling layer
        self.avg_pool = nn.AvgPool2d(2)
        # buiulding classifier
        self.linear = nn.Linear(last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layers(self, in_planes):
        layers = []
        alpha = self.width_mult
        for x in self.cfg:
            out_planes = int(x * alpha) if isinstance(x, int) else int(x[0] * alpha)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def mobilenet(data='cifar10', **kwargs):
    r"""MobileNet models from "[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)"

    Args:
        data (str): the name of datasets
    """
    width_mult = kwargs.get('width_mult')

    if data in ['cifar10', 'cifar100']:
        model = MobileNet_CIFAR(int(data[5:]), width_mult)
        image_size = 32
    elif data == 'imagenet':
        model = MobileNet(1000, width_mult)
        image_size = 224
    elif data == 'things':
        model = MobileNet(41, width_mult)
        image_size = 224
    else:
        model = None
        image_size = None

    return model, image_size