import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mcanet", "MCABlock"]


class MCANet(nn.Module):
    def __init__(self, num_classes):
        super(MCANet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MCABlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mcanet(num_classes):
    return MCANet(num_classes=num_classes)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class MCABlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=True, use_cbam=False,
                 padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(MCABlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        padding_11 = padding - kernel_size // 2

        # self.nonlinearity = nn.SiLU()
        self.nonlinearity = nn.ReLU()

        if use_se or use_cbam:
            if use_se:
                self.att = SEBlock(out_channels, internal_neurons=out_channels // 16)
            if use_cbam:
                self.att = CBAM(out_channels, out_channels // 16)
        else:
            self.att = nn.Identity()

        self.rbr_identity = nn.BatchNorm2d(
            num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                               padding=padding_11, groups=groups)

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.att(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

# SE
class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool= nn.AdaptiveAvgPool2d(1)
        self.max_pool= nn.AdaptiveMaxPool2d(1)

        self.fc1= nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1= nn.ReLU()
        self.fc2= nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        avg_out= self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out= self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out= avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention,self).__init__()
        self.conv1= nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)  # kernel size = 7 Padding is 3: (n - 7 + 1) + 2P = n
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channelIn, channelOut):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channelIn)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out