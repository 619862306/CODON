import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from math import sqrt

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

def logsumexp_2d(tensor):

    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class BaseNet_non(nn.Module):
    def __init__(self):
        super(BaseNet_non, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_inputc = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv14 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv15 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv16 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv17 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        self.non1_channel = ChannelGate(gate_channels=64, reduction_ratio=8, pool_types=['avg', 'max'])
        self.non1_spatial = SpatialGate()

        self.non2_channel = ChannelGate(gate_channels=64, reduction_ratio=8, pool_types=['avg', 'max'])
        self.non2_spatial = SpatialGate()

        self.non3_channel = ChannelGate(gate_channels=64, reduction_ratio=8, pool_types=['avg', 'max'])
        self.non3_spatial = SpatialGate()

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out_1_1 = self.relu(self.conv1_1(inputs))
        out_2_1 = self.relu(self.conv2_1(out_1_1))
        out_3_1 = self.relu(self.conv3_1(out_2_1))
        out_4_1 = self.relu(self.conv4_1(out_3_1))
        out_5_1 = self.relu(self.conv5_1(out_4_1))
        out_6_1 = self.relu(self.conv6_1(out_5_1))

        non1_channel = self.non1_channel(out_6_1)
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_8_1 = non1_channel * out_8_1 + out_6_1

        non1_spatial = self.non1_spatial(out_8_1)
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_10_1 = non1_spatial * out_10_1 + out_8_1

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))

        non2_channel = self.non2_channel(out_6_2)
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_8_2 = out_8_2 * non2_channel + out_6_2

        non2_spatial = self.non1_spatial(out_8_2)
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_10_2 = out_10_2 * non2_spatial + out_8_2

        fuse = torch.cat((out_10_2, out_10_1), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        non3_channel = self.non3_channel(fuse)
        out_fuse_1 = out_fuse_1 * non3_channel + fuse

        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        non3_spatial = self.non3_spatial(out_fuse_1)
        out_fuse_2 = out_fuse_2 * non3_spatial + out_fuse_1
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse