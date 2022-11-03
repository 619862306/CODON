from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn.functional import pad
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from attention.CBAM import CBAM
from attention.ResCBAM import ResCBAM, ResCBAM_c, ResCBAM_d, ChannelGate, SpatialGate
from wechat_guide import ChannelGate as CHANNEL
from wechat_guide import SpatialGate as SPATIAL
from attention.wechat_2 import ChannelGate as CA
from attention.wechat_2 import SpatialGate as SA



class PAM_Module(Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class CAM_Module(Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class SEPNON(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEPNON, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)
        return sasc_output

class SpatialCGNL(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        nn.init.constant_(self.z.weight, 0)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, h, w):

        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class BaseNet_non_corr(nn.Module):
    def __init__(self):
        super(BaseNet_non_corr, self).__init__()
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

        self.non1 = ResCBAM(64)
        self.non2 = ResCBAM(64)
        self.non3 = ResCBAM(64)

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_10_1 = self.non1(out_10_1)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_10_2 = self.non2(out_10_2)

        fuse = torch.cat((out_10_2, out_10_1), 1)
        fuse = self.relu(self.conv11(fuse))
        fuse = self.non3(fuse)
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse


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

        self.non1 = ResCBAM(64)
        self.non2 = ResCBAM(64)
        self.non3 = ResCBAM(64)

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_10_1 = self.non1(out_10_1)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_10_2 = self.non2(out_10_2)

        fuse = torch.cat((out_10_2, out_10_1), 1)
        fuse = self.relu(self.conv11(fuse))
        fuse = self.non3(fuse)
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse


class BaseNet_non2(nn.Module):
    def __init__(self):
        super(BaseNet_non2, self).__init__()
        self.pa = PAM_Module(64)
        self.ca = CAM_Module(64)
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

        self.non1 = ResCBAM(64)
        self.non2 = ResCBAM(64)
        self.non3 = ResCBAM(64)

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_10_1 = self.non1(out_10_1)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_10_2 = self.non2(out_10_2)

        fuse = torch.cat((out_10_2, out_10_1), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse_3 = self.non3(out_fuse_3)
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_non3(nn.Module):
    def __init__(self):
        super(BaseNet_non3, self).__init__()
        self.pa = PAM_Module(64)
        self.ca = CAM_Module(64)
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

        self.non1 = ResCBAM_d(64)
        self.non2 = ResCBAM_c(64)
        self.non3 = ResCBAM(64)

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_10_1 = self.non1(out_10_1)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_10_2 = self.non2(out_10_2)

        fuse = torch.cat((out_10_2, out_10_1), 1)
        fuse = self.relu(self.conv11(fuse))
        fuse = self.non3(fuse)
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_non_cat(nn.Module):
    def __init__(self):
        super(BaseNet_non_cat, self).__init__()
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
        self.concat_d = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
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
        self.concat_c = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.concat_fuse = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
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

        self.non1 = ResCBAM(64)
        self.non2 = ResCBAM(64)
        self.non3 = ResCBAM(64)

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_10_1_attention = self.non1(out_10_1)
        out_10_1 = self.concat_d(torch.cat((out_10_1, out_10_1_attention), 1))

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_10_2_attention = self.non2(out_10_2)
        out_10_2 = self.concat_c(torch.cat((out_10_2, out_10_2_attention), 1))

        fuse = torch.cat((out_10_2, out_10_1), 1)
        fuse = self.relu(self.conv11(fuse))
        fuse_attention = self.non3(fuse)
        fuse = self.concat_fuse(torch.cat((fuse, fuse_attention), 1))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

# class BaseNetRMCRFuseAttention(nn.Module):
#     def __init__(self):
#         '''
#         no non_local
#         '''
#         super(BaseNetRMCRFuseAttention, self).__init__()
#         self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
#         self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
# #--------------------------------------------------------------------------------------------------------------------------#
#         self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
#         self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
# #--------------------------------------------------------------------------------------------------------------------------#
#         self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
# #--------------------------------------------------------------------------------------------------------------------------#
#         self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU()
#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
#         self.attention_c0 = ChannelGate(128)
#         self.attention_c1 = ChannelGate(128)
#         self.attention_c2 = ChannelGate(128)
#         self.attention_c3 = ChannelGate(128)
#         self.attention_c4 = ChannelGate(128)
#         self.attention_s0 = SpatialGate()
#         self.attention_s1 = SpatialGate()
#         self.attention_s2 = SpatialGate()
#         self.attention_s3 = SpatialGate()
#         self.attention_s4 = SpatialGate()
#     def forward(self, x, y):
#         residual = x
#         inputs = self.relu(self.input(x))
#         inputs = self.relu(self.conv_input(inputs))
#         out = inputs
#         inputs_c = self.relu(self.input_c(y))
#         inputs_c = self.relu(self.conv_input_c(inputs_c))
#         out_c = inputs_c
#         for _ in range(5):
#             out_1 = self.relu(self.conv1(out))
#             out_2_c = self.relu(self.conv5(out_c))
#             out_2 = self.relu(self.conv2(out))
#             out_1_c = self.relu(self.conv4(out_c))
#             out_stage1 = torch.cat((out_1, out_2), 1)
#             out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
#             out_4 = self.relu(self.conv3(out_stage1))
#             out_3_c = self.relu(self.conv6(out_stage1_c))
#             feature_cat = torch.cat((out_4, out_3_c), 1)
#             if _ == 0:
#                 attention_channel = self.attention_c0(feature_cat)
#                 out_4 = out_4 * attention_channel
#                 out_3_c = out_3_c * attention_channel
#                 feature_cat = torch.cat((out_4, out_3_c), 1)
#                 attention_spatial = self.attention_s0(feature_cat)
#                 out_4 = out_4 * attention_spatial
#                 out_3_c = out_3_c * attention_spatial
#             if _ == 1:
#                 attention_channel = self.attention_c1(feature_cat)
#                 out_4 = out_4 * attention_channel
#                 out_3_c = out_3_c * attention_channel
#                 feature_cat = torch.cat((out_4, out_3_c), 1)
#                 attention_spatial = self.attention_s1(feature_cat)
#                 out_4 = out_4 * attention_spatial
#                 out_3_c = out_3_c * attention_spatial
#             if _ == 2:
#                 attention_channel = self.attention_c2(feature_cat)
#                 out_4 = out_4 * attention_channel
#                 out_3_c = out_3_c * attention_channel
#                 feature_cat = torch.cat((out_4, out_3_c), 1)
#                 attention_spatial = self.attention_s2(feature_cat)
#                 out_4 = out_4 * attention_spatial
#                 out_3_c = out_3_c * attention_spatial
#             if _ == 3:
#                 attention_channel = self.attention_c3(feature_cat)
#                 out_4 = out_4 * attention_channel
#                 out_3_c = out_3_c * attention_channel
#                 feature_cat = torch.cat((out_4, out_3_c), 1)
#                 attention_spatial = self.attention_s3(feature_cat)
#                 out_4 = out_4 * attention_spatial
#                 out_3_c = out_3_c * attention_spatial
#             if _ == 4:
#                 attention_channel = self.attention_c4(feature_cat)
#                 out_4 = out_4 * attention_channel
#                 out_3_c = out_3_c * attention_channel
#                 feature_cat = torch.cat((out_4, out_3_c), 1)
#                 attention_spatial = self.attention_s4(feature_cat)
#                 out_4 = out_4 * attention_spatial
#                 out_3_c = out_3_c * attention_spatial
#
#             out = self.confuse(out_4)
#             out_c = self.confuse_c(out_3_c)
#             out = torch.add(out, inputs)
#             out_c = torch.add(out_c, inputs_c)
#
#         fuse = torch.cat((out, out_c), 1)
#         fuse = self.relu(self.conv7(fuse))
#         out_fuse = fuse
#         for _ in range(3):
#             out_fuse = self.relu(self.conv9(self.relu(self.conv8(out_fuse))))
#             out_fuse = torch.add(out_fuse, fuse)
#
#         out = self.relu(self.conv10(out_fuse))
#         out_fuse = self.output(out)
#         out_fuse = torch.add(out_fuse, residual)
#         return out_fuse

class BaseNet_RMCR(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2 = self.relu(self.conv2(out))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out = self.confuse(out_4)
            out = torch.add(out, inputs)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1_c = self.relu(self.conv4(out_c))
            out_2_c = self.relu(self.conv5(out_c))
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out_c = torch.add(out_c, inputs_c)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse = self.relu(self.conv9(self.relu(self.conv8(out_fuse))))
            out_fuse = torch.add(out_fuse, fuse)

        out = self.relu(self.conv10(out_fuse))
        out_fuse = self.output(out)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_RMCR_NLAR(nn.Module):
    def __init__(self):
        super(BaseNet_RMCR_NLAR, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        self.non1 = SpatialCGNL(64, 32, use_scale=False, groups=8)
        self.non2 = SpatialCGNL(64, 32, use_scale=False, groups=8)
        self.non3 = SpatialCGNL(64, 32, use_scale=False, groups=8)

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2 = self.relu(self.conv2(out))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out = self.confuse(out_4)
            out = torch.add(out, inputs)
        out_non_d = self.non1(out)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1_c = self.relu(self.conv4(out_c))
            out_2_c = self.relu(self.conv5(out_c))
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out_c = torch.add(out_c, inputs_c)
        out_non_c = self.non2(out_c)

        fuse = torch.cat((out_non_d, out_non_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse = self.relu(self.conv9(self.relu(self.conv8(out_fuse))))
            out_fuse = torch.add(out_fuse, fuse)

        out_non = self.non3(out_fuse)
        out = self.relu(self.conv10(out_non))
        out = self.output(out)
        out = torch.add(out, residual)
        return out

# class NonLocalBlock2D(nn.Module):
#     '''
#     non_local_block
#     '''
#     def __init__(self, inplanes, planes, use_scale=True):
#         self.use_scale = use_scale
#         super(NonLocalBlock2D, self).__init__()
#         # conv theta
#         self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
#         # conv phi
#         self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
#         # conv g
#         self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
#         # conv z
#         self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
#         # nn.init.constant_(self.z.weight, 0)
#         self.relu = nn.ReLU(inplace=True)
#     def kernel(self, t, p, g, b, c, h, w):
#         t = t.view(b, 1, c * h * w)
#         # print('t', t.shape)
#         p = p.view(b, 1, c * h * w)
#         # print('p', p.shape)
#         g = g.view(b, c * h * w, 1)
#         # print('g', g.shape)
#         att = torch.bmm(p, g)
#         # print('att', att.shape)
#         if self.use_scale:
#             att = att.div((c*h*w)**0.5)
# 
#         x = torch.bmm(att, t)
#         # print('x_kernel', x.shape)
#         x = x.view(b, c, h, w)
# 
#         return x
# 
# 
#     def forward(self, x):
#         residual = x
# 
#         t = self.t(x)
#         # print(t.shape)
#         p = self.p(x)
#         # print(p.shape)
#         g = self.g(x)
#         # print(g.shape)
# 
#         b, c, h, w = t.size()
#         x = self.kernel(t, p, g, b, c, h, w)
#         # print(x.shape)
#         x = self.z(x)
#         x = x + residual
#         return x

class NonLocalBlock2D_BN(nn.Module):
    '''
    non_local_block
    '''
    def __init__(self, inplanes, planes, use_scale=True):
        self.use_scale = use_scale
        super(NonLocalBlock2D_BN, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(inplanes)
        nn.init.constant_(self.z.weight, 0)
        self.relu = nn.ReLU(inplace=True)
    def kernel(self, t, p, g, b, c, h, w):
        t = t.view(b, 1, c * h * w)
        # print('t', t.shape)
        p = p.view(b, 1, c * h * w)
        # print('p', p.shape)
        g = g.view(b, c * h * w, 1)
        # print('g', g.shape)
        att = torch.bmm(p, g)
        # print('att', att.shape)
        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        # print('x_kernel', x.shape)
        x = x.view(b, c, h, w)

        return x


    def forward(self, x):
        residual = x

        t = self.t(x)
        # print(t.shape)
        p = self.p(x)
        # print(p.shape)
        g = self.g(x)
        # print(g.shape)

        b, c, h, w = t.size()
        x = self.kernel(t, p, g, b, c, h, w)
        # print(x.shape)
        x = self.z(x)
        x = self.bn4(x) + residual
        return x

#double_normal_relu // double_normal_relu_non// base0_reluback// base0_non_reluback

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))

        fuse = torch.cat((out_10_1, out_10_2), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_only_fuse_attention(nn.Module):
    def __init__(self):
        super(BaseNet_only_fuse_attention, self).__init__()
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
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))

        fuse = torch.cat((out_10_1, out_10_2), 1)
        fuse = self.relu(self.conv11(fuse))
        residule_fuse = fuse
        attention_channel = self.attention_c5(fuse)
        fuse = fuse * attention_channel
        attention_spatial = self.attention_s5(fuse)
        fuse = fuse * attention_spatial + residule_fuse

        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse



class BaseNet_Cross(nn.Module):
    def __init__(self):
        super(BaseNet_Cross, self).__init__()
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
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))

        out_1_1 = self.relu(self.conv1_1(inputs))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_1 = self.relu(self.conv2_1(out_1_1))
        out_2_2 = self.relu(self.conv2_2(out_1_2))

        residule_1 = out_2_1
        residule_2 = out_2_2
        attention1 = torch.cat((out_2_1, out_2_2), 1)
        attention_channel_1 = self.attention_c0(attention1)
        out_2_1 = out_2_1 * attention_channel_1
        out_2_2 = out_2_2 * attention_channel_1
        attention2 = torch.cat((out_2_1, out_2_2), 1)
        attention_spatial_1 = self.attention_s0(attention2)
        out_2_1 = out_2_1 * attention_spatial_1 + residule_1
        out_2_2 = out_2_2 * attention_spatial_1 + residule_2

        out_3_1 = self.relu(self.conv3_1(out_2_1))
        out_4_1 = self.relu(self.conv4_1(out_3_1))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        residule_1 = out_4_1
        residule_2 = out_4_2
        attention1 = torch.cat((out_4_1, out_4_2), 1)
        attention_channel_1 = self.attention_c1(attention1)
        out_4_1 = out_4_1 * attention_channel_1
        out_4_2 = out_4_2 * attention_channel_1
        attention2 = torch.cat((out_4_1, out_4_2), 1)
        attention_spatial_1 = self.attention_s1(attention2)
        out_4_1 = out_4_1 * attention_spatial_1 + residule_1
        out_4_2 = out_4_2 * attention_spatial_1 + residule_2

        out_5_1 = self.relu(self.conv5_1(out_4_1))
        out_6_1 = self.relu(self.conv6_1(out_5_1))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        residule_1 = out_6_1
        residule_2 = out_6_2
        attention1 = torch.cat((out_6_1, out_6_2), 1)
        attention_channel_1 = self.attention_c2(attention1)
        out_6_1 = out_6_1 * attention_channel_1
        out_6_2 = out_6_2 * attention_channel_1
        attention2 = torch.cat((out_6_1, out_6_2), 1)
        attention_spatial_1 = self.attention_s2(attention2)
        out_6_1 = out_6_1 * attention_spatial_1 + residule_1
        out_6_2 = out_6_2 * attention_spatial_1 + residule_2

        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        residule_1 = out_8_1
        residule_2 = out_8_2
        attention1 = torch.cat((out_8_1, out_8_2), 1)
        attention_channel_1 = self.attention_c3(attention1)
        out_8_1 = out_8_1 * attention_channel_1
        out_8_2 = out_8_2 * attention_channel_1
        attention2 = torch.cat((out_8_1, out_8_2), 1)
        attention_spatial_1 = self.attention_s3(attention2)
        out_8_1 = out_8_1 * attention_spatial_1 + residule_1
        out_8_2 = out_8_2 * attention_spatial_1 + residule_2


        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        residule_1 = out_10_1
        residule_2 = out_10_2
        attention1 = torch.cat((out_10_1, out_10_2), 1)
        attention_channel_1 = self.attention_c4(attention1)
        out_10_1 = out_10_1 * attention_channel_1
        out_10_2 = out_10_2 * attention_channel_1
        attention2 = torch.cat((out_10_1, out_10_2), 1)
        attention_spatial_1 = self.attention_s4(attention2)
        out_10_1 = out_10_1 * attention_spatial_1 + residule_1
        out_10_2 = out_10_2 * attention_spatial_1 + residule_2

        fuse = torch.cat((out_10_1, out_10_2), 1)
        fuse = self.relu(self.conv11(fuse))
        residule_fuse = fuse
        attention_channel = self.attention_c5(fuse)
        fuse = fuse * attention_channel
        attention_spatial = self.attention_s5(fuse)
        fuse = fuse * attention_spatial + residule_fuse

        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_only_cross_attention(nn.Module):
    def __init__(self):
        super(BaseNet_only_cross_attention, self).__init__()
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
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))

        out_1_1 = self.relu(self.conv1_1(inputs))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_1 = self.relu(self.conv2_1(out_1_1))
        out_2_2 = self.relu(self.conv2_2(out_1_2))

        residule_1 = out_2_1
        residule_2 = out_2_2
        attention1 = torch.cat((out_2_1, out_2_2), 1)
        attention_channel_1 = self.attention_c0(attention1)
        out_2_1 = out_2_1 * attention_channel_1
        out_2_2 = out_2_2 * attention_channel_1
        attention2 = torch.cat((out_2_1, out_2_2), 1)
        attention_spatial_1 = self.attention_s0(attention2)
        out_2_1 = out_2_1 * attention_spatial_1 + residule_1
        out_2_2 = out_2_2 * attention_spatial_1 + residule_2

        out_3_1 = self.relu(self.conv3_1(out_2_1))
        out_4_1 = self.relu(self.conv4_1(out_3_1))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        residule_1 = out_4_1
        residule_2 = out_4_2
        attention1 = torch.cat((out_4_1, out_4_2), 1)
        attention_channel_1 = self.attention_c1(attention1)
        out_4_1 = out_4_1 * attention_channel_1
        out_4_2 = out_4_2 * attention_channel_1
        attention2 = torch.cat((out_4_1, out_4_2), 1)
        attention_spatial_1 = self.attention_s1(attention2)
        out_4_1 = out_4_1 * attention_spatial_1 + residule_1
        out_4_2 = out_4_2 * attention_spatial_1 + residule_2

        out_5_1 = self.relu(self.conv5_1(out_4_1))
        out_6_1 = self.relu(self.conv6_1(out_5_1))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        residule_1 = out_6_1
        residule_2 = out_6_2
        attention1 = torch.cat((out_6_1, out_6_2), 1)
        attention_channel_1 = self.attention_c2(attention1)
        out_6_1 = out_6_1 * attention_channel_1
        out_6_2 = out_6_2 * attention_channel_1
        attention2 = torch.cat((out_6_1, out_6_2), 1)
        attention_spatial_1 = self.attention_s2(attention2)
        out_6_1 = out_6_1 * attention_spatial_1 + residule_1
        out_6_2 = out_6_2 * attention_spatial_1 + residule_2

        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        residule_1 = out_8_1
        residule_2 = out_8_2
        attention1 = torch.cat((out_8_1, out_8_2), 1)
        attention_channel_1 = self.attention_c3(attention1)
        out_8_1 = out_8_1 * attention_channel_1
        out_8_2 = out_8_2 * attention_channel_1
        attention2 = torch.cat((out_8_1, out_8_2), 1)
        attention_spatial_1 = self.attention_s3(attention2)
        out_8_1 = out_8_1 * attention_spatial_1 + residule_1
        out_8_2 = out_8_2 * attention_spatial_1 + residule_2


        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        residule_1 = out_10_1
        residule_2 = out_10_2
        attention1 = torch.cat((out_10_1, out_10_2), 1)
        attention_channel_1 = self.attention_c4(attention1)
        out_10_1 = out_10_1 * attention_channel_1
        out_10_2 = out_10_2 * attention_channel_1
        attention2 = torch.cat((out_10_1, out_10_2), 1)
        attention_spatial_1 = self.attention_s4(attention2)
        out_10_1 = out_10_1 * attention_spatial_1 + residule_1
        out_10_2 = out_10_2 * attention_spatial_1 + residule_2

        fuse = torch.cat((out_10_1, out_10_2), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_only_cross_attention_advise1_nores(nn.Module):
    def __init__(self):
        super(BaseNet_only_cross_attention_advise1_nores, self).__init__()
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
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))

        out_1_1 = self.relu(self.conv1_1(inputs))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_1 = self.relu(self.conv2_1(out_1_1))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        attention1 = torch.cat((out_2_1, out_2_2), 1)
        attention_channel_1 = self.attention_c0(attention1)
        attention_spatial_1 = self.attention_s0(attention1)
        fuse_attention1 = attention_channel_1 * attention_spatial_1
        out_2_1 = out_2_1 * fuse_attention1
        out_2_2 = out_2_2 * fuse_attention1

        out_3_1 = self.relu(self.conv3_1(out_2_1))
        out_4_1 = self.relu(self.conv4_1(out_3_1))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        # residule_1 = out_4_1
        # residule_2 = out_4_2
        attention1 = torch.cat((out_4_1, out_4_2), 1)
        attention_channel_1 = self.attention_c1(attention1)
        attention_spatial_1 = self.attention_s1(attention1)
        fuse_attention2 = attention_channel_1 * attention_spatial_1
        out_4_1 = out_4_1 * fuse_attention2
        out_4_2 = out_4_2 * fuse_attention2

        out_5_1 = self.relu(self.conv5_1(out_4_1))
        out_6_1 = self.relu(self.conv6_1(out_5_1))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        # residule_1 = out_6_1
        # residule_2 = out_6_2
        attention1 = torch.cat((out_6_1, out_6_2), 1)
        attention_channel_1 = self.attention_c2(attention1)
        attention_spatial_1 = self.attention_s2(attention1)
        fuse_attention3 = attention_channel_1 * attention_spatial_1
        out_6_1 = out_6_1 * fuse_attention3
        out_6_2 = out_6_2 * fuse_attention3

        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        # residule_1 = out_8_1
        # residule_2 = out_8_2
        attention1 = torch.cat((out_8_1, out_8_2), 1)
        attention_channel_1 = self.attention_c3(attention1)
        attention_spatial_1 = self.attention_s3(attention1)
        fuse_attention4 = attention_channel_1 * attention_spatial_1
        out_8_1 = out_8_1 * fuse_attention4
        out_8_2 = out_8_2 * fuse_attention4


        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        # residule_1 = out_10_1
        # residule_2 = out_10_2
        attention1 = torch.cat((out_10_1, out_10_2), 1)
        attention_channel_1 = self.attention_c4(attention1)
        attention_spatial_1 = self.attention_s4(attention1)
        fuse_attention5 = attention_channel_1 * attention_spatial_1
        out_10_1 = out_10_1 * fuse_attention5
        out_10_2 = out_10_2 * fuse_attention5

        fuse = torch.cat((out_10_1, out_10_2), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_only_cross_attention_advise1(nn.Module):
    def __init__(self):
        super(BaseNet_only_cross_attention_advise1, self).__init__()
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
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))

        out_1_1 = self.relu(self.conv1_1(inputs))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_1 = self.relu(self.conv2_1(out_1_1))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        residule_1 = out_2_1
        residule_2 = out_2_2
        attention1 = torch.cat((out_2_1, out_2_2), 1)
        attention_channel_1 = self.attention_c0(attention1)
        attention_spatial_1 = self.attention_s0(attention1)
        fuse_attention1 = attention_channel_1 * attention_spatial_1
        out_2_1 = out_2_1 * fuse_attention1 + residule_1
        out_2_2 = out_2_2 * fuse_attention1 + residule_2

        out_3_1 = self.relu(self.conv3_1(out_2_1))
        out_4_1 = self.relu(self.conv4_1(out_3_1))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        residule_1 = out_4_1
        residule_2 = out_4_2
        attention1 = torch.cat((out_4_1, out_4_2), 1)
        attention_channel_1 = self.attention_c1(attention1)
        attention_spatial_1 = self.attention_s1(attention1)
        fuse_attention2 = attention_channel_1 * attention_spatial_1
        out_4_1 = out_4_1 * fuse_attention2 + residule_1
        out_4_2 = out_4_2 * fuse_attention2 + residule_2

        out_5_1 = self.relu(self.conv5_1(out_4_1))
        out_6_1 = self.relu(self.conv6_1(out_5_1))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        residule_1 = out_6_1
        residule_2 = out_6_2
        attention1 = torch.cat((out_6_1, out_6_2), 1)
        attention_channel_1 = self.attention_c2(attention1)
        attention_spatial_1 = self.attention_s2(attention1)
        fuse_attention3 = attention_channel_1 * attention_spatial_1
        out_6_1 = out_6_1 * fuse_attention3 + residule_1
        out_6_2 = out_6_2 * fuse_attention3 + residule_2

        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        residule_1 = out_8_1
        residule_2 = out_8_2
        attention1 = torch.cat((out_8_1, out_8_2), 1)
        attention_channel_1 = self.attention_c3(attention1)
        attention_spatial_1 = self.attention_s3(attention1)
        fuse_attention4 = attention_channel_1 * attention_spatial_1
        out_8_1 = out_8_1 * fuse_attention4 + residule_1
        out_8_2 = out_8_2 * fuse_attention4 + residule_2


        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        residule_1 = out_10_1
        residule_2 = out_10_2
        attention1 = torch.cat((out_10_1, out_10_2), 1)
        attention_channel_1 = self.attention_c4(attention1)
        attention_spatial_1 = self.attention_s4(attention1)
        fuse_attention5 = attention_channel_1 * attention_spatial_1
        out_10_1 = out_10_1 * fuse_attention5 + residule_1
        out_10_2 = out_10_2 * fuse_attention5 + residule_2

        fuse = torch.cat((out_10_1, out_10_2), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse = self.relu(self.conv18(out_fuse_3))
        out_fuse = self.output(out_fuse)
        out_fuse = torch.add(out_fuse, residual)
        return out_fuse

class BaseNet_NLAR(nn.Module):
    def __init__(self):
        super(BaseNet_NLAR, self).__init__()
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        self.non1 = SpatialCGNL(64, 32, use_scale=False, groups=8)
        self.non2 = SpatialCGNL(64, 32, use_scale=False, groups=8)
        self.non3 = SpatialCGNL(64, 32, use_scale=False, groups=8)

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
        out_7_1 = self.relu(self.conv7_1(out_6_1))
        out_8_1 = self.relu(self.conv8_1(out_7_1))
        out_9_1 = self.relu(self.conv9_1(out_8_1))
        out_10_1 = self.relu(self.conv10_1(out_9_1))
        out_non_d = self.non1(out_10_1)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_inputc(inputs_c))
        out_1_2 = self.relu(self.conv1_2(inputs_c))
        out_2_2 = self.relu(self.conv2_2(out_1_2))
        out_3_2 = self.relu(self.conv3_2(out_2_2))
        out_4_2 = self.relu(self.conv4_2(out_3_2))
        out_5_2 = self.relu(self.conv5_2(out_4_2))
        out_6_2 = self.relu(self.conv6_2(out_5_2))
        out_7_2 = self.relu(self.conv7_2(out_6_2))
        out_8_2 = self.relu(self.conv8_2(out_7_2))
        out_9_2 = self.relu(self.conv9_2(out_8_2))
        out_10_2 = self.relu(self.conv10_2(out_9_2))
        out_non_c = self.non2(out_10_2)

        fuse = torch.cat((out_non_d, out_non_c), 1)
        fuse = self.relu(self.conv11(fuse))
        out_fuse_1 = self.relu(self.conv13(self.relu(self.conv12(fuse))))
        out_fuse_2 = self.relu(self.conv15(self.relu(self.conv14(out_fuse_1))))
        out_fuse_3 = self.relu(self.conv17(self.relu(self.conv16(out_fuse_2))))
        out_fuse_non = self.non3(out_fuse_3)
        out = self.relu(self.conv18(out_fuse_non))
        out = self.output(out)
        out = torch.add(out, residual)
        return out

class BaseNet_RMCR_fuseRMCR(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2 = self.relu(self.conv2(out))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out = self.confuse(out_4)
            out = torch.add(out, inputs)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1_c = self.relu(self.conv4(out_c))
            out_2_c = self.relu(self.conv5(out_c))
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out_c = torch.add(out_c, inputs_c)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)

        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final


class BaseNet_RMCR_fuseRMCR_2(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_2, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2 = self.relu(self.conv2(out))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out = self.confuse(out_4)
            out = torch.add(out, inputs)

        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1_c = self.relu(self.conv4(out_c))
            out_2_c = self.relu(self.conv5(out_c))
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out_c = torch.add(out_c, inputs_c)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)

        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final


class BaseNet_RMCR_fuseRMCR_cross_advise2(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross_advise2, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_channel = self.attention_c0(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s0(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
                fuse_attention = attention_channel * attention_spatial
                out_c = out_c * fuse_attention
                out = out * fuse_attention
            if _ == 1:
                attention_channel = self.attention_c1(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s1(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
                fuse_attention = attention_channel * attention_spatial
                out_c = out_c * fuse_attention
                out = out * fuse_attention

            if _ == 2:
                attention_channel = self.attention_c2(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s2(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
                fuse_attention = attention_channel * attention_spatial
                out_c = out_c * fuse_attention
                out = out * fuse_attention

            if _ == 3:
                attention_channel = self.attention_c3(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s3(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
                fuse_attention = attention_channel * attention_spatial
                out_c = out_c * fuse_attention
                out = out * fuse_attention

            if _ == 4:
                attention_channel = self.attention_c4(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s4(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
                fuse_attention = attention_channel * attention_spatial
                out_c = out_c * fuse_attention
                out = out * fuse_attention

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_channel = self.attention_c0(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s0(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 1:
                attention_channel = self.attention_c1(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s1(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 2:
                attention_channel = self.attention_c2(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s2(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 3:
                attention_channel = self.attention_c3(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s3(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 4:
                attention_channel = self.attention_c4(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s4(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        residule_fuse = fuse
        attention_c_fuse = self.attention_c5(fuse)
        fuse = fuse * attention_c_fuse
        attention_s_fuse = self.attention_s5(fuse)
        fuse = fuse * attention_s_fuse + residule_fuse
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_channel = self.attention_c0(attention_cat)
                attention_spatial = self.attention_s0(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 1:
                attention_channel = self.attention_c1(attention_cat)
                attention_spatial = self.attention_s1(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 2:
                attention_channel = self.attention_c2(attention_cat)
                attention_spatial = self.attention_s2(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 3:
                attention_channel = self.attention_c3(attention_cat)
                attention_spatial = self.attention_s3(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 4:
                attention_channel = self.attention_c4(attention_cat)
                attention_spatial = self.attention_s4(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1_parall(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1_parall, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()

        self.attention_c01 = CHANNEL(128)
        self.attention_c11 = CHANNEL(128)
        self.attention_c21 = CHANNEL(128)
        self.attention_c31 = CHANNEL(128)
        self.attention_c41 = CHANNEL(128)
        self.attention_s01 = SPATIAL()
        self.attention_s11 = SPATIAL()
        self.attention_s21 = SPATIAL()
        self.attention_s31 = SPATIAL()
        self.attention_s41 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_channel = self.attention_c0(attention_cat)
                attention_spatial = self.attention_s0(attention_cat)

                attention_channel1 = self.attention_c01(attention_cat)
                attention_spatial1 = self.attention_s01(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                ad_1attention1 = attention_channel1 * attention_spatial1
                out = out * ad_1attention
                out_c = out_c * ad_1attention1
            if _ == 1:
                attention_channel = self.attention_c1(attention_cat)
                attention_spatial = self.attention_s1(attention_cat)

                attention_channel1 = self.attention_c11(attention_cat)
                attention_spatial1 = self.attention_s11(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                ad_1attention1 = attention_channel1 * attention_spatial1
                out = out * ad_1attention
                out_c = out_c * ad_1attention1
            if _ == 2:
                attention_channel = self.attention_c2(attention_cat)
                attention_spatial = self.attention_s2(attention_cat)

                attention_channel1 = self.attention_c21(attention_cat)
                attention_spatial1 = self.attention_s21(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                ad_1attention1 = attention_channel1 * attention_spatial1
                out = out * ad_1attention
                out_c = out_c * ad_1attention1
            if _ == 3:
                attention_channel = self.attention_c3(attention_cat)
                attention_spatial = self.attention_s3(attention_cat)

                attention_channel1 = self.attention_c31(attention_cat)
                attention_spatial1 = self.attention_s31(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                ad_1attention1 = attention_channel1 * attention_spatial1
                out = out * ad_1attention
                out_c = out_c * ad_1attention1
            if _ == 4:
                attention_channel = self.attention_c4(attention_cat)
                attention_spatial = self.attention_s4(attention_cat)

                attention_channel1 = self.attention_c41(attention_cat)
                attention_spatial1 = self.attention_s41(attention_cat)
                ad_1attention = attention_channel * attention_spatial
                ad_1attention1 = attention_channel1 * attention_spatial1
                out = out * ad_1attention
                out_c = out_c * ad_1attention1

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1_onlys(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1_onlys, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_spatial = self.attention_s0(attention_cat)
                ad_1attention =  attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 1:
                attention_spatial = self.attention_s1(attention_cat)
                ad_1attention = attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 2:
                attention_spatial = self.attention_s2(attention_cat)
                ad_1attention =  attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 3:
                attention_spatial = self.attention_s3(attention_cat)
                ad_1attention =  attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 4:
                attention_spatial = self.attention_s4(attention_cat)
                ad_1attention =  attention_spatial
                out = out * ad_1attention
                out_c = out_c * ad_1attention

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1_onlyc(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1_onlyc, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_channel = self.attention_c0(attention_cat)
                ad_1attention = attention_channel
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 1:
                attention_channel = self.attention_c1(attention_cat)
                ad_1attention = attention_channel
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 2:
                attention_channel = self.attention_c2(attention_cat)
                ad_1attention = attention_channel
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 3:
                attention_channel = self.attention_c3(attention_cat)
                ad_1attention = attention_channel
                out = out * ad_1attention
                out_c = out_c * ad_1attention
            if _ == 4:
                attention_channel = self.attention_c4(attention_cat)
                ad_1attention = attention_channel
                out = out * ad_1attention
                out_c = out_c * ad_1attention

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_ECCV(nn.Module):
    def __init__(self):
        super(BaseNet_RMCR_fuseRMCR_ECCV, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CBAM(64)
        self.attention_c1 = CBAM(64)
        self.attention_c2 = CBAM(64)
        self.attention_c3 = CBAM(64)
        self.attention_c4 = CBAM(64)
        self.attention_d0 = CBAM(64)
        self.attention_d1 = CBAM(64)
        self.attention_d2 = CBAM(64)
        self.attention_d3 = CBAM(64)
        self.attention_d4 = CBAM(64)

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            if _ == 0:
                attention_c = self.attention_c0(out_c)
                attention_d = self.attention_d0(out)
                out = out * attention_d
                out_c = out_c * attention_c
            if _ == 1:
                attention_c = self.attention_c1(out_c)
                attention_d = self.attention_d1(out)
                out = out * attention_d
                out_c = out_c * attention_c
            if _ == 2:
                attention_c = self.attention_c2(out_c)
                attention_d = self.attention_d2(out)
                out = out * attention_d
                out_c = out_c * attention_c
            if _ == 3:
                attention_c = self.attention_c3(out_c)
                attention_d = self.attention_d3(out)
                out = out * attention_d
                out_c = out_c * attention_c
            if _ == 4:
                attention_c = self.attention_c4(out_c)
                attention_d = self.attention_d4(out)
                out = out * attention_d
                out_c = out_c * attention_c

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_RCAN(nn.Module):
    def __init__(self):
        super(BaseNet_RMCR_fuseRMCR_RCAN, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CALayer(64)
        self.attention_c1 = CALayer(64)
        self.attention_c2 = CALayer(64)
        self.attention_c3 = CALayer(64)
        self.attention_c4 = CALayer(64)
        self.attention_d0 = CALayer(64)
        self.attention_d1 = CALayer(64)
        self.attention_d2 = CALayer(64)
        self.attention_d3 = CALayer(64)
        self.attention_d4 = CALayer(64)

    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            if _ == 0:
                out = self.attention_c0(out_c)
                out_c = self.attention_d0(out)
            if _ == 1:
                out = self.attention_c1(out_c)
                out_c = self.attention_d1(out)
            if _ == 2:
                out = self.attention_c2(out_c)
                out_c = self.attention_d2(out)
            if _ == 3:
                out = self.attention_c3(out_c)
                out_c = self.attention_d3(out)
            if _ == 4:
                out = self.attention_c3(out_c)
                out_c = self.attention_d3(out)

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross_only_corss(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross_only_corss, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            attention_cat = torch.cat((out_c, out), 1)
            if _ == 0:
                attention_channel = self.attention_c0(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s0(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 1:
                attention_channel = self.attention_c1(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s1(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 2:
                attention_channel = self.attention_c2(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s2(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 3:
                attention_channel = self.attention_c3(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s3(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 4:
                attention_channel = self.attention_c4(attention_cat)
                out_c = out_c * attention_channel
                out = out * attention_channel
                feature_cat = torch.cat((out_c, out), 1)
                attention_spatial = self.attention_s4(feature_cat)
                out_c = out_c * attention_spatial
                out = out * attention_spatial

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        # residule_fuse = fuse
        # attention_c_fuse = self.attention_c5(fuse)
        # fuse = fuse * attention_c_fuse
        # attention_s_fuse = self.attention_s5(fuse)
        # fuse = fuse * attention_s_fuse + residule_fuse
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final

class BaseNet_RMCR_fuseRMCR_cross2(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross2, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CA(64)
        self.attention_c1 = CA(64)
        self.attention_c2 = CA(64)
        self.attention_c3 = CA(64)
        self.attention_c4 = CA(64)
        self.attention_s0 = SA()
        self.attention_s1 = SA()
        self.attention_s2 = SA()
        self.attention_s3 = SA()
        self.attention_s4 = SA()
        self.attention_c5 = CA(64)
        self.attention_s5 = SA()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            if _ == 0:
                attention_channel = self.attention_c0(out)
                out_c = out_c * attention_channel
                out = out * attention_channel
                attention_spatial = self.attention_s0(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 1:
                attention_channel = self.attention_c1(out)
                out_c = out_c * attention_channel
                out = out * attention_channel
                attention_spatial = self.attention_s1(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 2:
                attention_channel = self.attention_c2(out)
                out_c = out_c * attention_channel
                out = out * attention_channel
                attention_spatial = self.attention_s2(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 3:
                attention_channel = self.attention_c3(out)
                out_c = out_c * attention_channel
                out = out * attention_channel
                attention_spatial = self.attention_s3(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 4:
                attention_channel = self.attention_c4(out)
                out_c = out_c * attention_channel
                out = out * attention_channel
                attention_spatial = self.attention_s4(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        residule_fuse = fuse
        attention_c_fuse = self.attention_c5(fuse)
        fuse = fuse * attention_c_fuse
        attention_s_fuse = self.attention_s5(fuse)
        fuse = fuse * attention_s_fuse + residule_fuse
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final


class BaseNet_RMCR_fuseRMCR_cross3(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(BaseNet_RMCR_fuseRMCR_cross3, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#--------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.attention_c0 = CA(64)
        self.attention_c1 = CA(64)
        self.attention_c2 = CA(64)
        self.attention_c3 = CA(64)
        self.attention_c4 = CA(64)

        self.attention_c0_c = CA(64)
        self.attention_c1_c = CA(64)
        self.attention_c2_c = CA(64)
        self.attention_c3_c = CA(64)
        self.attention_c4_c = CA(64)

        self.attention_s0 = SA()
        self.attention_s1 = SA()
        self.attention_s2 = SA()
        self.attention_s3 = SA()
        self.attention_s4 = SA()
        self.attention_c5 = CA(64)
        self.attention_s5 = SA()
    def forward(self, x, y):
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):
            out_1 = self.relu(self.conv1(out))
            out_2_c = self.relu(self.conv5(out_c))
            out_2 = self.relu(self.conv2(out))
            out_1_c = self.relu(self.conv4(out_c))
            out_stage1 = torch.cat((out_1, out_2), 1)
            out_stage1_c = torch.cat((out_1_c, out_2_c), 1)
            out_4 = self.relu(self.conv3(out_stage1))
            out_3_c = self.relu(self.conv6(out_stage1_c))
            out_c = self.confuse_c(out_3_c)
            out = self.confuse(out_4)
            if _ == 0:
                attention_channel = self.attention_c0(out)
                out_c = out_c * self.attention_c0_c(out_c)
                out = out * attention_channel
                attention_spatial = self.attention_s0(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 1:
                attention_channel = self.attention_c1(out)
                out_c = out_c * self.attention_c1_c(out_c)
                out = out * attention_channel
                attention_spatial = self.attention_s1(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 2:
                attention_channel = self.attention_c2(out)
                out_c = out_c * self.attention_c2_c(out_c)
                out = out * attention_channel
                attention_spatial = self.attention_s2(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 3:
                attention_channel = self.attention_c3(out)
                out_c = out_c * self.attention_c3_c(out_c)
                out = out * attention_channel
                attention_spatial = self.attention_s3(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            if _ == 4:
                attention_channel = self.attention_c4(out)
                out_c = out_c * self.attention_c4_c(out_c)
                out = out * attention_channel
                attention_spatial = self.attention_s4(out)
                out_c = out_c * attention_spatial
                out = out * attention_spatial
            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)

        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        residule_fuse = fuse
        attention_c_fuse = self.attention_c5(fuse)
        fuse = fuse * attention_c_fuse
        attention_s_fuse = self.attention_s5(fuse)
        fuse = fuse * attention_s_fuse + residule_fuse
        out_fuse = fuse
        for _ in range(3):
            out_fuse_1 = self.relu(self.conv8(out_fuse))
            out_fuse_2 = self.relu(self.conv9(out_fuse))
            out_fuse_stage = torch.cat((out_fuse_1, out_fuse_2), 1)
            out_fuse_3 = self.relu(self.conv10(out_fuse_stage))
            out_fuse = self.confuse_fuse(out_fuse_3)
            out_fuse = torch.add(out_fuse, fuse)
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)

        return out_fuse_final
