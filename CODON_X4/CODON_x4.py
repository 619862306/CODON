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
from attention.ResCBAM import ResCBAM, ResCBAM_c, ResCBAM_d, ChannelGate, SpatialGate
from CAC_module import CAC_channel as CHANNEL
from CAC_module import CAC_spatial as SPATIAL


class CODONNet(nn.Module):
    def __init__(self):
        '''
        no non_local
        '''
        super(CODONNet, self).__init__()
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
    def forward(self, x, y):      #x深度图 y彩色图
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(y))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):  #网络一共五层MC
            out_MC_R1 = self.relu(self.conv1(out))
            out_MC_P1_c = self.relu(self.conv5(out_c))
            out_MC_P1 = self.relu(self.conv2(out))
            out_MC_R1_c = self.relu(self.conv4(out_c))
            out_MC_stage = torch.cat((out_MC_R1, out_MC_P1), 1)
            out_MC_stage_c = torch.cat((out_MC_R1_c, out_MC_P1_c), 1)
            out_MC_R2 = self.relu(self.conv3(out_MC_stage))
            out_MC_R2_c = self.relu(self.conv6(out_MC_stage_c))
            out_c = self.confuse_c(out_MC_R2_c)
            out = self.confuse(out_MC_R2)
            CAC_cat = torch.cat((out_c, out), 1)    #Fcat
            if _ == 0:
                CAC_channel = self.attention_c0(CAC_cat)
                CAC_spatial = self.attention_s0(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 1:
                CAC_channel = self.attention_c1(CAC_cat)
                CAC_spatial = self.attention_s1(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 2:
                CAC_channel = self.attention_c2(CAC_cat)
                CAC_spatial = self.attention_s2(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 3:
                CAC_channel = self.attention_c3(CAC_cat)
                CAC_spatial = self.attention_s3(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 4:
                CAC_channel = self.attention_c4(CAC_cat)
                CAC_spatial = self.attention_s4(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):      #MC是一次循环
            out_fuse_MC_R1 = self.relu(self.conv8(out_fuse))
            out_fuse_MC_P1 = self.relu(self.conv9(out_fuse))
            out_fuse_MC_stage = torch.cat((out_fuse_MC_R1, out_fuse_MC_P1), 1)
            out_fuse_MC_R2 = self.relu(self.conv10(out_fuse_MC_stage))
            out_fuse = self.confuse_fuse(out_fuse_MC_R2)
            out_fuse = torch.add(out_fuse, fuse)   #每次MC后add residual
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out_fuse_final = torch.add(out_fuse_final, residual)
        return out_fuse_final
