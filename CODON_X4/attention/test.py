#-*- coding:utf-8 -*-
from __future__ import print_function
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from base_net_withoutBN import BaseNet_RMCR_fuseRMCR, BaseNet, BaseNet_RMCR_fuseRMCR_cross, BaseNet_RMCR_fuseRMCR_cross2
import cv2
import numpy as np
import math
import time
import torch._utils
from termcolor import cprint
# Testsing settings
def main():
    print_info('----------------------------------------------------------------------\n'
               '|                       M2Det Training Program                       |\n'
               '----------------------------------------------------------------------', ['yellow', 'bold'])

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def test():
    input_path = '/media/hp/Elements/科研/exp/GRRN-Net/Bicubic/X4/'
    label_path = '/media/hp/Elements/科研/exp/GRRN-Net/GTblack/'
    gray_path = '/media/hp/Elements/科研/exp/GRRN-Net/Gray/'
    input_dir = os.listdir(input_path)
    label_dir = os.listdir(label_path)
    gray_dir = os.listdir(gray_path)
    gray_dir.sort()
    input_dir.sort()
    label_dir.sort()
    psnr = []
    for input_file, label_file, gray_file in zip(input_dir, label_dir, gray_dir):
        input_name = os.path.join(input_path, input_file)
        label_name = os.path.join(label_path, label_file)
        gray_name = os.path.join(gray_path, gray_file)
        input_pic = cv2.imread(input_name, 0)
        label_pic = cv2.imread(label_name, 0)
        gray_pic = cv2.imread(gray_name, 0)

        MSE_dark, RMSE_darkc, PSNR_dark = EvaluationResults(label_pic, input_pic)
        psnr.append(PSNR_dark)
    return psnr

def EvaluationResults(depth_high, output):  #MSE, RMSE, PSNR, SSIM, MAD
    depth_high = depth_high.astype(np.float64)
    output = output.astype(np.float64)
    depth_high = depth_high[:output.shape[0],:output.shape[1]]
    mn = depth_high.size
    e = np.zeros([depth_high.shape[0], depth_high.shape[1]])

    for i in range(depth_high.shape[0]):
        for j in range(depth_high.shape[1]):
            if depth_high[i][j] == 0:
                mn = mn - 1
            else:
                e[i][j] = depth_high[i][j] - output[i][j]

    MSE = (e ** 2).sum() / mn
    RMSE = math.sqrt(MSE)
    PSNR = 10 * math.log10(255 ** 2 / MSE)
    return MSE, RMSE, PSNR


if __name__ == "__main__":
    main()

