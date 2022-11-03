# -*- coding:utf-8 -*-
import argparse, os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from base_net_withoutBN import BaseNet_RMCR_fuseRMCR, BaseNet_only_cross_attention_advise1, BaseNet, \
    BaseNet_RMCR_fuseRMCR_cross, BaseNet_RMCR_fuseRMCR_cross_only_corss_advise1
from CODON_x8 import CODONNet
import cv2
import numpy as np
import math
import time
import torch.utils
from ssim_2 import *
from Loger import Logger
# Testsing settings
parser = argparse.ArgumentParser(description="non-local")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda?")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="2", type=str, help="gpu ids (default: 0)")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    print("===> Building model")
    model = CODONNet() # 5 & 3
    # model = BaseNet_RMCR_fuseRMCR3_2()  # 3 & 2
    print("===> Setting GPU")
    if cuda:
        model = model.cuda().half()
    sys.stdout = Logger('./test_sintel.txt')
    for i in range(94, 95):
        # checkpoint = torch.load('/home/user/cq/ni-folder/eassay_fixed_v2/X4/X4.pth')
        checkpoint = torch.load('X8.pth')
        
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
        test(model)
        print('------------------------------------------', i)


def test(model):
    ssim_sum = 0.0
    rmse_sum = 0.0
    model.eval()

    # Middleburry
    input_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/106/CaoqiResponse/16/middlebury/X8/'
    gray_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/106/CaoqiResponse/16/middlebury/color/'
    #fix_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/106/CaoqiResponse/16/middlebury/fix/'
    label_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/106/CaoqiResponse/16/middlebury/label/'
    #test_path = '/media/fourcard/TOSHIBA EXT/CONDOM/CODON_X8/X8'


    # sintel(60)
    # input_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/sintel/Bicubic/X4'
    # gray_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/sintel/color/'
    # fix_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/sintel/label/'
    # label_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/sintel/label/'


    # tsu(60)
    # input_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/tsu/Bicubic/X4'
    # gray_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/tsu/color/'
    # fix_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/tsu/label/'
    # label_path = '/home/user/cq/ni-folder/eassay_fixed_v2/X16/X16_Sintel_Stu_data/tsu/label/'


    # new NYU(450)
    #input_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/ni-folder/eassay_fixed_v2/X16/NYU/Bicubic/X4/'
    #gray_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/ni-folder/eassay_fixed_v2/X16/NYU/color/'
    #fix_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/ni-folder/eassay_fixed_v2/X16/NYU/label/'
    #label_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/ni-folder/eassay_fixed_v2/X16/NYU/label/'
    #sellect_path = '/media/fourcard/5c64d0be-acf0-4ea0-8645-96feba23ea7e/user/cq/ni-folder/eassay_fixed_v2/compare_sintel_tsu_nyu/nyu/our_sellect/X4/'

    #sellect_dir = os.listdir(sellect_path)
    #sellect_dir.sort()
    # msg =
    input_dir = os.listdir(input_path)
    label_dir = os.listdir(label_path)
    gray_dir = os.listdir(gray_path)
    #test_dir = os.listdir(test_path)
    gray_dir.sort()
    #test_dir.sort()
    input_dir.sort()
    label_dir.sort()
    i = 0
    for gray_file in gray_dir:
        input_name = os.path.join(input_path, gray_file)
        label_name = os.path.join(label_path, gray_file)
        gray_name = os.path.join(gray_path, gray_file)
        #fix_name = os.path.join(fix_path, gray_file)
        #test_name = os.path.join(test_path, gray_file)

        input_pic = cv2.imread(input_name, 0)
        label_pic = cv2.imread(label_name, 0)
        gray_pic = cv2.imread(gray_name, 0)
        #fix_pic = cv2.imread(fix_name, 0)
        #test_pic = cv2.imread(test_name, 0)

        input_pic = torch.from_numpy(input_pic / 255).float().unsqueeze(0).unsqueeze(0).cuda().half()
        gray_pic = torch.from_numpy(gray_pic / 255).float().unsqueeze(0).unsqueeze(0).cuda().half()

        out = model(input_pic, gray_pic)
        # out = input_pic
        out = out.squeeze(0).squeeze(0)
        out = out.data.cpu().numpy()

        out = np.clip(out, 0, 1)
        torch.cuda.empty_cache()
        out = (out * 255).astype(np.uint8)
        # cv2.imwrite('/home/user/cq/ni-folder/eassay_fixed_v2/compare_sintel_tsu_nyu/nyu/X8/bicubic_450_result/' + gray_file, out)
        # cv2.imwrite('/home/user/cq/ni-folder/eassay_fixed_v2/compare_sintel_tsu_nyu/nyu/X8/SRCNN_450_result/' + gray_file, out)
        # cv2.imwrite('/home/user/cq/ni-folder/eassay_fixed_v2/compare_sintel_tsu_nyu/nyu/X8/VDSR_450_result/' + gray_file, out)
        # cv2.imwrite('/home/user/cq/ni-folder/eassay_fixed_v2/compare_sintel_tsu_nyu/nyu/X8/MSG_450_result/' + gray_file, out)
        cv2.imwrite('CODON_result_save/' + gray_file, out)
        RMSE_darkc = EvaluationResults(label_pic, out)
        ssim = ssim_exact(fix_pic / 255, out / 255)
        rmse_sum += RMSE_darkc
        ssim_sum += ssim
        print(gray_file, RMSE_darkc, ssim)
        i += 1
    print(i)
    print(rmse_sum / i, ssim_sum / i)


def EvaluationResults(depth_high, output):  # MSE, RMSE, PSNR, SSIM, MAD
    depth_high = depth_high.astype(np.float64)
    output = output.astype(np.float64)
    depth_high = depth_high[:output.shape[0], :output.shape[1]]
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
    return RMSE


if __name__ == "__main__":
    main()

