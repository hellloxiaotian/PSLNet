#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-7 20:23
# @Software: PyCharm
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DPAUNet
from utils import *
import matplotlib.image as matImage


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
config = get_config('configs/config.yaml')
parser = argparse.ArgumentParser(description="watermark removal")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--model", type=str, default=config['train_model_out_path_DPAUNet'],
                    help='load model')
parser.add_argument("--net", type=str, default="IRCNN", help='Network used in test')
parser.add_argument("--test_data", type=str, default='DPAUNet_test', help='The set of tests we created')
parser.add_argument("--test_noiseL", type=float, default=0, help='noise level used on test set')
parser.add_argument("--alpha", type=float, default=0.3, help="The opacity of the watermark")
parser.add_argument("--alphaL", type=float, default=0.3, help="The opacity of the watermark")
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--mode_wm", type=str, default="S", help='with known alpha level (S) or blind training (B)')
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--display", type=str, default="True", help='Whether to display an image')
parser.add_argument("--GPU_id", type=str, default="1", help='GPU_id')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_id

torch.manual_seed(0)

if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"
if opt.mode == "S":
    model_name_4 = "S" + str(int(opt.test_noiseL))
else:
    model_name_4 = "B"
if opt.loss == "L2":
    model_name_5 = "L2"
else:
    model_name_5 = "L1"
if opt.mode_wm == "S":
    model_name_6 = "aS"
else:
    model_name_6 = "aB"
tensorboard_name = opt.net + model_name_3 + model_name_4 + model_name_5 + model_name_6 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"


def normalize(data):
    return data / 255.


def water_test():
    print('Loading model ...\n')
    if opt.net == "HN":
        net = HN()
    elif opt.net == "SUNet":
        net = SUNet()
    elif opt.net == "HN2":
        net = HN2()
    elif opt.net == "DPUNet":
        net = DPUNet()
    elif opt.net == "PSLNet":
        net = DPAUNet()
    else:
        assert False
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.model, model_name)))
    model.eval()
    print('Loading data info ...\n')
    data_path = config['train_data_path']
    files_source = glob.glob(os.path.join(data_path, opt.test_data, '*.jpg'))
    files_source.sort()
    all_psnr_source_avg = 0
    all_ssim_source_avg = 0
    all_mse_source_avg = 0

    all_psnr_avg = 0
    all_ssim_avg = 0
    all_mse_avg = 0

    wm_id = -1
    for img_index in range(12):
        img_index += 1
        psnr_test = 0
        f_index = 0
        ssim_test = 0
        mse_test = 0
        psnr_source_avg = 0
        ssim_source_avg = 0
        mse_source_avg = 0

        wm_id += 1
        if opt.display == "True":
            print(wm_id)
            if wm_id != 9:
                continue
        for f in files_source:
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:, :, :]))
            Img = np.expand_dims(Img, 0)
            Img = np.transpose(Img, (0, 3, 1, 2))
            _, _, w, h = Img.shape
            w = int(int(w / 32) * 32)
            h = int(int(h / 32) * 32)
            Img = Img[:, :, 0:w, 0:h]
            ISource = torch.Tensor(Img)

            if opt.alphaL != 0.0:
                INoisy = add_watermark_noise_test(ISource, 0., img_id=img_index, scale_img=1.0, alpha=opt.alphaL)
            else:
                INoisy = ISource

            if (opt.test_noiseL == 0) & (opt.mode == 'S'):
                INoisy = torch.Tensor(INoisy)
            else:
                noise_gs = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
                INoisy = torch.Tensor(INoisy) + noise_gs
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            with torch.no_grad():
                if opt.net == "FFDNet":
                    noise_sigma = opt.test_noiseL / 255.
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(INoisy.shape[0])]))
                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.cuda()
                    Out = torch.clamp(model(INoisy, noise_sigma), 0., 1.)
                else:
                    Out = torch.clamp(model(INoisy)[0], 0., 1.)
                INoisy = torch.clamp(INoisy, 0., 1.)

            psnr_source = batch_PSNR(INoisy, ISource, 1.0)
            ssim_source = batch_SSIM(INoisy, ISource, 1.0)
            mse_source = batch_RMSE(INoisy, ISource, 1.0)
            psnr_api = batch_PSNR(Out, ISource, 1.)
            ssim_api = batch_SSIM(Out, ISource, 1.)
            mse_api = batch_RMSE(Out, ISource, 1.)

            if opt.display == "True":
                Out_np = Out.cpu().numpy()
                INoisy_np = INoisy.cpu().numpy()
                print("Out_np", Out_np.shape)
                pic = Out_np[0]
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                print("pic.shape", pic.shape)
                save_path1 = data_path + "/Result_DPAUNet_img/pic_out" + str(f_index) + opt.net + "_psnr_" + str(psnr_api) + "_ssim_" + str(ssim_api) + ".jpg"
                matImage.imsave(save_path1, pic)
                pic = INoisy_np[0]
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                matImage.imsave(data_path + "/Result_DPAUNet_img/pic_input" + str(f_index) +"_psnr_" + str(psnr_source)+ "_ssim_" + str(ssim_source)+".jpg", pic)
                f_index += 1

            psnr_test += psnr_api
            ssim_test += ssim_api
            mse_test += mse_api
            psnr_source_avg += psnr_source
            ssim_source_avg += ssim_source
            mse_source_avg += mse_source
        psnr_test /= len(files_source)
        ssim_test /= len(files_source)
        mse_test /= len(files_source)
        psnr_source_avg /= len(files_source)
        ssim_source_avg /= len(files_source)
        mse_source_avg /= len(files_source)

        all_psnr_avg += psnr_test
        all_mse_avg += mse_test
        all_ssim_avg += ssim_test

        all_psnr_source_avg += psnr_source_avg
        all_mse_source_avg += mse_source_avg
        all_ssim_source_avg += ssim_source_avg

    all_ssim_source_avg /= 12
    all_mse_source_avg /= 12
    all_psnr_source_avg /= 12

    all_ssim_avg /= 12
    all_mse_avg /= 12
    all_psnr_avg /= 12

    print("\nPSNR on test data %f SSIM on test data %f" % (all_psnr_avg, all_ssim_avg ))


if __name__ == "__main__":
    # main()
    water_test()
