from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
import glob

from utils import is_image_file, load_img, save_img, save_img_fusion,save_img_fusion_real,save_img_fusion_old,save_img_fusion_weight,save_img_fusion_weight3

import datetime
# 年-月-日 时:分:秒
from PIL import Image
import SimpleITK as sitk

from torch.utils.data import DataLoader
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=600, help='saved model of which epochs')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
device = torch.device("cuda:0" if opt.cuda else "cpu")

checkpoint_path="Log/StudentSwinGancheckpoint10-10_22-08"
model_path = "{}/{}/netG_model_epoch_{}.pth".format(checkpoint_path,opt.dataset, opt.nepochs)
#print(torch.load(model_path))
net_g = torch.load(model_path).to(device)

total_num = sum(p.numel() for p in net_g.parameters())
trainable_num = sum(p.numel() for p in net_g.parameters() if p.requires_grad)
print( 'Total',total_num,  trainable_num)

)


from data_isles_2015 import get_training_set, get_test_set
test_set = get_test_set("" + opt.dataset, opt.direction)

testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

transform=None


nowTime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# result_path="fusionResult/fusionResult{}".format(nowTime)
result_path="fusionResult/{}".format(nowTime)

t=test_set, num_workers=4, batch_size=1, shuffle=False)
import numpy as np

import scipy.ndimage as ndimage
def resize(img, shape,order=1, mode='constant', orig_shape=(240, 240)):

    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1]
    )
    
    # Resize to the given shape
    return ndimage.zoom(img, factors, order=1,mode=mode)

def zero_padding(img, size0, pad1, pad2):
    zero_padding = np.zeros((img.shape[0], size0, size0), dtype=np.float32)
    pad1 = pad1 // 2
    pad2 = pad2 //2
    zero_padding[:, pad1:size0 - pad1, pad2:size0 - pad2] = img
    return zero_padding
from ssim_cal  import ssim
import torch.nn as nn
from math import log10
ssim_avg=0
psnr_avg=0
nmse_avg=0

ssim_avg_t1=0
psnr_avg_t1=0
nmse_avg_t1=0

ssim_avg_flair=0
psnr_avg_flair=0
nmse_avg_flair=0

ssim_avg_x=0
psnr_avg_x=0
nmse_avg_x=0
criterionMSE = nn.MSELoss().to(device)
import random
# net_g.eval()
import torch.nn.functional as F
with torch.no_grad():

    for i, batch in enumerate(testing_data_loader):

        input_t1, input_t2,input_flair = batch[0].cuda(), batch[1].cuda(),batch[2].cuda()

        t1_out,x_out,prediction,alpha_soft= net_g(input_flair)
        # #计算SSIM
        ssim0=ssim(prediction, input_t2)
        print("ssim0",ssim0)
        ssim_avg += ssim0    
        ssim0_t1=ssim(t1_out, input_t2)
        print("ssim0_t1",ssim0_t1)
        ssim_avg_t1 += ssim0_t1
        ssim0_flair=ssim(flair_out, input_t2)
        print("ssim0_flair",ssim0_flair)
        ssim_avg_flair += ssim0_flair
        ssim0_x=ssim(x_out, input_t2)
        print("ssim0_x",ssim0_x)
        ssim_avg_x += ssim0_x  

        #计算PSNR
        mse = criterionMSE(prediction, input_t2)
        # writer.add_scalar('mse', mse.item(), epoch)
        psnr = 10 * log10(1 / mse.item())
        print("psnr",psnr)
        psnr_avg += psnr

        mse_t1 = criterionMSE(t1_out, input_t2)
        # writer.add_scalar('mse', mse.item(), epoch)
        psnr_t1 = 10 * log10(1 / mse_t1.item())
        print("psnr_t1",psnr_t1)
        psnr_avg_t1 += psnr_t1

        mse_flair = criterionMSE(flair_out, input_t2)
        # writer.add_scalar('mse', mse.item(), epoch)
        psnr_flair = 10 * log10(1 / mse_flair.item())
        print("psnr_flair",psnr_flair)
        psnr_avg_flair += psnr_flair  
        mse_x = criterionMSE(x_out, input_t2)
        # writer.add_scalar('mse', mse.item(), epoch)
        psnr_x  = 10 * log10(1 / mse_x .item())
        print("psnr_x ",psnr_x )
        psnr_avg_x  += psnr_x         
        t2_mean=torch.mean(input_t2)**2
        nmse=mse.item()/t2_mean.item()
        nmse_avg+=nmse
        t2_mean_t1=torch.mean(input_t2)**2
        nmse_t1=mse_t1.item()/t2_mean_t1.item()
        nmse_avg_t1+=nmse_t1

        t2_mean_flair=torch.mean(input_t2)**2
        nmse_flair=mse_flair.item()/t2_mean_flair.item()
        nmse_avg_flair+=nmse_flair

        t2_mean_x=torch.mean(input_t2)**2
        nmse_x=mse_x.item()/t2_mean_x.item()
        nmse_avg_x+=nmse_x

        # out=input_t1
        out_img = prediction.detach().squeeze(0).cpu()
        out_img_t1 = t1_out.detach().squeeze(0).cpu()
        out_img_flair =flair_out.detach().squeeze(0).cpu()
        out_img_x = x_out.detach().squeeze(0).cpu()
        out_img_cnn=prediction_cnn.detach().squeeze(0).cpu()
        input_t2 = input_t2.detach().squeeze(0).cpu()

        filename='{:s}{:s}'.format( str(i + 1), '.jpg')
        # filename=filename.replace('/', '_')
        if not os.path.exists(os.path.join(result_path, opt.dataset+"/fuse")):
            os.makedirs(os.path.join(result_path, opt.dataset+"/fuse"))
        save_img_fusion_old(out_img, "{}/{}/{}".format(result_path,opt.dataset+"/fuse",filename ),transform)
       
        if not os.path.exists(os.path.join(result_path, opt.dataset+"/resize")):
             os.makedirs(os.path.join(result_path, opt.dataset+"/resize"))    
        save_img_fusion_old(input_t2, "{}/{}/{}".format(result_path,opt.dataset+"/resize", filename),transform)

        if not os.path.exists(os.path.join(result_path, opt.dataset+"/t1_out")):
             os.makedirs(os.path.join(result_path, opt.dataset+"/t1_out"))    
        save_img_fusion_old(out_img_t1, "{}/{}/{}".format(result_path,opt.dataset+"/t1_out", filename),transform)

        if not os.path.exists(os.path.join(result_path, opt.dataset+"/flair_out")):
             os.makedirs(os.path.join(result_path, opt.dataset+"/flair_out"))    
        save_img_fusion_old(out_img_flair, "{}/{}/{}".format(result_path,opt.dataset+"/flair_out", filename),transform)

        if not os.path.exists(os.path.join(result_path, opt.dataset+"/x_out")):
             os.makedirs(os.path.join(result_path, opt.dataset+"/x_out"))    
        save_img_fusion_old(out_img_x, "{}/{}/{}".format(result_path,opt.dataset+"/x_out", filename),transform)

        if not os.path.exists(os.path.join(result_path, opt.dataset+"/fuse_cnn")):
            os.makedirs(os.path.join(result_path, opt.dataset+"/fuse_cnn"))
        save_img_fusion_old(out_img_cnn, "{}/{}/{}".format(result_path,opt.dataset+"/fuse_cnn",filename ),transform)

        filename='{:s}{:s}'.format( str(i + 1), '.jpg')
        # filename=filename.replace('/', '_')
        if not os.path.exists(os.path.join(result_path, opt.dataset+"/fuse")):
            os.makedirs(os.path.join(result_path, opt.dataset+"/fuse"))
        save_img_fusion_old(out_img, "{}/{}/{}".format(result_path,opt.dataset+"/fuse",filename ),transform)

    number=len(testing_data_loader)
    print("ssim_avg",ssim_avg/number)
    print("psnr_avg",psnr_avg/number)
    print("nmse_avg",nmse_avg/number)
    print("ssim_avg_t1",ssim_avg_t1/number)
    print("psnr_avg_t1",psnr_avg_t1/number)
    print("nmse_avg_t1",nmse_avg_t1/number)
    print("ssim_avg_flair",ssim_avg_flair/number)
    print("psnr_avg_flair",psnr_avg_flair/number)
    print("nmse_avg_flair",nmse_avg_flair/number)
    print("ssim_avg_x",ssim_avg_x/number)
    print("psnr_avg_x",psnr_avg_x/number)
    print("nmse_avg_x",nmse_avg_x/number)
