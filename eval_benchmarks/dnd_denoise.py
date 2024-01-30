 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.
import sys, os
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = 'cuda'

import numpy as np
import scipy.io as sio
import os
import h5py
from eval_benchmarks.pytorch_wrapper import pytorch_denoiser
from torch.optim import Adam
from utils.util import get_args,get_model
from utils.img_utils import check_image_size,down_image,up_image2
from utils.smooth_util import smooth_out_withmask
import time
import torch


arg_path='../configs/Fill_m_sidd.yml'
args=get_args(arg_path)
args.modelpath='../'+args.modelpath


def train(args,img_noisy_torch,net_fill,optimizer_fill,criterion_fill):
    B,C,H,W=img_noisy_torch.shape
    if not args.shuffle==1:
        img_noisy_torch=check_image_size(img_noisy_torch,args.shuffle)
        img_noisy_torch=torch.cat(down_image(img_noisy_torch,args.shuffle),dim=0)#(shuffle**2,c,h,w)

    psnr_list=[];psnr_noisy_list=[]
    out_avg=img_noisy_torch.clone().detach() if args.load_initial=='noisy' else None

    start = time.time()
    for iter in range(args.num_iter):
        # for random mask
        img_noisy_torch_tmp=img_noisy_torch
        out = net_fill(img_noisy_torch_tmp)
        mask = net_fill.get_mask()

        if args.is_smooth:
            out_avg=smooth_out_withmask(out_avg,out,mask,exp_weight=args.exp_weight)

        optimizer_fill.zero_grad()
        # noisy2noisy loss
        loss_fill = criterion_fill(torch.mul(out,1-mask), torch.mul(img_noisy_torch,1-mask))
        loss_fill.backward()
        optimizer_fill.step()

        if (iter+1) % args.show_every ==0:            
            print('Iteration:%05d'%(iter+1), '\r', end='')

    if not args.shuffle==1:
        out_avg=up_image2(out_avg,args.shuffle).detach().cpu()[...,:H,:W]
    else:
        out_avg=out_avg[...,:H,:W]

    end = time.time();ref_time=end-start
    return ref_time,out_avg


def MPI_denoise(noisy,sigma):
    net_fill,lr_groups=get_model(args,device=device)
    optimizer_fill = Adam(lr_groups, lr=args.lr)
    criterion_fill = torch.nn.MSELoss().to(device)
    ref_time,out_avg_plt=train(args,noisy,net_fill,optimizer_fill,criterion_fill)
    print('elapsed time: %.2f s'%ref_time)
    return out_avg_plt

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx,yy,bb]
    return sigma

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma


def denoise_srgb(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            for yy in range(2):
                for xx in range(2):
                    nlf["sigma"] = load_sigma_srgb(info, i, k)
                    Idenoised_crop = denoiser(Inoisy_crop, nlf)
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))


if __name__=='__main__':
    denoise_srgb(pytorch_denoiser(MPI_denoise,use_cuda=True),data_folder='/data3/mxx/datasets/DND/',out_folder='../outputs/DND_benchmark_beta')