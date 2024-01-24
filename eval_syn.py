from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = 'cuda'

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam,lr_scheduler
import random
from dataset import add_noise_torch,Get_Dataset
from util import compare_psnr,get_args,get_model,test,tensor2plot,write_logs
from models.Fill_network_arch import Fill_network
from PIL import Image
import time
from torch.utils.data import DataLoader


def smooth_out_withmask(out_avg,out,mask,exp_weight=0.99):
    # Smoothing
    if out_avg is None:
        out_avg = torch.div(torch.mul(out.detach(),1-mask).sum(dim=0,keepdim=True),torch.sum(1-mask,dim=0,keepdim=True))
        out_avg=torch.where(torch.isnan(out_avg),torch.full_like(out_avg,0),out_avg)
    else:
        out_ = torch.div(torch.mul(out.detach(),1-mask).sum(dim=0,keepdim=True),torch.sum(1-mask,dim=0,keepdim=True))
        mask_=torch.mean(1-mask.float(),dim=0,keepdim=True).ceil()
        out_=torch.where(torch.isnan(out_),torch.full_like(out_,0),out_)
        out_avg = (out_avg * exp_weight + out_ * (1 - exp_weight))*mask_+out_avg*(1-mask_)
    return out_avg


def eval(dataloader,args,sigma=0,savepath='./outputs'):
    psnr_list=[]
    ssim_list=[]
    time_list=[]
    imgs=os.listdir(path)

    dataset=dataloader.dataset.discrip
    sigma=dataloader.dataset.sigma
    ntype=dataloader.dataset.noise_type
    write_logs(dataset+' in '+dataloader.dataset.path)
    savepath=os.path.join(savepath,dataset+'_'+ntype+'_'+str(sigma))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    write_logs('found %d imgs, saving to %s'%(dataloader.dataset.__len__(), savepath))

    psnr_list=[]
        
    for idx, data in enumerate(dataloader):
        im_name=data['im_name'][0]
        img_noisy_torch=data['img_noisy_torch']
        img_gt_torch=data['img_gt_torch']

        print('processing %s'%im_name)

        net_fill,lr_groups=get_model(args,device=device)
        optimizer_fill = Adam(lr_groups, lr=args.lr)
        criterion_fill = torch.nn.MSELoss().to(device)

        psnr_out,ssim_out,psnr_noisy,ref_time,out_avg_plt=\
            train(args,data,net_fill,optimizer_fill,criterion_fill)
        
        write_logs(im_name+'   psnr= %.4f,   ssim= %.4f'%(psnr_out,ssim_out))

        Image.fromarray((np.clip(out_avg_plt,0,1) * 255).astype(np.uint8))\
                    .save(os.path.join(savepath,im_name.split('.')[0]+'_%.4f.png'%psnr_out))
        Image.fromarray((np.clip(img_noisy_torch.detach().cpu().numpy()[0].transpose(1,2,0),0,1)
                          * 255).astype(np.uint8))\
                    .save(os.path.join(savepath,im_name.split('.')[0]+'noisy_%.4f.png'%psnr_noisy))
        
        psnr_list.append(psnr_out),ssim_list.append(ssim_out),time_list.append(ref_time)

        if args.plot:
            plt.figure();plt.plot(psnr_out,label='pretrained')
            plt.savefig(os.path.join(savepath,'psnr_plot.png'))

        write_logs('elapsed time: %.2f s'%ref_time)

    avg_psnr=sum(psnr_list)/len(psnr_list);avg_ssim=sum(ssim_list)/len(ssim_list)
    avg_time=sum(time_list)/len(time_list)
    write_logs('total avg psnr= %.4f'%avg_psnr)
    write_logs('total avg ssim= %.4f'%avg_ssim)
    write_logs('total avg time= %.4f'%avg_time)
    os.rename(savepath,savepath+'_psnr%.4f_ssim%.4f'%(avg_psnr,avg_ssim))

def train(args,data,net_fill,optimizer_fill,criterion_fill):
    psnr_list=[];psnr_noisy_list=[]
    img_noisy_torch=data['img_noisy_torch'].to(device)
    img_gt_torch=data['img_gt_torch'].to(device)
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
            psnr_noisy = compare_psnr(img_noisy_torch.detach(), out_avg.detach()) if not out_avg==None else 0
            psnr_gt    = compare_psnr(torch.mul(img_gt_torch.detach(),1-mask), torch.mul(out.detach(),1-mask))
            psnr_gt_sm = compare_psnr(img_gt_torch.detach(), out_avg.detach()) if not out_avg==None else 0
            
            print ('Iteration:%05d  PSNR_noisy: %f PSRN_gt: %f PSNR_gt_sm: %f' \
                % (iter+1, psnr_noisy, psnr_gt, psnr_gt_sm), '\r', end='')
            psnr_list.append(psnr_gt_sm)
            psnr_noisy_list.append(psnr_noisy)

    end = time.time();ref_time=end-start
    psnr,ssim=test(img_gt_torch[0].detach(),out_avg[0].detach())
    out_avg_plt=out_avg[0].detach().cpu().numpy().transpose(1,2,0)
    print('\nfinal psnr= ',psnr_gt_sm)
    return psnr,ssim,psnr_noisy,ref_time,out_avg_plt


if __name__ == "__main__":
    noisetype='gauss'
    sigma_list=[25]
    dataset_list=['kodak']
    path_list=['/Kodak24/']
    arg_path='./configs/Fill_m_syn.yml'
    args=get_args(arg_path)
    
    for dataset_name,path in zip(dataset_list,path_list):
        for sigma in sigma_list:
            dataset=Get_Dataset(discrip=dataset_name,
                                path=path,noise_type=noisetype,sigma=sigma)
            dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
            eval(dataloader,args,sigma=sigma)