import torch
import os
import numpy as np
from torch.utils import data as data
import cv2


def read_img(path):
    img=cv2.imread(path, -1)
    img = np.asarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).transpose(2,0,1)/255.
    return torch.from_numpy(img).float()


class Get_Dataset(data.Dataset):
    def __init__(self, discrip='cset', path=None, path_gt=None, noise_type='gauss', sigma=0):
        '''for syn dataset, only use path
        noise_type in [gauss,poiss,speckle,S&P,gauss_localvar]'''
        self.imgs=os.listdir(path)
        self.path=path
        self.sigma=sigma
        self.noise_type=noise_type
        self.discrip=discrip
        if not noise_type=='real':
            self.path_gt=path
        else:
            self.path_gt=path_gt

    def __getitem__(self, index):
        imname=self.imgs[index]
        file=os.path.join(self.path,imname)
        file_gt=os.path.join(self.path_gt,imname)

        img_gt_torch=read_img(file_gt)
        if self.noise_type=='real':
            img_noisy_torch=read_img(file)
        else:
            img_noisy_torch=add_noise_torch(img_gt_torch,self.sigma) if not self.sigma==0 else img_gt_torch
            

        data={'img_noisy_torch':img_noisy_torch,
            'img_gt_torch':img_gt_torch,
            'im_name':imname.split('.')[0]}
        
        return data


    def __len__(self):
        return len(self.imgs)
    


def add_noise_torch(img,sigma,noise_type='gauss'):
    if noise_type == 'gauss':
        noise=torch.randn(img.shape, requires_grad= False, device = img.device)*(sigma/255.)
        noisy=noise+img
    elif noise_type == 'poiss':
        noisy = torch.poisson(img*sigma)/sigma
    elif noise_type == 'speckle':
        noise=img * torch.FloatTensor(img.shape).normal_(mean=0, std=sigma/255.)
        noisy=noise+img
    elif noise_type == 'S&P':
        salt_vs_pepper=0.5;amount=1/sigma
        def _bernoulli(p, shape):
            return torch.rand(shape) <= p
        
        noisy=img.clone()
        flipped = _bernoulli(amount, img.shape)
        salted = _bernoulli(salt_vs_pepper, img.shape)
        peppered = ~salted

        noisy[flipped & salted] = 1
        noisy[flipped & peppered] = 0
    elif noise_type == 'gauss_localvar':
        log_shot=np.log(sigma/10000)
        sigma_shot=np.exp(log_shot)
        line = lambda x: 2.18 * x + 1.20
        log_read=line(log_shot)
        sigma_read=np.exp(log_read)

        loc_var=img*sigma_shot+sigma_read
        print('loc_var:',sigma_shot,sigma_read)
        noise=torch.randn(img.shape, requires_grad= False, device = img.device)*(np.sqrt(loc_var))
        noisy = img + noise
    return noisy