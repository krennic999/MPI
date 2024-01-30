import torch
import math
from skimage.metrics import structural_similarity



def test(noisy_img, clean_img):
    psnr = compare_psnr(clean_img,noisy_img)
    ssim=cal_ssim(clean_img,noisy_img)
    return psnr,ssim


def compare_psnr(a,b):
    a=torch.clip(a,0,1);b=torch.clip(b,0,1)
    x = torch.mean((a - b) ** 2, dim=[-3, -2, -1])
    return torch.mean(20 * torch.log(1 / torch.sqrt(x)) / math.log(10)).item()


def cal_ssim(pack1, pack2):
    '''b,c,t,h,w'''
    C,H,W=pack1.shape
    pack1=pack1.detach().cpu().numpy().transpose(1,2,0)
    pack2=pack2.detach().cpu().numpy().transpose(1,2,0)
    return structural_similarity(pack1,pack2,channel_axis=2,data_range=1)