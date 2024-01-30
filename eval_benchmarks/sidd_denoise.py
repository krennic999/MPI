import sys, os
sys.path.append('..')

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = 'cuda'

import numpy as np
import time
import torch
from torch.optim import Adam
from scipy.io import loadmat,savemat
from utils.util import get_args,get_model
from utils.img_utils import check_image_size,down_image,up_image2
from utils.smooth_util import smooth_out_withmask
from eval_benchmarks.pytorch_wrapper import pytorch_denoiser


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


def DenoiseSrgb(Denoiser, siddDataDir, out_folder):

    """
    DenoiseSrgb denoises SIDD benchmark images in sRGB space using some Denoiser.
    :param Denoiser: A function handle to the denoiser to be evaluated on the SIDD Benchmark.
    :param siddDataDir: The directory containing the SIDD Benchmark images.
    :return: The denoised blocks from sRGB images and denoising time per megapixel.
    """
    try:
        os.makedirs(out_folder)
    except:pass

    # Load block positions
    BenchmarkNoisyBlocks = loadmat(os.path.join(siddDataDir, '..', 'BenchmarkNoisyBlocksSrgb.mat'))
    BenchmarkNoisyBlocks=BenchmarkNoisyBlocks['BenchmarkNoisyBlocksSrgb']
    nImages = BenchmarkNoisyBlocks.shape[0]
    nBlocks = BenchmarkNoisyBlocks.shape[1]

    DenoisedBlocksSrgb = [[None for _ in range(nBlocks)] for _ in range(nImages)]
    TimeMP = 0

    # For each image
    for i in range(nImages):
        # Load noisy sRGB image

        # Estimate noise Sigma for image i
        print(f'Estimating noise Sigma for image {i+1:02d} ... ')
        noiseSigma = None # Replace with your noise estimation method if needed

        # For each block
        for j in range(nBlocks):
            noisyImage = BenchmarkNoisyBlocks[i][j].astype(np.float32) / 255.0

            print(f'Denoising sRGB image {i+1:02d}, block {j+1:02d} ... ', end='')

            # Denoise 3 RGB channels simultaneously
            t0 = time.time()
            denoisedBlock = Denoiser(noisyImage, noiseSigma)
            t1 = time.time() - t0

            denoised_out=np.round((denoisedBlock * 255)).astype(np.uint8)
            DenoisedBlocksSrgb[i][j] = denoised_out
            save_file = os.path.join(out_folder, '%d_%d.mat'%(i,j))
            savemat(save_file, {"denoised": denoised_out})

            # Total time
            TimeMP += t1
            print(f'Time = {t1:.6f} seconds')

    save_file_total = os.path.join(out_folder, 'SubmitSrgb.mat')
    savemat(save_file_total, {"denoised": DenoisedBlocksSrgb})

    return DenoisedBlocksSrgb

if __name__=='__main__':
    # Example usage
    denoised_blocks = DenoiseSrgb(\
        pytorch_denoiser(MPI_denoise,use_cuda=True), 
        siddDataDir='/SIDD/SIDD_Benchmark_Data/',
        out_folder='../outputs/SIDD_benchmark')
