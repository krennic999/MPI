import torch
import math
import cv2
import yaml
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity


from models.Fill_skip_arch import Fill_skip


def test(noisy_img, clean_img):
    psnr = compare_psnr(clean_img,noisy_img)
    ssim=cal_ssim(clean_img,noisy_img)
    return psnr,ssim

def cal_ssim(pack1, pack2):
    '''b,c,t,h,w'''
    C,H,W=pack1.shape
    pack1=pack1.detach().cpu().numpy().transpose(1,2,0)
    pack2=pack2.detach().cpu().numpy().transpose(1,2,0)
    return structural_similarity(pack1,pack2,channel_axis=2,data_range=1)


def write_logs(string,log_dir='log.txt'):
    with open(log_dir,'a') as f:
        f.write(string+'\n')

def read_img(path):
    img=cv2.imread(path, -1)
    img = np.asarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).transpose(2,0,1)/255.
    return torch.from_numpy(img).float()


def compare_psnr(a,b):
    a=torch.clip(a,0,1);b=torch.clip(b,0,1)
    x = torch.mean((a - b) ** 2, dim=[-3, -2, -1])
    return torch.mean(20 * torch.log(1 / torch.sqrt(x)) / math.log(10)).item()


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


def smooth_out_withmask(out_avg,out,mask,exp_weight=0.99):
    if out_avg is None:
        out_avg = torch.div(torch.mul(out.detach(),1-mask),torch.sum(1-mask,dim=0,keepdim=True))
    else:
        out_avg=torch.where((~torch.isnan(out_avg)) & (mask==0),out.detach()*(1-exp_weight)+out_avg*exp_weight,out_avg)
        out_avg=torch.where(torch.isnan(out_avg) & (mask==0),out.detach(),out_avg)
    return out_avg


class DictToClass(object):
    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('new', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_


def check_image_size(x,mod=3):
    x=x.unsqueeze(0)
    B,C,H,W = x.shape
    mod_pad_h = (mod - H % mod) % mod + mod
    mod_pad_w = (mod - W % mod) % mod + mod
    x = F.pad(x,(0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def get_args(yml_path):
    ## setup
    args=DictToClass._to_class(
                yaml.load(open(yml_path,'r',encoding='utf-8').read(),Loader=yaml.FullLoader)
                )
    return args


def tensor2plot(input):
    return torch.clip(input,0,1).cpu().squeeze(0).permute(1,2,0)


def comp_params(net):
    '''Compute number of parameters'''
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)


def get_model(args,resume=True,device='cuda'):
    net_fill=Fill_skip(in_chans=args.INPUT_DEPTH,mask_ratio=args.mask_ratio,
                       mode='small' if 'small' in args.modelpath else 'normal',
                       multchannel=True if 'multchannel' in args.modelpath else False)
    net_fill=net_fill.to(device)#去噪
    comp_params(net_fill)

    lr_groups=[]
    if resume:
        saved_state_dict = torch.load(args.modelpath, map_location = device)['params']
        net_fill.load_state_dict(saved_state_dict,strict=True)
    lr_groups.append({'params':net_fill.parameters(),'lr':args.LR})

    return net_fill,lr_groups