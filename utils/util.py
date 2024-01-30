import torch
import numpy as np
import torch.nn.functional as F
from models.Fill_network_arch import Fill_network
import torch.nn as nn
from math import ceil
import yaml


def write_logs(string,log_dir='log.txt'):
    print(string)
    with open(log_dir,'a') as f:
        f.write(string+'\n')


class DictToClass(object):
    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('new', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_


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
    net_fill=Fill_network(in_chans=args.input_depth,mask_ratio=args.mask_ratio,
                       multchannel=True if 'multchannel' in args.modelpath else False,
                       network=args.modelpath.split('_')[-1].split('.')[0])
    net_fill=net_fill.to(device)
    comp_params(net_fill)

    lr_groups=[]
    if resume:
        saved_state_dict = torch.load(args.modelpath, map_location = device)['params']
        net_fill.load_state_dict(saved_state_dict,strict=True)
    lr_groups.append({'params':net_fill.parameters(),'lr':args.lr})

    return net_fill,lr_groups