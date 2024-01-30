import torch


def smooth_out_withmask(out_avg,out,mask,exp_weight=0.99):
    # Smoothing
    if out_avg is None:
        out_avg = torch.div(torch.mul(out.detach(),1-mask),1-mask)
    else:
        out_avg=torch.where((~torch.isnan(out_avg)) & (mask==0),out.detach()*(1-exp_weight)+out_avg*exp_weight,out_avg)
        out_avg=torch.where(torch.isnan(out_avg) & (mask==0),out.detach(),out_avg)
    return out_avg

