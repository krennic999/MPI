import torch
import torch.nn.functional as F
import torch.nn as nn
from math import ceil


def check_image_size(x,mod=3):
    B,C,H,W = x.shape
    mod_pad_h = (mod - H % mod) % mod + mod
    mod_pad_w = (mod - W % mod) % mod + mod
    x = F.pad(x,(0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def pad_to_patch_size(image, patch_size):
    """
    Pads the input image so that its dimensions are multiples of patch_size.

    Parameters:
    image (torch.Tensor): The input image tensor of shape (C, H, W).
    patch_size (int): The size of the patch.

    Returns:
    torch.Tensor: The padded image tensor.
    """
    _,_, height, width = image.size()
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    # Pad the image
    # The padding is added to the bottom and right sides of the image
    padding = (0, pad_width, 0, pad_height)
    padded_image = F.pad(image, padding, mode='constant', value=0)
    return padded_image


def crop_patch(input, patch_size=1024, margin=30):
    '''
    Crops the input into patches with overlap.
    input: The input tensor.
    patch_size: The size of each patch.
    margin: The overlap between adjacent patches.
    '''
    H, W = input.shape[-2:]
    input=pad_to_patch_size(input,patch_size)
    cropped_input = []
    step_size = patch_size - margin  # Calculate the step size considering the overlap

    for xx in range(0, ceil((H - margin) / step_size)):
        for yy in range(0, ceil((W - margin) / step_size)):
            x_start = xx * step_size
            y_start = yy * step_size
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            cropped_input.append(input[..., x_start:x_end, y_start:y_end])
    return cropped_input


def resume_from_patch(denoise_all, img_size, patch_size=1024, margin=30):
    H, W = img_size
    denoise_resumed = torch.zeros((1, 3, H, W), device=denoise_all[0].device)
    overlap_count = torch.zeros((1, 3, H, W), device=denoise_all[0].device)

    step_size = patch_size - margin  # Calculate the step size considering the overlap

    ind = 0
    for xx in range(ceil((H - margin) / step_size)):
        for yy in range(ceil((W - margin) / step_size)):
            x_start = xx * step_size
            y_start = yy * step_size
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            # Add the patch to the resumed image and update the overlap count
            denoise_resumed[..., x_start:x_end, y_start:y_end] += denoise_all[ind]
            overlap_count[..., x_start:x_end, y_start:y_end] += 1
            ind += 1

    # Average the overlapping regions
    denoise_resumed /= overlap_count

    return denoise_resumed[...,:H,:W]


def down_image(img,shuffle):
    down=nn.PixelUnshuffle(shuffle)
    return torch.chunk(down(img),chunks=shuffle**2,dim=1)


def up_image2(imgs,shuffle):
    B,C,H,W=imgs.shape
    up=nn.PixelShuffle(shuffle)
    # ret=[]
    # for i in range(C):
    #     ret.append(up(imgs[:,i].unsqueeze(0)))
    # return torch.cat(ret,dim=1)
    return up(torch.cat(torch.chunk(imgs,chunks=B,dim=0),dim=1))