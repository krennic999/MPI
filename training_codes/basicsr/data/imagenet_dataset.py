# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
import torch
import numpy as np
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, scandir


class ImageNetDataset(data.Dataset):
    """data from ImageNet dataset.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ImageNetDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.lq_folder = opt['dataroot_lq']
        self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load lq image. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']   # patch size

            # random crop
            img_lq = random_crop(img_lq, gt_size, lq_path)
            # flip, rotation
            img_lq = augment(img_lq, self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq,
                                    bgr2rgb=True,
                                    float32=True)
                                    
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        if self.opt['noise']==True:
            noise=torch.randn(img_lq.shape, requires_grad= False, device = img_lq.device)*(self.opt['sigma']/255.)
            img_lq_noise=img_lq.detach().clone()+noise

        return {
            'lq': img_lq_noise if self.opt['noise']==True else img_lq,
            'gt': img_lq,
            'lq_path': lq_path,
            'gt_path': lq_path
        }

    def __len__(self):
        return len(self.paths)
