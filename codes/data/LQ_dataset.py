import random
import numpy as np
import cv2
import lmdb
import torch
import torch.nn.functional as F
import torch.utils.data as data
import data.util as util
import sys
import os

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util as utils
except ImportError:
    pass


class LQDataset(data.Dataset):
    '''
    Load  LR images only, e.g. real-world images
    '''

    def __init__(self, opt):
        super(LQDataset, self).__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt['LR_size'], opt['GT_size']

        # read image list from lmdb or image files
        if opt['data_type'] == 'lmdb':
            self.LR_paths, self.LR_sizes = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
            self.GT_paths, self.GT_sizes = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])
        elif opt['data_type'] == 'img':
            self.LR_paths = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])  # LR list
            self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list
        else:
            print('Error: data_type is not matched in Dataset')
        assert self.LR_paths, 'Error: LQ paths are empty.'
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths), 'GT and LR datasets have different number of images - {}, {}.'.format(
                len(self.LR_paths), len(self.GT_paths))
        self.random_scale_list = [1]

    def __getitem__(self, index):

        GT_path, LQ_path = None, None

        # get GT image
        LQ_path = self.LR_paths[index]
        img_LQ = util.read_img(None, LQ_path, None)  # return: Numpy float32, HWC, BGR, [0,1]

        if self.GT_paths:  # LR exist
            raise ValueError('LQ_dataset.py doesn Not allow HR input.')

        else:
            # force to 3 channels
            if img_LQ.ndim == 2:
                img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_GRAY2BGR)

        # change color space if necessary, deal with gray image
        if self.opt['color']:
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if GT_path is None:
            GT_path = LQ_path

        # don't need LR because it's generated from HR batches.
        img_GT = torch.ones(1, 1, 1)

        # deal with the image margins for real-world images
        img_LQ = img_LQ.unsqueeze(0)
        x_gt = F.interpolate(img_LQ, scale_factor=self.opt['scale'], mode='nearest')
        if self.opt['scale'] == 4:
            real_crop = 3
        elif self.opt['scale'] == 2:
            real_crop = 6
        elif self.opt['scale'] == 1:
            real_crop = 11
        assert real_crop * self.opt['scale'] * 2 > self.opt['kernel_size']
        x_gt = F.pad(x_gt, pad=(
            real_crop * self.opt['scale'], real_crop * self.opt['scale'], real_crop * self.opt['scale'],
            real_crop * self.opt['scale']), mode='replicate')  # 'constant', 'reflect', 'replicate' or 'circular

        kernel_gt, sigma_gt = utils.stable_batch_kernel(1, l=self.opt['kernel_size'], sig=10, sig1=0, sig2=0,
                                                        theta=0, rate_iso=1, scale=self.opt['scale'],
                                                        tensor=True)  # generate kernel [BHW], y [BCHW]

        blur_layer = utils.BatchBlur(l=self.opt['kernel_size'], padmode='zero')
        sample_layer = utils.BatchSubsample(scale=self.opt['scale'])
        y_blurred = sample_layer(blur_layer(x_gt, kernel_gt))
        y_blurred[:, :, real_crop:-real_crop, real_crop:-real_crop] = img_LQ
        img_LQ = y_blurred.squeeze(0)

        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.LR_paths)
