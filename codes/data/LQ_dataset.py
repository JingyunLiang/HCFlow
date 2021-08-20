import random
import numpy as np
import cv2
import lmdb
import torch
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
    Load  LR images only.
    '''

    def __init__(self, opt):
        super(LQDataset, self).__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.scale = opt['scale']
        if self.opt['phase'] == 'train':
            self.GT_size = opt['GT_size']
            self.LR_size = self.GT_size // self.scale

        # read image list from lmdb or image files
        if opt['data_type'] == 'lmdb':
            self.LR_paths, self.LR_sizes = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
            self.GT_paths, self.GT_sizes = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])
        elif opt['data_type'] == 'img':
            self.LR_paths = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])  # LR list
            # self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list
        else:
            print('Error: data_type is not matched in Dataset')
        assert self.LR_paths, 'Error: LR paths are empty.'

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        if self.opt['dataroot_LQ'] is not None:
            self.LR_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                    meminit=False)
        else:
            self.LR_env = 'No lmdb input for LR'

    def __getitem__(self, index):
        if self.opt['data_type'] == 'lmdb':
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        if self.opt['data_type'] == 'lmdb':
            resolution = [int(s) for s in self.GT_sizes[index].split('_')]
        else:
            resolution = None


        # loading code from srflow test
        # img_GT = cv2.imread(GT_path)[:, :, [2, 1, 0]]
        # img_GT = torch.Tensor(img_GT.transpose([2, 0, 1]).astype(np.float32)) / 255
        # img_LR = cv2.imread(LR_path)[:, :, [2, 1, 0]]
        # pad_factor = 2
        # h, w, c = img_LR.shape
        # img_LR = impad(img_LR, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
        #    right=int(np.ceil(w / pad_factor) * pad_factor - w))
        # img_LR = torch.Tensor(img_LR.transpose([2, 0, 1]).astype(np.float32)) / 255


        # get LR image
        LR_path = self.LR_paths[index]
        img_LR = util.read_img(self.LR_env, LR_path, resolution)

        # change color space if necessary, deal with gray image
        if self.opt['color']:
            img_LR = util.channel_convert(img_LR.shape[2], self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {'LQ': img_LR, 'LQ_path': LR_path}

    def __len__(self):
        return len(self.LR_paths)


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')
