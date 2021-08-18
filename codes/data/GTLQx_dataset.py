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


class GTLQxDataset(data.Dataset):
    '''
    Load  HR-LR image pairs.
    '''

    def __init__(self, opt):
        super(GTLQxDataset, self).__init__()
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
            self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list
        else:
            print('Error: data_type is not matched in Dataset')
        assert self.GT_paths, 'Error: GT paths are empty.'
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths), 'GT and LR datasets have different number of images - {}, {}.'.format(
                len(self.LR_paths), len(self.GT_paths))

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


        # get GT and LR image
        GT_path = self.GT_paths[index]
        # LR_path = self.LR_paths[index]
        LR_path = GT_path.replace('HR', 'LR_bicubic/X4').replace('.png','x{}.png'.format(self.scale))
        img_GT = util.read_img(self.GT_env, GT_path, resolution)  # return: Numpy float32, HWC, BGR, [0,1]
        img_LR = util.read_img(self.LR_env, LR_path, resolution)


        if self.opt['phase'] == 'train':
            # crop
            H, W, C = img_LR.shape
            rnd_top_LR = random.randint(0, max(0, H - self.LR_size))
            rnd_left_LR = random.randint(0, max(0, W - self.LR_size))
            rnd_top_GT = rnd_top_LR * self.scale
            rnd_left_GT = rnd_left_LR * self.scale

            img_GT = img_GT[rnd_top_GT:rnd_top_GT + self.GT_size, rnd_left_GT:rnd_left_GT + self.GT_size, :]
            img_LR = img_LR[rnd_top_LR:rnd_top_LR + self.LR_size, rnd_left_LR:rnd_left_LR + self.LR_size, :]

            # augmentation - flip, rotate
            img_GT, img_LR = util.augment([img_GT, img_LR], self.opt['use_flip'],
                                  self.opt['use_rot'], self.opt['mode'])

        # change color space if necessary, deal with gray image
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LR = util.channel_convert(img_LR.shape[2], self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()


        # modcrop
        _, H, W = img_LR.size()
        img_GT = img_GT[:, :H*self.scale, :W*self.scale]

        return {'LQ': img_LR, 'GT': img_GT, 'LQ_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.GT_paths)


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')
