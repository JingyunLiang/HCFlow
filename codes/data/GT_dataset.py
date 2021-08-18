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


class GTDataset(data.Dataset):
    '''
    Load  GT images only. 30s faster than LQGTKer (90s for 200iter).
    '''

    def __init__(self, opt):
        super(GTDataset, self).__init__()
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
        assert self.GT_paths, 'Error: GT paths are empty.'
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths), 'GT and LR datasets have different number of images - {}, {}.'.format(
                len(self.LR_paths), len(self.GT_paths))
        self.random_scale_list = [1]

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

        GT_path, LR_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt['data_type'] == 'lmdb':
            resolution = [int(s) for s in self.GT_sizes[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        img_GT = util.modcrop(img_GT, scale)

        # get LR image
        if self.LR_paths:  # LR exist
            raise ValueError('GTker_dataset.py doesn Not allow LR input.')

        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                if random_scale != 1:
                    H_s, W_s, _ = img_GT.shape
                    H_s = _mod(H_s, random_scale, scale, GT_size)
                    W_s = _mod(W_s, random_scale, scale, GT_size)
                    img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)

                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

        if self.opt['phase'] == 'train':
            H, W, C = img_GT.shape

            # randomly crop on HR, more positions than first crop on LR and HR simultaneously
            rnd_h_GT = random.randint(0, max(0, H - GT_size))
            rnd_w_GT = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_GT = util.augment(img_GT, self.opt['use_flip'],
                                  self.opt['use_rot'], self.opt['mode'])

        # change color space if necessary, deal with gray image
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = GT_path

        # don't need LR because it's generated from HR batches.
        img_LR = torch.ones(1, 1, 1)

        return {'LQ': img_LR, 'GT': img_GT, 'LQ_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.GT_paths)


def _mod(n, random_scale, scale, thres):
    rlt = int(n * random_scale)
    rlt = (rlt // scale) * scale
    return thres if rlt < thres else rlt
