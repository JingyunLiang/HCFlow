import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import glob

import natsort
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import collections
from scipy import signal
try:
    import accimage
except ImportError:
    accimage = None

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import scipy
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


####################
# PCA
####################

def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k] # PCA matrix

def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma

####################
# anisotropic gaussian kernels, identical to 'mvnpdf(X,mu,sigma)' in matlab
# due to /np.sqrt((2*np.pi)**2 * sig1*sig2), `sig1=sig2=8` != `sigma=8` in matlab
# rotation matrix [[cos, -sin],[sin, cos]]
####################

def anisotropic_gaussian_kernel_matlab(l, sig1, sig2, theta, tensor=False):
    # mean = [0, 0]
    # v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    # V = np.array([[v[0], v[1]], [v[1], -v[0]]]) # [[cos, sin], [sin, -cos]]
    # D = np.array([[sig1, 0], [0, sig2]])
    # cov = np.dot(np.dot(V, D), V) # VD(V^-1), V=V^-1

    cov11 = sig1*np.cos(theta)**2 + sig2*np.sin(theta)**2
    cov22 = sig1*np.sin(theta)**2 + sig2*np.cos(theta)**2
    cov21 = (sig1-sig2)*np.cos(theta)*np.sin(theta)
    cov = np.array([[cov11, cov21], [cov21, cov22]])

    center = l / 2.0 - 0.5
    x, y = np.mgrid[-center:-center+l:1, -center:-center+l:1]
    pos = np.dstack((y, x))
    k = scipy.stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov)

    k[k < scipy.finfo(float).eps * k.max()] = 0
    sumk = k.sum()
    if sumk != 0:
        k = k/sumk

    return torch.FloatTensor(k) if tensor else k

####################
# isotropic gaussian kernels, identical to 'fspecial('gaussian',hsize,sigma)' in matlab
####################

def isotropic_gaussian_kernel_matlab(l, sigma, tensor=False):
    center = [(l-1.0)/2.0, (l-1.0)/2.0]
    [x, y] = np.meshgrid(np.arange(-center[1], center[1]+1), np.arange(-center[0], center[0]+1))
    arg = -(x*x + y*y)/(2*sigma*sigma)
    k = np.exp(arg)

    k[k < scipy.finfo(float).eps * k.max()] = 0
    sumk = k.sum()
    if sumk != 0:
        k = k/sumk

    return torch.FloatTensor(k) if tensor else k

####################
# random/stable ani/isotropic gaussian kernel batch generation
####################

def random_anisotropic_gaussian_kernel(l=15, sig_min=0.2, sig_max=4.0, scale=3, tensor=False):
    sig1 = sig_min + (sig_max-sig_min)*np.random.rand()
    sig2 = sig_min + (sig1-sig_min)*np.random.rand()
    theta = np.pi*np.random.rand()

    k = anisotropic_gaussian_kernel_matlab(l=l, sig1=sig1, sig2=sig2, theta=theta, tensor=tensor)
    return k, np.array([sig1, sig2, theta])

def stable_anisotropic_gaussian_kernel(l=15, sig1=2.6, sig2=2.6, theta=0, scale=3, tensor=False):
    k = anisotropic_gaussian_kernel_matlab(l=l, sig1=sig1, sig2=sig2, theta=theta, tensor=tensor)
    return k, np.array([sig1, sig2, theta])

def random_isotropic_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, scale=3, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel_matlab(l, x, tensor=tensor)
    return k, np.array([x, x, 0])


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def stable_isotropic_gaussian_kernel(l=21, sig=2.6, scale=3, tensor=False):
    k = isotropic_gaussian_kernel_matlab(l, sig, tensor=tensor)
    # shift version 1: interpolation
    # k = shift_pixel(k, scale)
    # k = k/k.sum()
    return k, np.array([sig, sig, 0])

def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scale=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scale=scale, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scale=scale, tensor=tensor)

def stable_gaussian_kernel(l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, rate_iso=1.0, scale=3, tensor=False):
    if np.random.random() < rate_iso:
        return stable_isotropic_gaussian_kernel(l=l, sig=sig, scale=scale, tensor=tensor)
    else:
        return stable_anisotropic_gaussian_kernel(l=l, sig1=sig1, sig2=sig2, theta=theta, scale=scale, tensor=tensor)

# only these two func can be used outside this script
def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scale=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    batch_sigma = np.zeros((batch, 3))
    shifted_l = l - scale + 1
    for i in range(batch):
        batch_kernel[i, :shifted_l, :shifted_l], batch_sigma[i, :] = \
            random_gaussian_kernel(l=shifted_l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scale=scale, tensor=False)
    if tensor:
        return torch.FloatTensor(batch_kernel), torch.FloatTensor(batch_sigma)
    else:
        return batch_kernel, batch_sigma

def stable_batch_kernel(batch, l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, rate_iso=1.0, scale=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    batch_sigma = np.zeros((batch, 3))
    shifted_l = l - scale + 1
    for i in range(batch):
        batch_kernel[i, :shifted_l, :shifted_l], batch_sigma[i, :] = \
            stable_gaussian_kernel(l=shifted_l, sig=sig, sig1=sig1, sig2=sig2, theta=theta, rate_iso=rate_iso, scale=scale, tensor=False)
    if tensor:
        return torch.FloatTensor(batch_kernel), torch.FloatTensor(batch_sigma)
    else:
        return batch_kernel, batch_sigma

# for SVKE, MutualAffineConv
def stable_batch_kernel_SV_mode(batch, img_H=250, img_W=250, divide_H=1, divide_W=1, sv_mode=0, l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, rate_iso=1.0, scale=3, tensor=True):
    batch_kernel = np.zeros((batch, img_H*img_W, l, l))
    batch_sigma = np.zeros((batch, img_H*img_W, 3))
    shifted_l = l - scale + 1
    a = (2.5-0.175)*scale
    b = 0.175*scale
    for ibatch in range(batch):
        block_H = math.ceil(img_H/divide_H)
        block_W = math.ceil(img_W/divide_W)
        for h in range(block_H):
            for w in range(block_W):
                if sv_mode == 1:
                    sig1 = a + b
                    sig2 = a * h/block_H + b
                    theta = 0
                elif sv_mode == 2:
                    sig1 = a * w/block_W + b
                    sig2 = a * h/block_H + b
                    theta = 0
                elif sv_mode == 3:
                    sig1 = a + b
                    sig2 = b
                    theta = np.pi * (h/block_H)
                elif sv_mode == 4:
                    sig1 = a * w/block_W + b
                    sig2 = a * h/block_H + b
                    theta = np.pi * (h/block_H)
                elif sv_mode == 5:
                    sig1 = np.random.uniform(b, a+b)
                    sig2 = np.random.uniform(b, a+b)
                    theta = np.random.uniform(0, np.pi)
                elif sv_mode == 6:
                    sig1 = a + b
                    sig2 = b
                    if (h+w)%2 == 0:
                        theta = np.pi/4
                    else:
                        theta = np.pi*3/4

                kernel_hw, sigma_hw = stable_gaussian_kernel(l=shifted_l, sig=sig, sig1=sig1, sig2=sig2, theta=theta,
                                                             rate_iso=rate_iso, scale=scale, tensor=False)

                for m in range(divide_H):
                    for k in range(divide_W):
                        pos_h, pos_w = h*divide_H+m, w*divide_W+k
                        if  pos_h < img_H and pos_w < img_W:
                            batch_kernel[ibatch, pos_h*img_W+pos_w, :shifted_l, :shifted_l], \
                            batch_sigma[ibatch, pos_h*img_W+pos_w, :] = kernel_hw, sigma_hw

    if tensor:
        return torch.FloatTensor(batch_kernel), torch.FloatTensor(batch_sigma)
    else:
        return batch_kernel, batch_sigma



####################
# bicubic downsampling
####################

def b_GPUVar_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_view

def b_CPUVar_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_v[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_v = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_v

class BatchBicubic(nn.Module):
    def __init__(self, scale=4):
        super(BatchBicubic, self).__init__()
        self.scale = scale

    def forward(self, input):
        tensor = input.cpu().data
        B, C, H, W = tensor.size()
        H_new = int(H / self.scale)
        W_new = int(W / self.scale)
        tensor_view = tensor.view((B*C, 1, H, W))
        re_tensor = torch.zeros((B*C, 1, H_new, W_new))
        for i in range(B*C):
            img = to_pil_image(tensor_view[i])
            re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
        re_tensor_view = re_tensor.view((B, C, H_new, W_new))
        return re_tensor_view

class BatchSubsample(nn.Module):
    def __init__(self, scale=4):
        super(BatchSubsample, self).__init__()
        self.scale = scale

    def forward(self, input):
        return input[:, :, 0::self.scale, 0::self.scale]

####################
# image noises
####################

def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(sigma.new_tensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


####################
# batch degradation
####################

class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scale=3):
        self.l = l
        self.sig = sig
        self.sig1 = sig1
        self.sig2 = sig2
        self.theta = theta
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate_iso = rate_iso
        self.scale = scale

    def __call__(self, random, batch, tensor=False):
        if random == True: #random kernel
            return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate_iso,
                                       scale=self.scale, tensor=tensor)
        else: #stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, sig1=self.sig1, sig2=self.sig2, theta=self.theta,
                                       rate_iso=self.rate_iso, scale=self.scale, tensor=tensor)

class BatchSRKernel_SV(object):
    def __init__(self, l=21, sig=2.6, sig1=2.6, sig2=2.6, theta=0, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scale=3, divide_H=1, divide_W=1, sv_mode=0):
        self.l = l
        self.sig = sig
        self.sig1 = sig1
        self.sig2 = sig2
        self.theta = theta
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate_iso = rate_iso
        self.scale = scale
        self.divide_H = divide_H
        self.divide_W = divide_W
        self.sv_mode = sv_mode
        assert rate_iso == 0, 'only support aniso kernel at present'

    # currently only support batch=1, stable mode
    def __call__(self, random, batch, img_H, img_W, tensor=False):
        return stable_batch_kernel_SV_mode(batch, img_H=img_H, img_W=img_W, divide_H=self.divide_H, divide_W=self.divide_W, sv_mode=self.sv_mode, l=self.l, sig=self.sig, sig1=self.sig1, sig2=self.sig2, theta=self.theta,
                                       rate_iso=self.rate_iso, scale=self.scale, tensor=tensor)

class PCAEncoder(object):
    def __init__(self, weight, device=torch.device('cuda')):
        self.weight = weight.to(device) #[l^2, k]
        self.size = self.weight.size()

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size() #[B, l, l]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))

class PCADecoder(object):
    def __init__(self, weight, device=torch.device('cuda')):
        self.weight = weight.permute(1,0).to(device) #[k, l^2]
        self.size = self.weight.size()

    def __call__(self, batch_kernel_map):
        B, _ = batch_kernel_map.size() #[B, l, l]
        return torch.bmm(batch_kernel_map.unsqueeze(1), self.weight.expand((B, ) + self.size)).view((B, int(self.size[1]**0.5), int(self.size[1]**0.5)))

class CircularPad2d(nn.Module):
    def __init__(self, pad):
        super(CircularPad2d, self).__init__()
        self.pad = pad

    def forward(self, input):
        return F.pad(input, pad=self.pad, mode='circular')

class BatchBlur(nn.Module):
    def __init__(self, l=15, padmode='reflection'):
        super(BatchBlur, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'circular':
            if l % 2 == 1:
                self.pad = CircularPad2d((l // 2, l // 2, l // 2, l // 2))
            else:
                self.pad = CircularPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        else:
            raise NotImplementedError

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))


# spatially variant blur
class BatchBlur_SV(nn.Module):
    def __init__(self, l=15, padmode='reflection'):
        super(BatchBlur_SV, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'circular':
            if l % 2 == 1:
                self.pad = CircularPad2d((l // 2, l // 2, l // 2, l // 2))
            else:
                self.pad = CircularPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))

    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            pad = pad.view(C * B, 1, H_p, W_p)
            pad = F.unfold(pad, self.l).transpose(1, 2) # [CB, HW, k^2]
            kernel = kernel.flatten(2).unsqueeze(0).expand(3,-1,-1,-1)
            out_unf = (pad*kernel.contiguous().view(-1,kernel.size(2),kernel.size(3))).sum(2).unsqueeze(1)
            out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)

            return out

class SRMDPreprocessing(object):
    def __init__(self, scale, random, l=21, add_noise=False, device=torch.device('cuda'), sig=2.6, sig1=2.6, sig2=2.6, theta=0,
                 sig_min=0.2, sig_max=4.0, rate_iso=1.0, rate_cln=0.2, noise_high=0.05882, is_training=False, sv_mode=0):

        self.device = device
        self.l = l
        self.noise = add_noise
        self.noise_high = noise_high
        self.rate_cln = rate_cln
        self.scale = scale
        self.random = random
        self.rate_iso = rate_iso
        self.is_training = is_training
        self.sv_mode = sv_mode

        if self.sv_mode == 0: # spatial-variant
            self.blur = BatchBlur(l=l, padmode='replication')
            self.kernel_gen = BatchSRKernel(l=l, sig=sig, sig1=sig1, sig2=sig2, theta=theta,
                                        sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scale=scale)
        else: # spatial-invariant
            self.blur = BatchBlur_SV(l=l, padmode='replication')
            self.kernel_gen = BatchSRKernel_SV(l=l, sig=sig, sig1=sig1, sig2=sig2, theta=theta,
                                        sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scale=scale,
                                               divide_H=40, divide_W=40, sv_mode=sv_mode)
        self.sample = BatchSubsample(scale=scale)


    def __call__(self, hr_tensor, kernel=False):
        B, C, H, W = hr_tensor.size()

        # generate kernel
        if self.sv_mode == 0:
            b_kernels, b_sigmas = self.kernel_gen(self.random, B, tensor=True)
        else:
            b_kernels, b_sigmas = self.kernel_gen(self.random, B, H, W, tensor=True)
        b_kernels, b_sigmas = b_kernels.to(self.device) , b_sigmas.to(self.device)

        # blur and downsample
        lr = self.sample(self.blur(hr_tensor, b_kernels))
        lr_n = lr

        # Gaussian noise
        if self.noise:
            if self.is_training:
                Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln)).to(self.device)
            else:
                Noise_level = (torch.ones(B, 1)*self.noise_high).to(self.device)
            lr_n = b_GaussianNoising(lr_n, Noise_level)
            if len(b_sigmas.size()) == 2: # only concat for spatially invariant kernel
                b_sigmas = torch.cat([b_sigmas, Noise_level * 10], dim=1)

        # image quantization
        lr = (lr * 255.).round()/255.
        lr_n = (lr_n * 255.).round()/255.

        return (lr, lr_n, b_sigmas, b_kernels) if kernel else (lr, lr_n, b_sigmas)


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default), BGR channel order
    '''
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach()
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def img2tensor(img):
    '''
    # BGR to RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    '''
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x

####################
# metric
####################

def calculate_mnc(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    img2 = img2/np.sqrt(np.sum(img2**2))
    import scipy.signal as signal
    temp = signal.convolve2d(img2, img1, 'full')
    temp2 = np.sqrt(np.sum(img1**2))
    return np.max(temp)/temp2

def calculate_kernel_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_mse(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    return mse

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr_ssim(img1, img2, crop_border=0):
    if crop_border == 0:
        cropped_img1 = img1
        cropped_img2 = img2
    else:
        cropped_img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        cropped_img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    psnr = calculate_psnr(cropped_img1 * 255, cropped_img2 * 255)
    ssim = calculate_ssim(cropped_img1 * 255, cropped_img2 * 255)

    if img2.shape[2] == 3:  # RGB image
        img1_y = bgr2ycbcr(img1, only_y=True)
        img2_y = bgr2ycbcr(img2, only_y=True)
        if crop_border == 0:
            cropped_img1_y = img1_y
            cropped_img2_y = img2_y
        else:
            cropped_img1_y = img1_y[crop_border:-crop_border, crop_border:-crop_border]
            cropped_img2_y = img2_y[crop_border:-crop_border, crop_border:-crop_border]
        psnr_y = calculate_psnr(cropped_img1_y * 255, cropped_img2_y * 255)
        ssim_y = calculate_ssim(cropped_img1_y * 255, cropped_img2_y * 255)
    else:
        psnr_y, ssim_y = 0, 0

    return psnr, ssim, psnr_y, ssim_y


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

####################
# for debug
####################
def surf(Z, cmap='rainbow', figsize=None):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')

    w, h = Z.shape[:2]
    xx = np.arange(0,w,1)
    yy = np.arange(0,h,1)
    X, Y = np.meshgrid(xx, yy)
    ax3.plot_surface(X,Y,Z,cmap=cmap)
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
    plt.show()
    plt.savefig('/home/jinliang/Downloads/tmp.png')

def imagesc(Z):
    f, ax = plt.subplots(1, 1, squeeze=False)
    im = ax[0,0].imshow(Z, vmin=0, vmax=Z.max())
    plt.colorbar(im, ax=ax[0,0])
    plt.show()
    plt.savefig('/home/jinliang/Downloads/tmp.png')


# copyed from data.util
def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr, following matlab version instead of opencv
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def compute_RF_numerical(net, img_np, re_init_para=False):
    '''
    https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
    if re_init_para:
        net.apply(weights_init)

    img_ = Variable(torch.from_numpy(img_np).float().cuda(),requires_grad=True)
    out_cnn=net(img_) # here we have two inputs and two outputs
    out_shape=out_cnn.size()
    ndims=len(out_cnn.size())
    grad=torch.zeros(out_cnn.size()).cuda()
    l_tmp=[]
    for i in range(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(int(out_shape[i]/2))

    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad)
    grad_np=img_.grad[0,0].data.cpu().numpy()
    idx_nonzeros=np.where(grad_np!=0)
    RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]

    return RF

def plot_kernel(out_k_np, savepath, gt_k_np=None):
    plt.clf()
    if gt_k_np is None:
        ax = plt.subplot(111)
        im = ax.imshow(out_k_np, vmin=out_k_np.min(), vmax=out_k_np.max())
        plt.colorbar(im, ax=ax)
    else:

        ax = plt.subplot(121)
        im = ax.imshow(gt_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        ax.set_title('GT Kernel')

        ax = plt.subplot(122)
        im = ax.imshow(out_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        ax.set_title('Kernel PSNR: {:.2f}'.format(calculate_kernel_psnr(out_k_np, gt_k_np)))

    plt.show()
    plt.savefig(savepath)

def get_resume_paths(opt):
    resume_state_path = None
    resume_model_path = None
    if opt.get('path', {}).get('resume_state', None) == "auto":
        wildcard = os.path.join(opt['path']['training_state'], "*")
        paths = natsort.natsorted(glob.glob(wildcard))
        if len(paths) > 0:
            resume_state_path = paths[-1]
            resume_model_path = resume_state_path.replace('training_state', 'models').replace('.state', '_G.pth')
    else:
        resume_state_path = opt.get('path', {}).get('resume_state')
    return resume_state_path, resume_model_path


def opt_get(opt, keys, default=None):
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max_grad: {tensor.max()} min_grad: {tensor.min()}"
                  f" mean_grad: {tensor.mean()}")
    return printer


def register_hook(tensor, msg=''):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line, to get the gradient of a variable in debugging
    """
    print(f"{msg} shape: {tensor.shape}"
                  f" max_value: {tensor.max()} min_value: {tensor.min()}"
                  f" mean_value: {tensor.mean()}")
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

