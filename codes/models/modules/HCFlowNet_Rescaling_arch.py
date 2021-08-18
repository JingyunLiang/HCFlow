import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import opt_get
from models.modules.FlowNet_Rescaling_x4 import FlowNet
from models.modules import Basic, thops



class HCFlowNet_Rescaling(nn.Module):
    def __init__(self, opt, step=None):
        super(HCFlowNet_Rescaling, self).__init__()
        self.opt = opt
        self.quant = opt_get(opt, ['datasets', 'train', 'quant'], 256)

        hr_size = opt_get(opt, ['datasets', 'train', 'GT_size'], 160)
        hr_channel = opt_get(opt, ['network_G', 'in_nc'], 3)

        # hr->lr+z
        self.flow = FlowNet((hr_size, hr_size, hr_channel), opt=opt)

    # hr: HR image, lr: LR image, z: latent variable, u: conditional variable
    def forward(self, hr=None, lr=None, z=None, u=None, eps_std=None,
                add_gt_noise=False, step=None, reverse=False, training=True):

        # hr->z
        if not reverse:
            return self.normal_flow_diracLR(hr, lr, u, step=step, training=training)
        # z->hr
        else: # setting z to lr!!!
            return self.reverse_flow_diracLR(lr, z, u, eps_std=eps_std, training=training)


    #########################################diracLR
    # hr->lr+z, diracLR
    def normal_flow_diracLR(self, hr, lr, u=None, step=None, training=True):
        # 1. quantitize HR
        # hr = hr + (torch.rand(hr.shape, device=hr.device)) / self.quant # no quantization is better

        # 2. hr->lr+z
        fake_lr_from_hr, fake_z1, fake_z2 = self.flow(hr=hr, u=u, logdet=None, reverse=False, training=training)

        return torch.clamp(fake_lr_from_hr, 0, 1), fake_z1, fake_z2

    # lr+z->hr
    def reverse_flow_diracLR(self, lr, z, u, eps_std, training=True):

        # lr+z->hr
        fake_hr = self.flow(z=lr, u=u, eps_std=eps_std, reverse=True, training=training)

        return torch.clamp(fake_hr, 0, 1)


    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real