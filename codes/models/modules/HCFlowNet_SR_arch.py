import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import opt_get
from models.modules import Basic, thops



class HCFlowNet_SR(nn.Module):
    def __init__(self, opt, step=None):
        super(HCFlowNet_SR, self).__init__()
        self.opt = opt
        self.quant = opt_get(opt, ['quant'], 256)

        hr_size = opt_get(opt, ['datasets', 'train', 'GT_size'], 160)
        hr_channel = opt_get(opt, ['network_G', 'in_nc'], 3)
        scale = opt_get(opt, ['scale'])

        if scale == 4:
            from models.modules.FlowNet_SR_x4 import FlowNet
        elif scale == 8:
            from models.modules.FlowNet_SR_x8 import FlowNet
        else:
            raise NotImplementedError('Scale {} is not implemented'.format(scale))

        # hr->lr+z
        self.flow = FlowNet((hr_size, hr_size, hr_channel), opt=opt)

        self.quantization = Basic.Quantization()

    # hr: HR image, lr: LR image, z: latent variable, u: conditional variable
    def forward(self, hr=None, lr=None, z=None, u=None, eps_std=None,
                add_gt_noise=False, step=None, reverse=False, training=True):

        # hr->z
        if not reverse:
            return self.normal_flow_diracLR(hr, lr, u, step=step, training=training)
        # z->hr
        else:
            return self.reverse_flow_diracLR(lr, z, u, eps_std=eps_std, training=training)


    #########################################diracLR
    # hr->lr+z, diracLR
    def normal_flow_diracLR(self, hr, lr, u=None, step=None, training=True):
        # 1. quantitize HR
        pixels = thops.pixels(hr)

        # according to Glow and ours, it should be u~U(0,a) (0.06 better in practice), not u~U(-0.5,0.5) (though better in theory)
        hr = hr + (torch.rand(hr.shape, device=hr.device)) / self.quant
        logdet = torch.zeros_like(hr[:, 0, 0, 0]) + float(-np.log(self.quant) * pixels)

        # 2. hr->lr+z
        fake_lr_from_hr, logdet = self.flow(hr=hr, u=u, logdet=logdet, reverse=False, training=training)

        # note in rescaling, we use LR for LR loss before quantization
        fake_lr_from_hr = self.quantization(fake_lr_from_hr)

        # 3. loss, Gaussian with small variance to approximate Dirac delta function of LR.
        # for the second term, using small log-variance may lead to svd problem, for both exp and tanh version
        objective = logdet + Basic.GaussianDiag.logp(lr, -torch.ones_like(lr)*6, fake_lr_from_hr)

        nll = ((-objective) / float(np.log(2.) * pixels)).mean()

        return torch.clamp(fake_lr_from_hr, 0, 1), nll

    # lr+z->hr
    def reverse_flow_diracLR(self, lr, z, u, eps_std, training=True):

        # lr+z->hr
        fake_hr = self.flow(z=lr, u=u, eps_std=eps_std, reverse=True, training=training)

        return torch.clamp(fake_hr, 0, 1)
