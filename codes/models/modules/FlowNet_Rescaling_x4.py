import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from utils.util import opt_get
from models.modules import Basic
from models.modules.FlowStep import FlowStep
from models.modules.ConditionalFlow import ConditionalFlow

class FlowNet(nn.Module):
    def __init__(self, image_shape, opt=None):
        assert image_shape[2] == 1 or image_shape[2] == 3
        super().__init__()
        H, W, self.C = image_shape
        self.opt = opt
        self.L = opt_get(opt, ['network_G', 'flowDownsampler', 'L'])
        self.K = opt_get(opt, ['network_G', 'flowDownsampler', 'K'])
        if isinstance(self.K, int): self.K = [self.K] * (self.L + 1)

        squeeze = opt_get(self.opt, ['network_G', 'flowDownsampler', 'squeeze'], 'checkerboard')
        n_additionalFlowNoAffine = opt_get(self.opt, ['network_G', 'flowDownsampler', 'additionalFlowNoAffine'], 0)
        flow_permutation = opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_permutation'], 'invconv')
        flow_coupling = opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_coupling'], 'Affine')
        cond_channels = opt_get(self.opt, ['network_G', 'flowDownsampler', 'cond_channels'], None)
        enable_splitOff = opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'enable'], False)
        after_splitOff_flowStep = opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'after_flowstep'], 0)
        if isinstance(after_splitOff_flowStep, int): after_splitOff_flowStep = [after_splitOff_flowStep] * (self.L + 1)

        # construct flow
        self.layers = nn.ModuleList()
        self.output_shapes = []

        for level in range(self.L):
            # 1. Squeeze
            if squeeze == 'checkerboard':
                self.layers.append(Basic.SqueezeLayer(factor=2)) # may need a better way for squeezing
            elif squeeze == 'haar':
                self.layers.append(Basic.HaarDownsampling(channel_in=self.C))

            self.C, H, W = self.C * 4, H // 2, W // 2
            self.output_shapes.append([-1, self.C, H, W])

            # 2. main FlowSteps (uncodnitional flow)
            for k in range(self.K[level]-after_splitOff_flowStep[level]):
                self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels,
                                                       flow_permutation=flow_permutation,
                                                       flow_coupling=flow_coupling,
                                                       LRvsothers=True if k%2==0 else False,
                                                       opt=opt['network_G']['flowDownsampler']))
                self.output_shapes.append([-1, self.C, H, W])

            # 3. additional FlowSteps (split + conditional flow)
            if enable_splitOff:
                if level == 0:
                    self.layers.append(Basic.Split(num_channels_split=self.C // 2 if level < self.L-1 else 3, level=level))
                    self.level0_condFlow = ConditionalFlow(num_channels=self.C,
                                                    num_channels_split=self.C // 2 if level < self.L-1 else 3,
                                                    n_flow_step=after_splitOff_flowStep[level],
                                                    opt=opt['network_G']['flowDownsampler']['splitOff'],
                                                          num_levels_condition=1, SR=False)
                elif level == 1:
                    self.layers.append(Basic.Split(num_channels_split=self.C // 2 if level < self.L-1 else 3, level=level))
                    self.level1_condFlow = (ConditionalFlow(num_channels=self.C,
                                                        num_channels_split=self.C // 2 if level < self.L-1 else 3,
                                                        n_flow_step=after_splitOff_flowStep[level],
                                                        opt=opt['network_G']['flowDownsampler']['splitOff'],
                                                        num_levels_condition=0, SR=False))

                self.C = self.C // 2 if level < self.L-1 else 3
                self.output_shapes.append([-1, self.C, H, W])


        self.H = H
        self.W = W
        self.scaleH = image_shape[0] / H
        self.scaleW = image_shape[1] / W
        print('shapes:', self.output_shapes)

    def forward(self, hr=None, z=None, u=None, eps_std=None, logdet=None, reverse=False, training=True):
        if not reverse:
            return self.normal_flow(hr, u=u, logdet=logdet, training=training)
        else:
            return self.reverse_flow(z, u=u, eps_std=eps_std, training=training)


    '''
    hr->y1+z1->y2+z2
    '''
    def normal_flow(self, z, u=None, logdet=None, training=True):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, FlowStep):
                z, _ = layer(z, u, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer) or isinstance(layer, Basic.HaarDownsampling):
                z, _ = layer(z, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.Split):
                if layer.level == 0:
                    z, a1 = layer(z, reverse=False)
                    y1 = z.clone()
                elif layer.level == 1:
                    z, a2 = layer(z, reverse=False)
                    fake_z2, conditional_feature2 = self.level1_condFlow(a2, z, logdet=logdet, reverse=False, training=training)

                    conditional_feature1 = torch.cat([y1, F.interpolate(conditional_feature2, scale_factor=2, mode='nearest')],1)
                    fake_z1, _ = self.level0_condFlow(a1, conditional_feature1, logdet=logdet, reverse=False, training=training)

        return z, fake_z1, fake_z2

    '''
    y2+z2->y1+z1->hr
    '''
    def reverse_flow(self, z, u=None, eps_std=None, training=True):
        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            if isinstance(layer, FlowStep):
                z, _ = layer(z, u, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer) or isinstance(layer, Basic.HaarDownsampling):
                z, _ = layer(z, reverse=True)
            elif isinstance(layer, Basic.Split):
                if layer.level == 1:
                    a2, conditional_feature2 = self.level1_condFlow(None, z, eps_std=eps_std, reverse=True, training=training)
                    z = layer(z, a2, reverse=True)
                elif layer.level == 0:
                    conditional_feature1 = torch.cat([z, F.interpolate(conditional_feature2, scale_factor=2, mode='nearest')],1)
                    a1, _ = self.level0_condFlow(None, conditional_feature1, eps_std=eps_std, reverse=True, training=training)
                    z = layer(z, a1, reverse=True)


        return z

