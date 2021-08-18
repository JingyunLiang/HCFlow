import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import thops
from utils.util import opt_get
from models.modules.Basic import Conv2d, Conv2dZeros, GaussianDiag, DenseBlock, RRDB, FCN
from models.modules.FlowStep import FlowStep

import functools
import models.modules.module_util as mutil


class ConditionalFlow(nn.Module):
    def __init__(self, num_channels, num_channels_split, n_flow_step=0, opt=None, num_levels_condition=0, SR=True):
        super().__init__()
        self.SR = SR

        # number of levels of RRDB features. One level of conditional feature is enough for image rescaling
        num_features_condition = 2 if self.SR else 1

        # feature extraction
        RRDB_nb = opt_get(opt, ['RRDB_nb'], [5, 5])
        RRDB_nf = opt_get(opt, ['RRDB_nf'], 64)
        RRDB_gc = opt_get(opt, ['RRDB_gc'], 32)
        RRDB_f = functools.partial(RRDB, nf=RRDB_nf, gc=RRDB_gc)
        self.conv_first = nn.Conv2d(num_channels_split + RRDB_nf*num_features_condition*num_levels_condition, RRDB_nf, 3, 1, 1, bias=True)
        self.RRDB_trunk0 = mutil.make_layer(RRDB_f, RRDB_nb[0])
        self.RRDB_trunk1 = mutil.make_layer(RRDB_f, RRDB_nb[1])
        self.trunk_conv1 = nn.Conv2d(RRDB_nf, RRDB_nf, 3, 1, 1, bias=True)

        # conditional flow
        self.additional_flow_steps = nn.ModuleList()
        for k in range(n_flow_step):
            self.additional_flow_steps.append(FlowStep(in_channels=num_channels-num_channels_split,
                                                                  cond_channels=RRDB_nf*num_features_condition,
                                                                  flow_permutation=opt['flow_permutation'],
                                                                  flow_coupling=opt['flow_coupling'], opt=opt))

        self.f = Conv2dZeros(RRDB_nf*num_features_condition, (num_channels-num_channels_split)*2)


    def forward(self, z, u, eps_std=None, logdet=0., reverse=False, training=True):
        # for image SR
        if self.SR:
            if not reverse:
                conditional_feature = self.get_conditional_feature_SR(u)

                for layer in self.additional_flow_steps:
                    z, logdet = layer(z, u=conditional_feature, logdet=logdet, reverse=False)

                h = self.f(conditional_feature)
                mean, logs = thops.split_feature(h, "cross")
                logdet += GaussianDiag.logp(mean, logs, z)

                return logdet, conditional_feature

            else:
                conditional_feature = self.get_conditional_feature_SR(u)

                h = self.f(conditional_feature)
                mean, logs = thops.split_feature(h, "cross")
                z = GaussianDiag.sample(mean, logs, eps_std)

                for layer in reversed(self.additional_flow_steps):
                    z, _ = layer(z, u=conditional_feature, reverse=True)

                return z, logdet, conditional_feature
        else:
            # for image rescaling
            if not reverse:
                conditional_feature = self.get_conditional_feature_Rescaling(u)

                for layer in self.additional_flow_steps:
                    z, logdet = layer(z, u=conditional_feature, logdet=logdet, reverse=False)

                h = self.f(conditional_feature)
                mean, scale = thops.split_feature(h, "cross")
                logscale = 0.318 * torch.atan(2 * scale)
                z = (z - mean) * torch.exp(-logscale)

                return z, conditional_feature

            else:
                conditional_feature = self.get_conditional_feature_Rescaling(u)

                h = self.f(conditional_feature)
                mean, scale = thops.split_feature(h, "cross")
                logscale = 0.318 * torch.atan(2 * scale)
                z = GaussianDiag.sample(mean, logscale, eps_std)

                for layer in reversed(self.additional_flow_steps):
                    z, _ = layer(z, u=conditional_feature, reverse=True)

                return z, conditional_feature


    def get_conditional_feature_SR(self, u):
        u_feature_first = self.conv_first(u)
        u_feature1 = self.RRDB_trunk0(u_feature_first)
        u_feature2 = self.trunk_conv1(self.RRDB_trunk1(u_feature1)) + u_feature_first

        return torch.cat([u_feature1, u_feature2], 1)

    def get_conditional_feature_Rescaling(self, u):
        u_feature_first = self.conv_first(u)
        u_feature = self.trunk_conv1(self.RRDB_trunk1(self.RRDB_trunk0(u_feature_first))) + u_feature_first

        return u_feature


