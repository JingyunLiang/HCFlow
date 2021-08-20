# base model for HCFlow
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from models.modules.loss import GANLoss
from .base_model import BaseModel
import utils.util as util

logger = logging.getLogger('base')


class HCFlowSRModel(BaseModel):
    def __init__(self, opt, step):
        super(HCFlowSRModel, self).__init__(opt)
        self.opt = opt

        self.hr_size = util.opt_get(opt, ['datasets', 'train', 'GT_size'], 160)
        self.lr_size = self.hr_size // opt['scale']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt, step).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        if self.is_train:
            train_opt = opt['train']
            self.netG.train()

            # NLL weight
            self.l_nll_w = train_opt['nll_weight']
            self.eps_std_reverse = train_opt['eps_std_reverse']

            # HR pixel loss
            if train_opt['pixel_weight_hr'] > 0:
                loss_type = train_opt['pixel_criterion_hr']
                if loss_type == 'l1':
                    self.cri_pix_hr = nn.L1Loss().to(self.device)
                elif loss_type == 'l2':
                    self.cri_pix_hr = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
                self.l_pix_w_hr = train_opt['pixel_weight_hr']
            else:
                logger.info('Remove HR pixel loss.')
                self.cri_pix_hr = None

            # HR feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']

                # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

            # HR GAN loss
            # put here to be compatible with PSNR version
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            if train_opt['gan_weight'] > 0:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_w = train_opt['gan_weight']

                # define GAN Discriminator
                self.netD = networks.define_D(opt).to(self.device)
                if opt['dist']:
                    self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
                else:
                    self.netD = DataParallel(self.netD)
                self.netD.train()
            else:
                logger.info('Remove GAN loss.')
                self.cri_gan = None

            # gradient clip & norm
            self.max_grad_clip = util.opt_get(train_opt, ['max_grad_clip'])
            self.max_grad_norm = util.opt_get(train_opt, ['max_grad_norm'])

            # optimizers
            # G
            wd_G = util.opt_get(train_opt, ['weight_decay_G'], 0)
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                # if v.requires_grad and ('additional_flow_steps' in k or 'additional_feature_steps' in k): # fixmainflow
                # if v.requires_grad and ('additional_flow_steps' in k): # fix mainflowRRDB
                    optim_params.append(v)
                else:
                    v.requires_grad = False
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # D
            if self.cri_gan:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                    weight_decay=wd_D,
                                                    betas=(train_opt['beta1_D'], train_opt['beta2_D']))
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                print('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        # val
        if 'val' in opt:
            self.heats = opt['val']['heats']
            self.n_sample = opt['val']['n_sample']
            self.sr_mode = opt['val']['sr_mode']

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
        else:
            self.real_H = None

    def optimize_parameters(self, step):
        # special initialization for actnorm; don't initialize when fine-tuning
        if step < self.opt['network_G']['act_norm_start_step'] and not (self.cri_pix_hr or self.cri_gan):
            self.set_actnorm_init(inited=False)

        # (1) G
        fake_H = None
        if (step % self.D_update_ratio == 0 and step > self.D_init_iters) or (not self.cri_gan):
            # normal flow
            l_g_total = 0

            _, nll = self.netG(hr=self.real_H, lr=self.var_L, u=None, reverse=False)
            nll = self.l_nll_w * nll.sum()
            self.log_dict['nll'] = nll.item()
            if not torch.isnan(nll).any():
                l_g_total += nll
            if l_g_total != 0:
                l_g_total.backward()
                self.gradient_clip()
                self.optimizer_G.step()

            # reverse flow (optimize NLL loss and HR loss seperately (takes less memory and more time, slightly better results))
            self.optimizer_G.zero_grad()
            if self.cri_pix_hr:
                fake_H = self.netG(lr=self.var_L, z=None, u=None, eps_std=0.0, reverse=True)
                l_g_total = 0
                if not torch.isnan(fake_H).any():
                    # pixel loss
                    l_g_pix_hr = self.l_pix_w_hr * self.cri_pix_hr(fake_H, self.real_H)
                    l_g_total += l_g_pix_hr
                    self.log_dict['l_g_pix_hr'] = l_g_pix_hr.item()
                if l_g_total != 0:
                    l_g_total.backward()
                    self.gradient_clip()
                    self.optimizer_G.step()


            ########################

            if self.cri_gan or self.cri_fea:
                self.optimizer_G.zero_grad()
                fake_H = self.netG(lr=self.var_L, z=None, u=None, eps_std=self.eps_std_reverse, reverse=True)
                l_g_fea_gan = 0

                # feature loss
                if self.cri_fea:
                    real_fea = self.netF(self.real_H).detach()
                    fake_fea = self.netF(fake_H)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_fea_gan += l_g_fea
                    self.log_dict['l_g_fea'] = l_g_fea.item()

                # gan loss
                if self.cri_gan:
                    for p in self.netD.parameters():
                        p.requires_grad = False

                    pred_g_fake = self.netD(fake_H)
                    if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgangp']:
                        l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                    elif self.opt['train']['gan_type'] == 'ragan':
                        pred_d_real = self.netD(self.real_H).detach()
                        l_g_gan = self.l_gan_w * ( self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                                   self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    l_g_fea_gan += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                if not torch.isnan(l_g_fea_gan):
                    l_g_fea_gan.backward()
                    self.gradient_clip()
                    self.optimizer_G.step()

        # (2) D
        if self.cri_gan:
            self.optimizer_G.zero_grad() # can help save memory

            for p in self.netD.parameters():
                p.requires_grad = True

            # initialize D
            if fake_H is None:
                with torch.no_grad():
                    fake_H = self.netG(lr=self.var_L, z=None, u=None, eps_std=self.eps_std_reverse, reverse=True)

            self.optimizer_D.zero_grad()
            pred_d_real = self.netD(self.real_H)
            pred_d_fake = self.netD(fake_H.detach())  # detach to avoid BP to G
            if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgangp']:
                l_d_real = self.cri_gan(pred_d_real, True)
                l_d_fake = self.cri_gan(pred_d_fake, False)
                l_d_total = l_d_real + l_d_fake
            elif self.opt['train']['gan_type'] == 'ragan':
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                l_d_total = (l_d_real + l_d_fake) / 2

            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

            if not torch.isnan(l_d_total):
                l_d_total.backward()
                self.optimizer_D.step()

    def gradient_clip(self):
        # gradient clip & norm, is not used in SRFlow
        if self.max_grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.max_grad_clip)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)

    def test(self):
        self.netG.eval()
        self.fake_H = {}

        with torch.no_grad():
            if self.real_H is None:
                nll = torch.zeros(1)
            else:
                # hr->lr+z, calculate nll
                self.fake_L_from_H, nll = self.netG(hr=self.real_H, lr=self.var_L, u=None, reverse=False, training=False)

            # lr+z->hr
            for heat in self.heats:
                for sample in range(self.n_sample):
                    # z = self.get_z(heat, seed=1, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                    self.fake_H[(heat, sample)] = self.netG(lr=self.var_L,
                                  z=None, u=None, eps_std=heat, reverse=True, training=False)

        self.netG.train()

        return nll.mean().item()

    def get_encode_nll(self, lq, hr, y_label=None):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(hr=hr, lr=lq, reverse=False, y_label=y_label)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None, y_label=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses, y_label=y_label)[0]

    def get_encode_z(self, lq, hr, epses=None, add_gt_noise=True, y_label=None):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(hr=hr, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise, y_label=y_label)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, hr, epses=None, y_label=None):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(hr=hr, lr=lq, reverse=False, epses=epses, y_label=y_label)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None, y_label=None):
        self.netG.eval()

        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape,
                       y_label=None) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses, y_label=None)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None, y_label=None):
        if y_label is None:
            pass
        if seed: torch.manual_seed(seed)
        if util.opt_get(self.opt, ['network_G', 'flowLR', 'splitOff', 'enable']):
            C, H, W = lr_shape[1], lr_shape[2], lr_shape[3]

            size = (batch_size, C, H, W)
            if heat == 0:
                z = torch.zeros(size)
            else:
                z = torch.normal(mean=0, std=heat, size=size)
        else:
            L = util.opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z.to(self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()

        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
            out_dict['LQ_fromH'] = self.fake_L_from_H.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        if self.is_train:
            # Discriminator
            if self.cri_gan:
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                        DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                     self.netD.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

            # F, Perceptual Network
            if self.cri_fea:
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        # resume training automatically if resume_state=='auto'
        _, get_resume_model_path = util.get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            logger.info('Automatically loading model for G [{:s}] ...'.format(get_resume_model_path))
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            self.set_actnorm_init(inited=True)

            if self.is_train and self.cri_gan:
                get_resume_model_path = get_resume_model_path.replace('_G.pth', '_D.pth')
                logger.info('Automatically loading model for D [{:s}] ...'.format(get_resume_model_path))
                self.load_network(get_resume_model_path, self.netD, strict=True, submodule=None)
            return

        # resume training according to given paths (pretrain path has been overrided by resume path)
        if self.opt.get('path') is not None:
            load_path_G = self.opt['path']['pretrain_model_G']
            if load_path_G is not None:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True))

                self.set_actnorm_init(inited=True)

        if self.is_train and self.cri_gan:
            load_path_D = self.opt['path']['pretrain_model_D']
            if load_path_D is not None:
                logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD, self.opt['path'].get('strict_load', True))

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        if self.cri_gan:
            self.save_network(self.netD, 'D', iter_label)

    def set_actnorm_init(self, inited=True):
        for name, m in self.netG.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    # from glow
    def generate_z(self, img):
        self.eval()
        B = self.hparams.Train.batch_size
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z,_, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in tqdm(range(0, N, B)):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz
