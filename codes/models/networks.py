import importlib
import logging
import torch
import models.modules.discriminator_vgg_arch as SRGAN_arch

logger = logging.getLogger('base')


def find_model_using_name(model_name):
    model_filename = "models.modules." + model_name + "_arch"
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace('_Net', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s." % (
                model_filename, target_model_name))
        exit(0)

    return model

def define_Flow(opt, step):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    Arch = find_model_using_name(which_model)
    netG = Arch(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt['scale'], K=opt_net['flow']['K'], opt=opt, step=step)
    return netG

def define_G(opt, step):
    which_model = opt['network_G']['which_model_G']

    Arch = find_model_using_name(which_model)
    netG = Arch(opt=opt, step=step)
    return netG

#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_vgg_160':
        netD = SRGAN_arch.Discriminator_VGG_160(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'PatchGANDiscriminator':
        netD = SRGAN_arch.PatchGANDiscriminator(in_nc=opt_net['in_nc'], ndf=opt_net['ndf'], n_layers=opt_net['n_layers'],)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
