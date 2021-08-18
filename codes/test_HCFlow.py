import os.path
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
import utils.util as util
from utils.imresize import imresize
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import lpips

os.environ['CUDA_HOME'] = '/scratch_net/rind/cuda-11.0'
os.environ['LD_LIBRARY_PATH'] = '/scratch_net/rind/cuda-11.0/lib64'
os.environ['PATH'] = '/scratch_net/rind/cuda-11.0/bin'

#### options
parser = argparse.ArgumentParser() # test_SR_CelebA_8X_HCFlow test_SR_DF2K_4X_HCFlow test_Rescaling_DF2K_4X_HCFlow
parser.add_argument('--opt', type=str, default='options/test/test_SR_CelebA_8X_HCFlow.yml', help='Path to options YMAL file.')
parser.add_argument('--save_kernel', action='store_true', default=False, help='Save Kernel Esimtation.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)
device_id = torch.cuda.current_device()

#### mkdir and logger
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key and 'load_submodule' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

# set random seed
util.set_random_seed(0)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\n\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    idx = 0
    psnr_dict={} # for HR image
    ssim_dict={}
    psnr_y_dict = {}
    ssim_y_dict = {}
    bic_hr_psnr_dict={} # for bic(HR)
    bic_hr_ssim_dict={}
    bic_hr_psnr_y_dict = {}
    bic_hr_ssim_y_dict = {}
    lpips_dict = {}
    diversity_dict = {} # pixel-wise variance
    avg_lr_psnr = 0.0 # for generated LR image
    avg_lr_ssim = 0.0
    avg_lr_psnr_y = 0.0
    avg_lr_ssim_y = 0.0
    avg_nll = 0.0

    for test_data in test_loader:
        idx += 1

        real_image = True if test_loader.dataset.opt['dataroot_GT'] is None else False
        generate_online = True if test_loader.dataset.opt['dataroot_GT'] is not None and test_loader.dataset.opt[
            'dataroot_LQ'] is None else False
        img_path = test_data['LQ_path'][0] if real_image else test_data['GT_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.feed_data(test_data)
        nll = model.test()
        avg_nll += nll
        visuals = model.get_current_visuals()

        # calculate PSNR for LR
        gt_img_lr = util.tensor2img(visuals['LQ'])
        sr_img_lr = util.tensor2img(visuals['LQ_fromH'])
        # save_img_path = os.path.join(dataset_dir, 'LR_{:s}_{:.1f}_{:d}.png'.format(img_name, 1.0, 0))
        # util.save_img(sr_img_lr, save_img_path)
        gt_img_lr = gt_img_lr / 255.
        sr_img_lr = sr_img_lr / 255.

        lr_psnr, lr_ssim, lr_psnr_y, lr_ssim_y = util.calculate_psnr_ssim(gt_img_lr, sr_img_lr, 0)
        avg_lr_psnr += lr_psnr
        avg_lr_ssim += lr_ssim
        avg_lr_psnr_y += lr_psnr_y
        avg_lr_ssim_y += lr_ssim_y

        # deal with real-world data (just save)
        if real_image:
            for heat in opt['val']['heats']:
                for sample in range(opt['val']['n_sample']):
                    sr_img = util.tensor2img(visuals['SR', heat, sample])

                    # deal with the image margins for real images
                    if opt['scale'] == 4:
                        real_crop = 3
                    elif opt['scale'] == 2:
                        real_crop = 6
                    elif opt['scale'] == 1:
                        real_crop = 11
                    assert real_crop * opt['scale'] * 2 > opt['kernel_size']
                    sr_img = sr_img[real_crop * opt['scale']:-real_crop * opt['scale'],
                             real_crop * opt['scale']:-real_crop * opt['scale'], :]
                    save_img_path = os.path.join(dataset_dir, 'SR_{:s}_{:.1f}_{:d}.png'.
                                             format(img_name, heat, sample))
                    util.save_img(sr_img, save_img_path)

        # deal with synthetic data (calculate psnr and save)
        else:
            for heat in opt['val']['heats']:
                psnr = 0.0
                ssim = 0.0
                psnr_y = 0.0
                ssim_y = 0.0
                lpips_value = 0.0
                bic_hr_psnr = 0.0
                bic_hr_ssim = 0.0
                bic_hr_psnr_y = 0.0
                bic_hr_ssim_y = 0.0

                sr_img_list =[]
                for sample in range(opt['val']['n_sample']):
                    gt_img = visuals['GT']
                    sr_img = visuals['SR', heat, sample]
                    sr_img_list.append(sr_img.unsqueeze(0)*255)
                    lpips_dict[(idx, heat, sample)] = float(loss_fn_alex(2 * gt_img.to('cuda') - 1, 2 * sr_img.to('cuda') - 1).cpu())
                    lpips_value += lpips_dict[(idx, heat, sample)]

                    gt_img = util.tensor2img(gt_img)  # uint8
                    sr_img = util.tensor2img(sr_img)  # uint8
                    suffix = opt['suffix']
                    if suffix:
                        save_img_path = os.path.join(dataset_dir, 'SR_{:s}_{:.1f}_{:d}_{:s}.png'.format(img_name, heat, sample, suffix))
                    else:
                        save_img_path = os.path.join(dataset_dir, 'SR_{:s}_{:.1f}_{:d}.png'.format(img_name, heat, sample))
                    util.save_img(sr_img, save_img_path)

                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    bic_hr_gt_img = imresize(gt_img, 1 / opt['scale'])
                    bic_hr_sr_img = imresize(sr_img, 1 / opt['scale'])

                    psnr_dict[(idx, heat, sample)], ssim_dict[(idx, heat, sample)], \
                    psnr_y_dict[(idx, heat, sample)], ssim_y_dict[(idx, heat, sample)] = util.calculate_psnr_ssim(gt_img, sr_img, crop_border)
                    psnr += psnr_dict[(idx, heat, sample)]
                    ssim += ssim_dict[(idx, heat, sample)]
                    psnr_y += psnr_y_dict[(idx, heat, sample)]
                    ssim_y += ssim_y_dict[(idx, heat, sample)]
                    bic_hr_psnr_dict[(idx, heat, sample)], bic_hr_ssim_dict[(idx, heat, sample)], \
                    bic_hr_psnr_y_dict[(idx, heat, sample)], bic_hr_ssim_y_dict[(idx, heat, sample)] = util.calculate_psnr_ssim(bic_hr_gt_img, bic_hr_sr_img, 0)
                    bic_hr_psnr += bic_hr_psnr_dict[(idx, heat, sample)]
                    bic_hr_ssim += bic_hr_ssim_dict[(idx, heat, sample)]
                    bic_hr_psnr_y += bic_hr_psnr_y_dict[(idx, heat, sample)]
                    bic_hr_ssim_y += bic_hr_ssim_y_dict[(idx, heat, sample)]


                # mean pixel-wise variance
                psnr /= opt['val']['n_sample']
                ssim /= opt['val']['n_sample']
                psnr_y /= opt['val']['n_sample']
                ssim_y /= opt['val']['n_sample']
                diversity_dict[(idx, heat)] = float(torch.cat(sr_img_list, 0).std([0]).mean().cpu())
                lpips_value /= opt['val']['n_sample']
                bic_hr_psnr /= opt['val']['n_sample']
                bic_hr_ssim /= opt['val']['n_sample']
                bic_hr_psnr_y /= opt['val']['n_sample']
                bic_hr_ssim_y /= opt['val']['n_sample']


                logger.info('{:20s} ({}samples),heat:{:.1f}) '
                            'HR:PSNR/SSIM/PSNR_Y/SSIM_Y/LPIPS/Diversity: {:.2f}/{:.4f}/{:.2f}/{:.4f}/{:.4f}/{:.4f}, '
                            'bicHR:PSNR/SSIM/PSNR_Y/SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}, '
                            'LR:PSNR/SSIM/PSNR_Y/SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}, NLL: {:.4f}'.format(
                            img_name, opt['val']['n_sample'], heat,
                            psnr, ssim, psnr_y, ssim_y, lpips_value, diversity_dict[(idx, heat)],
                            bic_hr_psnr, bic_hr_ssim, bic_hr_psnr_y, bic_hr_ssim_y,
                            lr_psnr, lr_ssim, lr_psnr_y, lr_ssim_y, nll))

    # Average PSNR/SSIM results
    avg_lr_psnr /= idx
    avg_lr_ssim /= idx
    avg_lr_psnr_y /= idx
    avg_lr_ssim_y /= idx
    avg_nll = avg_nll / idx

    if real_image:
        logger.info('----{} ({} images), avg LR PSNR/SSIM/PSNR_K/LR_SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}\n'.format(test_set_name, idx, avg_lr_psnr, avg_lr_ssim, avg_lr_psnr_y, avg_lr_ssim_y))
    else:
        logger.info('-------------------------------------------------------------------------------------')
        for heat in opt['val']['heats']:
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_psnr_y = 0.0
            avg_ssim_y = 0.0
            avg_lpips = 0.0
            avg_diversity = 0.0
            avg_bic_hr_psnr = 0.0
            avg_bic_hr_ssim = 0.0
            avg_bic_hr_psnr_y = 0.0
            avg_bic_hr_ssim_y = 0.0

            for iidx in range(1, idx+1):
                for sample in range(opt['val']['n_sample']):
                    avg_psnr += psnr_dict[(iidx, heat, sample)]
                    avg_ssim += ssim_dict[(iidx, heat, sample)]
                    avg_psnr_y += psnr_y_dict[(iidx, heat, sample)]
                    avg_ssim_y += ssim_y_dict[(iidx, heat, sample)]
                    avg_lpips += lpips_dict[(iidx, heat, sample)]
                    avg_bic_hr_psnr += bic_hr_psnr_dict[(iidx, heat, sample)]
                    avg_bic_hr_ssim += bic_hr_ssim_dict[(iidx, heat, sample)]
                    avg_bic_hr_psnr_y += bic_hr_psnr_y_dict[(iidx, heat, sample)]
                    avg_bic_hr_ssim_y += bic_hr_ssim_y_dict[(iidx, heat, sample)]
                avg_diversity += diversity_dict[(iidx, heat)]

            avg_psnr = avg_psnr / idx / opt['val']['n_sample']
            avg_ssim = avg_ssim / idx / opt['val']['n_sample']
            avg_psnr_y = avg_psnr_y / idx / opt['val']['n_sample']
            avg_ssim_y = avg_ssim_y / idx / opt['val']['n_sample']
            avg_lpips = avg_lpips / idx  / opt['val']['n_sample']
            avg_diversity = avg_diversity / idx
            avg_bic_hr_psnr = avg_bic_hr_psnr / idx / opt['val']['n_sample']
            avg_bic_hr_ssim = avg_bic_hr_ssim / idx / opt['val']['n_sample']
            avg_bic_hr_psnr_y = avg_bic_hr_psnr_y / idx / opt['val']['n_sample']
            avg_bic_hr_ssim_y = avg_bic_hr_ssim_y / idx / opt['val']['n_sample']

            # log
            logger.info(opt['path']['pretrain_model_G'])
            logger.info('----{} ({}images,{}samples,heat:{:.1f}) '
                        'average HR:PSNR/SSIM/PSNR_Y/SSIM_Y/LPIPS/Diversity: {:.2f}/{:.4f}/{:.2f}/{:.4f}/{:.4f}/{:.4f}, '
                        'bicHR:PSNR/SSIM/PSNR_Y/SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}, '
                        'LR:PSNR/SSIM/PSNR_Y/SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}, NLL: {:.4f}'.format(
                    test_set_name, idx, opt['val']['n_sample'], heat,
                avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y, avg_lpips, avg_diversity,
                avg_bic_hr_psnr, avg_bic_hr_ssim, avg_bic_hr_psnr_y, avg_bic_hr_ssim_y,
                avg_lr_psnr, avg_lr_ssim, avg_lr_psnr_y, avg_lr_ssim_y, avg_nll))
