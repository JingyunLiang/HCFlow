import os
import math
import argparse
import random
import logging
import numpy as np
import torch
from data.data_sampler import DistIterSampler, EnlargedSampler
from data.util import bgr2ycbcr

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.dist_util import get_dist_info, init_dist

import socket
import getpass
import lpips



def main():
    #### setup options
    parser = argparse.ArgumentParser() # train_SR_CelebA_8X_HCFlow train_SR_DF2K_4X_HCFlow train_Rescaling_DF2K_4X_HCFlow
    parser.add_argument('--opt', type=str, default='options/train/train_SR_CelebA_8X_HCFlow.yml',
                        help='Path to option YMAL file of MANet.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--job_id', type=str, default=0)
    parser.add_argument('--job_path', type=str, default='')
    args = parser.parse_args()
    opt = option.parse(args.opt, args.gpu_ids, is_train=True)
    device_id = torch.cuda.current_device()

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    print(torch.__version__)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = util.get_resume_paths(opt)
        if resume_state_path is None:
            resume_state = None
        else:
            # distributed resuming: all load into default GPU
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path,
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # override model pretrain path with resume path
    else:
        resume_state = None

    #### mkdir and loggers
    # normal training (rank -1) OR distributed training (rank (gpu id) 0-7)
    if opt['rank'] <= 0:
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train{}_'.format(args.job_id) + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val{}_'.format(args.job_id) + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(getpass.getuser(), socket.gethostname(),
                                                                   opt['gpu_ids'], args.job_id, args.job_path))
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # symlink the code/working dir
    try:
        os.symlink(args.job_path.replace('/cluster/home/{}'.format(getpass.getuser()),'/scratch/e_home'),
                   opt['path']['experiments_root']+'/{}'.format(os.path.basename(args.job_path)))
    except:
        pass
    try:
        os.symlink(args.job_path.replace('/cluster/home/{}'.format(getpass.getuser()),'/scratch/e_home')+'/options/train',
                   opt['path']['experiments_root']+'/{}'.format(args.job_id))
    except:
        pass

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, opt['world_size'], opt['rank'], dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if opt['rank'] <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if opt['rank'] <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training

    # logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            #### update learning rate, schedulers
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}:{:.4e} '.format(k, v)
                    # tensorboard logger, but sometimes cause dead
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if opt['rank'] <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if opt['rank'] <= 0:
                    logger.info(message)

            #### save models and training states before validation
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if opt['rank'] <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

            # validation
            if (current_step % opt['train']['val_freq'] == 0 ) and opt['rank'] <= 0:
                idx = 0
                psnr_dict = {}
                psnr_y_dict = {}
                loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
                lpips_dict = {}
                diversity_dict = {} # pixel-wise variance
                avg_lr_psnr_y = 0.0
                avg_nll = 0.0

                for _, val_data in enumerate(val_loader):
                    idx += 1

                    model.feed_data(val_data)
                    avg_nll += model.test()
                    visuals = model.get_current_visuals()

                    # create dir for each iteration
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], str(current_step))
                    util.mkdir(img_dir)

                    # calculate LR psnr
                    gt_img_lr = util.tensor2img(visuals['LQ'])
                    sr_img_lr = util.tensor2img(visuals['LQ_fromH'])
                    gt_img_lr = gt_img_lr / 255.
                    sr_img_lr = sr_img_lr / 255.

                    _, _, lr_psnr_y, _ = util.calculate_psnr_ssim(gt_img_lr, sr_img_lr, 0)
                    avg_lr_psnr_y += lr_psnr_y

                    # deal with sr images
                    for heat in opt['val']['heats']:
                        sr_img_list =[]
                        for sample in range(opt['val']['n_sample']):
                            gt_img = visuals['GT']
                            sr_img = visuals['SR', heat, sample]
                            sr_img_list.append(sr_img.unsqueeze(0)*255)
                            lpips_dict[(idx, heat, sample)] = float(loss_fn_alex(2 * gt_img.to('cuda') - 1, 2 * sr_img.to('cuda') - 1).cpu())

                            gt_img = util.tensor2img(gt_img)  # uint8
                            sr_img = util.tensor2img(sr_img)  # uint8
                            save_img_path = os.path.join(img_dir, 'SR_{:s}_{:.1f}_{:d}_{:d}.png'.format(img_name, heat, sample, current_step))
                            util.save_img(sr_img, save_img_path)

                            gt_img = gt_img / 255.
                            sr_img = sr_img / 255.

                            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
                            psnr_dict[(idx, heat, sample)], ssim, psnr_y_dict[(idx, heat, sample)], ssim_y = util.calculate_psnr_ssim(gt_img, sr_img, crop_border)

                        # mean pixel-wise variance
                        diversity_dict[(idx, heat)] = float(torch.cat(sr_img_list, 0).std([0]).mean().cpu())

                # log
                logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(getpass.getuser(), socket.gethostname(),
                                                                   opt['gpu_ids'], args.job_id, args.job_path))
                logger.info('# {}, Validation (<epoch:{:3d}, iter:{:8d}>)'.format(opt['name'], epoch, current_step))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('# {}, Validation (<epoch:{:3d}, iter:{:8d}>)'.format(opt['name'], epoch, current_step))

                avg_lr_psnr_y = avg_lr_psnr_y / idx
                avg_nll = avg_nll / idx
                for heat in opt['val']['heats']:
                    avg_psnr = 0.0
                    avg_psnr_y = 0.0
                    avg_lpips = 0.0
                    avg_diversity = 0.0

                    for iidx in range(1, idx+1):
                        for sample in range(opt['val']['n_sample']):
                            avg_psnr += psnr_dict[(iidx, heat, sample)]
                            avg_psnr_y += psnr_y_dict[(iidx, heat, sample)]
                            avg_lpips += lpips_dict[(iidx, heat, sample)]
                        avg_diversity += diversity_dict[(iidx, heat)]

                    avg_psnr = avg_psnr / idx / opt['val']['n_sample']
                    avg_psnr_y = avg_psnr_y / idx / opt['val']['n_sample']
                    avg_lpips = avg_lpips / idx  / opt['val']['n_sample']
                    avg_diversity = avg_diversity / idx

                    # log
                    logger.info('({}samples,heat:{:.1f}) PSNR/PSNR_Y/LPIPS/Diversity: {:.2f}/{:.2f}/{:.4f}/{:.4f}, LR_PSNR_Y: {:.2f}, NLL: {:.4f}'.format(
                            opt['val']['n_sample'], heat, avg_psnr, avg_psnr_y, avg_lpips, avg_diversity, avg_lr_psnr_y, avg_nll))
                    logger_val.info('({}samples,heat:{:.1f}) PSNR/PSNR_Y/LPIPS/Diversity: {:.2f}/{:.2f}/{:.4f}/{:.4f}, LR_PSNR_Y: {:.2f}, NLL: {:.4f}'.format(
                            opt['val']['n_sample'], heat, avg_psnr, avg_psnr_y, avg_lpips, avg_diversity, avg_lr_psnr_y, avg_nll))


                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr_{:.1f}'.format(heat), avg_psnr, current_step)
                        tb_logger.add_scalar('psnr_y_{:.1f}'.format(heat), avg_psnr_y, current_step)
                        tb_logger.add_scalar('lpips_{:.1f}'.format(heat), avg_lpips, current_step)
                        tb_logger.add_scalar('diversity_{:.1f}'.format(heat), avg_diversity, current_step)
                        tb_logger.add_scalar('lr_psnr_y_{:.1f}'.format(heat), avg_lr_psnr_y, current_step)
                        tb_logger.add_scalar('nll_{:.1f}'.format(heat), avg_nll, current_step)

                del loss_fn_alex, visuals

    if opt['rank'] <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of model training.')


if __name__ == '__main__':
    main()
