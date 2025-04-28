import os
import math
import argparse
import random
import logging
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from models import CBAM

import numpy as np

#设置分布式训练环境并初始化训练流程的部分，主要用于深度学习模型的训练。

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
#该函数用于初始化分布式训练环境：
#如果当前的多进程启动方法不是 spawn，则设置为 spawn。
#获取当前进程的排名（rank）和可用的GPU数量，并设置当前进程使用的GPU。初始化PyTorch的分布式进程组。

def main():
    #主程序负责设置训练所需的所有配置，并启动训练过程。
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    #解析命令行参数，包括配置文件路径、分布式训练启动器的选择和本地排名。
    #parser：一个 argparse.ArgumentParser 实例，用于定义和解析命令行参数。
    #args：一个命名空间对象，包含所有解析出来的命令行参数及其值。
    #opt：一个字典或类字典对象，包含从 YAML 配置文件中解析出的所有训练选项及其值。

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    #根据选择的启动器设置分布式训练标志，并初始化分布式环境。

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None
    #如果存在恢复状态，则加载该状态，并检查恢复选项。

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
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
            tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['root'], 'tb_logger', opt['name']))
            #tb_logger = SummaryWriter(log_dir=  + '/tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    #根据进程排名创建必要的目录，并设置日志记录器。

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    #设置随机种子以保证实验的可重复性。

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #禁用 CUDA 的 benchmark 模式，并启用确定性模式，以保证实验的一致性。

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #根据配置文件创建训练和验证数据集及数据加载器，并根据是否为分布式训练设置采样器。

    #### create model
    model = create_model(opt)

    #根据配置文件 opt 创建模型实例。

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
    #如果存在恢复状态，则从上次中断的地方继续训练；否则，从头开始训练。

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    # 初始化日志，记录开始训练的时间和初始的 epoch 和迭代步骤。
    first_time = True
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
            #如果是分布式训练，则设置数据集采样器的 epoch，以确保每个进程的数据分布是一致的
        for _, train_data in enumerate(train_loader):
            if first_time:
                start_time = time.time()
                first_time = False
            current_step += 1
            if current_step > total_iters:
                break
            #遍历训练数据加载器中的每一个 batch，更新 current_step，并在达到最大迭代次数 total_iters 后停止训练。
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            #将数据喂入模型，并执行参数优化。

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                end_time = time.time()
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, , time:{:.3f}> '.format(
                    epoch, current_step, model.get_current_learning_rate(), end_time-start_time)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
                start_time = time.time()
                #每隔一定步数记录训练日志，包括当前的 epoch、迭代次数、学习率、耗时以及其他训练指标。
                #如果使用 TensorBoard 日志记录，并且不是调试模式，则记录相应的标量数据。

            """
            进入训练循环，每一步包括：
            更新数据集的采样器（仅在分布式训练情况下）。
            读取一批次的训练数据，并更新训练步数。
            超过总迭代次数后停止训练。
            使用模型的 feed_data 方法输入数据，并优化参数。
            更新学习率。
            每隔一定步数记录训练日志，并使用 TensorBoard 进行可视化。
            如果到达验证频率，则进行模型验证。
            """

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_normalized_psnr = 0.0
                avg_tonemapped_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    # img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    # img_dir = os.path.join(opt['path']['val_images'], img_name)
                    # util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()

                    sr_img = util.tensor2numpy(visuals['SR']) # float32
                    gt_img = util.tensor2numpy(visuals['GT']) # float32

                    # calculate PSNR
                    avg_psnr += util.calculate_psnr(sr_img, gt_img)
                    avg_normalized_psnr += util.calculate_normalized_psnr(sr_img, gt_img, np.max(gt_img))
                    avg_tonemapped_psnr += util.calculate_tonemapped_psnr(sr_img, gt_img, percentile=99, gamma=2.24)

                avg_psnr = avg_psnr / idx
                avg_normalized_psnr = avg_normalized_psnr / idx
                avg_tonemapped_psnr = avg_tonemapped_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}, norm_PSNR: {:.4e}, mu_PSNR: {:.4e}'.format(avg_psnr, avg_normalized_psnr, avg_tonemapped_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} norm_PSNR: {:.4e} mu_PSNR: {:.4e}'.format(
                    epoch, current_step, avg_psnr, avg_normalized_psnr, avg_tonemapped_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('norm_PSNR', avg_normalized_psnr, current_step)
                    tb_logger.add_scalar('mu_PSNR', avg_tonemapped_psnr, current_step)

            #在验证阶段，模型对验证集中的每个样本进行处理，并计算 PSNR（峰值信噪比）、归一化 PSNR 和调制映射 PSNR。
            #然后记录这些指标，并使用 TensorBoard 可视化。

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

            #每隔一定的步数保存当前的模型权重和训练状态。

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')

    #训练结束后，保存最终的模型权重，并记录训练结束的信息。

if __name__ == '__main__':
    main()
#脚本的入口点

#代码实现了从加载模型、恢复训练状态、执行训练迭代、记录日志、模型验证到保存模型和训练状态的完整训练流程。
#通过这种方式，可以确保训练过程中的关键步骤都被正确记录，并且可以在需要的时候恢复训练。