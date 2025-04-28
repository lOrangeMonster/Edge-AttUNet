import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.customize_loss import tanh_L1Loss, tanh_L2Loss

logger = logging.getLogger('base')

class GenerationModel(BaseModel):
    def __init__(self, opt):
        super(GenerationModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'tanh_l1':
                self.cri_pix = tanh_L1Loss().to(self.device)
            elif loss_type == 'tanh_l2':
                self.cri_pix = tanh_L2Loss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

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
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        """
        参数：
        opt：模型配置字典。
        成员变量初始化：
        self.rank：分布式训练时的进程排名，非分布式训练时设为 -1。
        self.netG：生成网络。
        self.cri_pix：像素级别的损失函数。
        self.l_pix_w：像素级别损失的权重。
        self.optimizer_G：优化器。
        self.schedulers：学习率调度器。
        self.log_dict：记录训练过程中的日志信息。
        初始化步骤：
        定义生成网络 self.netG 并加载到指定设备。
        如果是分布式训练，则使用 DistributedDataParallel 包装网络；否则使用 DataParallel。
        打印网络结构。
        加载预训练模型（如果有）。
        如果是在训练模式下，设置网络为训练模式，并根据配置初始化损失函数、优化器和学习率调度器。
        """

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        self.var_cond = data['cond'].to(self.device) # cond
        
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    #参数：data：输入数据字典。need_GT：是否需要真实标签。
    #作用：将输入数据加载到模型的变量中。

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG((self.var_L, self.var_cond))
        
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

        #参数：step：当前的训练步数。
        #作用：执行一次前向传播，计算损失，反向传播并更新参数。

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG((self.var_L, self.var_cond))
        self.netG.train()

    # 作用：将网络设置为评估模式，执行一次前向传播，并将网络恢复为训练模式。

    def get_current_log(self):
        return self.log_dict

    # 作用：返回当前的日志信息

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict
    #参数：need_GT：是否需要真实标签。
    #作用：返回当前的可视化结果，包括输入、输出以及（如果需要的话）真实标签。

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

        #作用：打印网络的结构和参数数量。

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        #作用：加载预训练模型（如果有）。

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

        #参数：iter_label：迭代标签。
        #作用：保存当前模型的状态。

        """
        GenerationModel 类扩展了 BaseModel 类，增加了特定于生成模型的功能，如定义网络、初始化优化器和学习率调度器、定义损失函数、以及训练和测试的具体实现。
        这个类的设计使得它可以很容易地集成到更复杂的应用程序中，如图像生成、超分辨率重建等任务。
        通过继承 BaseModel，它可以重用一些通用功能，如保存/加载模型状态、日志记录等，从而减少了代码冗余，提高了代码的可维护性。
        """
