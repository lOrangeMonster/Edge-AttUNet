import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
"""
这段代码定义了一个名为 BaseModel 的基类，它为深度学习模型提供了一些基本的方法和属性。
这个类主要用于封装训练过程中常用的操作，如数据喂入、优化参数、获取当前可视化结果和损失等。
此外，它还包括了学习率调整、网络状态保存与加载等功能。
"""

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
    """
    参数：
    opt：一个配置字典，包含了模型运行时的各种配置选项。
    成员变量初始化：
    self.opt：存储配置选项。
    self.device：设置设备为 GPU 或 CPU，取决于配置。
    self.is_train：标记是否处于训练模式。
    self.schedulers 和 self.optimizers：初始化调度器和优化器列表。
    """
    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass
# """
# 这些方法定义了模型的基本操作，但在基类中并未实现具体逻辑。子类需要覆盖这些方法来提供具体实现。
# """
    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr
    #参数：lr_groups_l：一个列表，包含了每个优化器的学习率组。
    #作用：根据提供的学习率组更新所有优化器的学习率。
    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l
    #作用：获取每个优化器的初始学习率组。

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
    #参数：cur_iter：当前迭代次数。
    #warmup_iter：预热迭代次数。
    #作用：更新所有调度器，并在预热阶段调整学习率。

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']
    #作用：返回当前的学习率。

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
    #参数：network：网络对象。
    #作用：获取网络的字符串描述及其参数总数。

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
    #参数：network：网络对象。network_label：网络标签。iter_label：迭代标签。
    #作用：保存网络的状态字典到指定文件。

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)
    #参数：load_path：网络状态字典的路径。network：网络对象。strict：是否严格匹配键。
    #作用：从文件加载网络的状态字典，并加载到网络中。


    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)
    #参数：epoch：当前的轮次。iter_step：当前的迭代步数。
    #作用：保存当前的训练状态，包括优化器和调度器的状态。

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
    # 参数：resume_state：要恢复的训练状态字典。
    # 作用：从给定的训练状态字典恢复优化器和调度器的状态。
"""
这个 BaseModel 类为模型提供了一个框架，子类可以通过继承并实现其中的抽象方法来定制自己的模型行为。
此外，这个类还提供了许多实用的方法，如学习率管理、网络状态保存与加载、训练状态的保存与恢复等，这些都是训练深度学习模型时常见的需求。
通过这样的设计，可以简化模型开发的工作量，并提高代码的可复用性和可维护性。
"""