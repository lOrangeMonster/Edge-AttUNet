import math
from collections import Counter
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import _LRScheduler

"""
这段代码定义了两种学习率调度策略的实现：MultiStepLR_Restart 和 CosineAnnealingLR_Restart。
这两种调度策略都允许在训练过程中重启学习率，并且可以重新设定学习率的权重。
"""

class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

        # """
        # 初始化方法：
        # 参数：
        # optimizer：优化器。
        # milestones：多步学习率衰减的时间点。
        # restarts：重启的时间点，默认为 [0] 表示没有重启。
        # weights：重启时学习率的权重，默认为 [1] 表示权重为 1。
        # gamma：每达到一个里程碑时的学习率衰减因子，默认为 0.1。
        # clear_state：是否清除优化器的状态，默认为 False。
        # last_epoch：上一个周期的最后一步，默认为 -1。
        # 作用：
        # 根据当前周期 last_epoch 来决定是否重启学习率或调整学习率。
        # """

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    """
    初始化方法：
    参数：
    optimizer：优化器。
    T_period：每个周期的时间长度。
    restarts：重启的时间点，默认为 [0] 表示没有重启。
    weights：重启时学习率的权重，默认为 [1] 表示权重为 1。
    eta_min：最小学习率。
    last_epoch：上一个周期的最后一步，默认为 -1。
    作用：
    根据当前周期 last_epoch 来决定是否重启学习率或按余弦退火的方式调整学习率。
    """

if __name__ == "__main__":
    optimizer = torch.optim.Adam([torch.zeros(3, 64, 3, 3)], lr=2e-4, weight_decay=0,
                                 betas=(0.9, 0.99))
    ##############################
    # MultiStepLR_Restart
    ##############################
    ## Original
    lr_steps = [200000, 400000, 600000, 800000]
    restarts = None
    restart_weights = None

    ## two
    lr_steps = [100000, 200000, 300000, 400000, 490000, 600000, 700000, 800000, 900000, 990000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    lr_steps = [
        50000, 100000, 150000, 200000, 240000, 300000, 350000, 400000, 450000, 490000, 550000,
        600000, 650000, 700000, 740000, 800000, 850000, 900000, 950000, 990000
    ]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(optimizer, lr_steps, restarts, restart_weights, gamma=0.5,
                                    clear_state=False)

    ##############################
    # Cosine Annealing Restart
    ##############################
    ## two
    T_period = [500000, 500000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    T_period = [250000, 250000, 250000, 250000]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = CosineAnnealingLR_Restart(optimizer, T_period, eta_min=1e-7, restarts=restarts,
                                          weights=restart_weights)

    ##############################
    # Draw figure
    ##############################
    N_iter = 1000000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_l[i] = current_lr

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Title', fontsize=16, color='k')
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.show()

    # MultiStepLR_Restart 示例：
    # 配置了不同的 lr_steps、restarts 和 restart_weights，以展示不同的学习率调度方式。
    # 创建 MultiStepLR_Restart 实例，并设置相应的参数。
    # CosineAnnealingLR_Restart 示例：
    # 同样配置了不同的 T_period、restarts 和 restart_weights，以展示不同的学习率调度方式。
    # 创建 CosineAnnealingLR_Restart 实例，并设置相应的参数。
    # 绘制学习率变化图
    # 使用 matplotlib 和 seaborn 库绘制学习率随迭代次数变化的曲线图。
    # 通过循环调用 scheduler.step() 来模拟迭代过程，并记录每次迭代后的学习率。
    # 最终绘制图表，并进行了适当的格式化以便于查看。
    # 总结
    # 这段代码展示了如何自定义学习率调度器，并且提供了两种不同的调度策略：一种是基于多个里程碑的学习率下降策略，另一种是基于余弦退火的学习率下降策略。
    # 这两种策略都支持在特定时刻重启学习率，并且可以设置重启时的学习率权重。
    # 此外，通过绘制学习率随迭代次数的变化曲线，可以直观地看到学习率是如何随着时间变化的，这对于理解训练过程中的学习率动态调整是非常有帮助的。
