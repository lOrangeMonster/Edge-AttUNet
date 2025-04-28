"""
Modified from torch.utils.data.distributed.DistributedSampler
Support enlarging the dataset for *iter-oriented* training, for saving time when restart the
dataloader after each epoch
"""
import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class DistIterSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    DistIterSampler 提供了一种高效的方式来在分布式训练中进行数据抽样。通过确保每个进程加载不同的子集，结合 epoch 的变化，能够实现更高效的训练过程。

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    """
    参数说明：
    dataset：要采样的数据集。
    num_replicas：参与分布式训练的进程数。
    rank：当前进程的排名。
    ratio：用于调整每个进程加载的样本数量的比例，默认为100。
    """
    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
            """如果未提供 num_replicas，则从分布式环境中获取。"""
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            """如果未提供 rank，则从分布式环境中获取。"""
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        """
        将参数保存为实例变量，并计算每个进程要处理的样本数量 (num_samples) 和总样本大小 (total_size)。
        """
    def __iter__(self):
        # deterministically shuffle based on epoch 定义了如何迭代采样器。
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # 创建一个随机数生成器，并基于当前的 epoch 设置种子，以确保可重复性。
        indices = torch.randperm(self.total_size, generator=g).tolist()
        #生成一个从 0 到 total_size-1 的随机排列，表示数据集的索引。
        dsize = len(self.dataset)
        indices = [v % dsize for v in indices]
        #将随机索引映射到数据集的实际大小，以处理可能的越界。
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        #从随机排列中选择当前进程对应的样本。并且确保选择的样本数量与预期一致。
        return iter(indices)
    #返回一个可迭代的索引对象。

    def __len__(self):
        return self.num_samples
    #返回当前进程要处理的样本数量。
    def set_epoch(self, epoch):
        self.epoch = epoch
    #更新当前的 epoch，用于在下一次迭代中确定随机数生成器的种子，以实现确定性的随机性。