import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from shutil import get_terminal_size

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
"""
这段代码包含了一系列辅助函数，用于处理深度学习模型训练和测试过程中的一些常见操作，比如日志记录、图像处理、随机种子设置、路径管理和性能度量等。
"""

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

"""
这个函数扩展了 YAML 的处理能力，使其支持有序字典 OrderedDict，这有助于保持配置文件中的顺序。
"""

####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

#返回当前日期和时间的时间戳字符串。

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#如果指定的目录不存在，则创建它。


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

#如果传入的是单个路径字符串，则直接创建；如果是路径列表，则遍历列表创建所有目录。

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

#如果指定的目录存在，则重命名该目录并在其名称后附加时间戳；然后创建新的目录。


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#设置 Python 的 random 模块、NumPy 和 PyTorch 的随机种子，确保结果的一致性和可重复性。

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

#设置日志记录器，可以将日志记录到文件和/或屏幕。

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        #img_np = tensor.numpy()
        img_np = tensor.numpy()
        img_np = np.expand_dims(img_np, axis=2)
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        img_np = np.squeeze(img_np)
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

#将 PyTorch 的 Tensor 转换为 NumPy 数组格式的图像，并可选择输出类型（默认为 uint8）。

def tensor2numpy(tensor):
    img_np = tensor.numpy()
    img_np[img_np < 0] = 0
    # 判断图像是否有多个通道
    if img_np.shape[0] == 1:  # 单通道图像
        img_np = img_np[0, :, :]  # 只保留单通道图像的内容
    else:
        # 处理多通道图像（例如 RGB）
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)

#将 PyTorch 的 Tensor 转换为 NumPy 数组，并且对数组进行一些基本的处理，如转换维度顺序。

def save_img_with_ratio(image_path, image, alignratio_path):
    align_ratio = (2 ** 16 - 1) / image.max()
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, uint16_image_gt)
    return None

#保存图像，并将图像的最大值比率保存为 .npy 文件。

def generate_paths(folder, name):
    id = name[:4]
    image_path = os.path.join(folder, id+'.png')
    alignratio_path = os.path.join(folder, id+'_alignratio.npy')
    return image_path, alignratio_path

#根据文件夹路径和文件名生成图像路径和比率文件路径。

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

#保存图像到指定路径。

def save_npy(img, img_path):
    img = np.squeeze(img)
    np.save(img_path, img)

#保存 NumPy 数组到 .npy 文件。

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))

#计算两个图像之间的峰值信噪比（PSNR）。

def calculate_normalized_psnr(img1, img2, norm):
    normalized_psnr = -10*np.log10(np.mean(np.power(img1/norm - img2/norm, 2)))
    if normalized_psnr == 0:
        return float('inf')
    return normalized_psnr

#计算经过归一化后的 PSNR。

def mu_tonemap(hdr_image, mu=5000):
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

#应用 Mu 合成映射算法，用于 HDR 图像的 Tone Mapping。

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

#先使用双曲正切函数对 HDR 图像进行范围限制，再应用 Mu 合成映射算法。

def calculate_tonemapped_psnr(res, ref, percentile=99, gamma=2.24):
    res = res ** gamma
    ref = ref ** gamma
    norm_perc = np.percentile(ref, percentile)
    tonemapped_psnr = -10*np.log10(np.mean(np.power(tanh_norm_mu_tonemap(ref, norm_perc) - tanh_norm_mu_tonemap(res, norm_perc), 2)))
    return tonemapped_psnr

#计算经过 Tone Mapping 后的图像之间的 PSNR。

#代码提供了许多实用的功能，用于简化深度学习模型开发中的常见任务，如文件系统的操作、日志记录、图像处理和性能度量等。
# 这些工具函数可以帮助开发者更专注于模型的设计与优化，而不需要关心太多细节性的底层实现。