import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
import scipy.ndimage

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.npy']
#IMG_EXTENSIONS: 一个列表，包含了所有支持的图像文件扩展名的大写和小写形式。

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#这个函数用来检查给定的文件名是否是一个图像文件。它通过检查文件名的扩展名是否在 IMG_EXTENSIONS 列表中来进行判断。

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images
#此函数用于获取指定路径下的所有图像文件的完整路径，并返回一个包含这些路径的列表。如果路径不存在或不包含有效的图像文件，则会抛出异常。

def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes
#用于从LMDB数据库中获取图像路径和大小信息。它首先加载LMDB元数据文件，然后从中提取图像路径和大小信息。

def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return sizes, paths
#根据提供的数据类型（'lmdb' 或 'img'），此函数会选择合适的函数来获取图像路径和大小。如果数据类型不是这两个中的一个，它将抛出一个未实现错误。

###################### read images ######################
def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img
#用于从LMDB环境中读取图像。它接受一个环境对象、键值和大小，然后读取相应的图像并根据大小调整形状。

def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        if os.path.splitext(path)[1] == '.npy':
            img = np.load(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img
#此函数用于从文件系统或LMDB读取图像。读取后，它将图像转换为浮点数，并根据其原始类型进行归一化。

def read_npy(path):
    return np.load(path)
#专门用于读取 .npy 文件，返回一个 NumPy 数组。

def read_imgdata(path, ratio=255.0):
    # 读取图像数据
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    # 打印调试信息
    # print(f"Original image shape: {img.shape}")
    # print(f"Ratio: {ratio}, Ratio type: {type(ratio)}")
    ratio = ratio
    # 检查 ratio 是否为标量值
    if not isinstance(ratio, (int, float)):
        raise ValueError("Ratio should be a scalar value.")

    # 将图像数据除以 ratio
    img = img / ratio

    return img
    #return cv2.imread(path, cv2.IMREAD_UNCHANGED) / ratio
#读取图像文件，并根据给定的比例因子对图像进行归一化。

def expo_correct(img, exposures, idx):
    floating_exposures = exposures - exposures[1]
    gamma=2.24
    img_corrected = (((img**gamma)*2.0**(-1*floating_exposures[idx]))**(1/gamma))
    return img_corrected
#对图像进行曝光校正，根据曝光值调整图像亮度。

####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if len(img.shape) == 2:  # 灰度图像
            if hflip:
                img = img[:, ::-1]
            if vflip:
                img = img[::-1, :]
            if rot90:
                img = img.transpose(1, 0)
        else:  # 彩色图像
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]
#对图像列表进行增强处理，包括随机水平翻转、垂直翻转和旋转90度。

def augment_flow(img_list, flow_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if len(img.shape) == 2:  # 灰度图像
            if hflip:
                img = img[:, ::-1]
            if vflip:
                img = img[::-1, :]
            if rot90:
                img = img.transpose(1, 0)
        else:  # 彩色图像
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if len(flow.shape) == 2:  # 灰度图像
            if hflip:
                flow = flow[:, ::-1]
                flow[:, 0] *= -1
            if vflip:
                flow = flow[::-1, :]
                flow[:, 1] *= -1
            if rot90:
                flow = flow.transpose(1, 0)
                flow = flow[:, [1, 0]]
        else:  # 彩色图像
            if hflip:
                flow = flow[:, ::-1, :]
                flow[:, :, 0] *= -1
            if vflip:
                flow = flow[::-1, :, :]
                flow[:, :, 1] *= -1
            if rot90:
                flow = flow.transpose(1, 0, 2)
                flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list
#类似于 augment 函数，但同时处理光流图，并根据翻转和旋转操作调整光流向量的方向。

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img
#模块裁剪函数，用于确保图像尺寸可以被给定的尺度整除，常用于准备图像以进行尺度相关的处理。

def calculate_gradient(img, ksize=-1):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobelxy.astype(np.float32) / 255.
#计算图像的梯度，使用Sobel算子分别计算水平和垂直方向的梯度，然后组合这两个方向的梯度得到最终的结果。

#这些函数组合在一起，提供了一套完整的图像处理工具集，适用于各种图像处理任务，特别是在深度学习模型训练前的数据预处理阶段非常有用。

####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))
#定义了一个三次样条插值函数，用于图像重采样过程中的像素值计算。此函数根据输入值的不同返回不同的计算结果。

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)
#该函数用于计算重采样过程中使用的权重和索引。它根据输入长度、输出长度、缩放比例等因素，计算出用于重采样的权重矩阵和对应的像素索引。

def imresize(img, scale, antialiasing=True):
    if img.dim() == 2:  # 单通道图像
        img = img.unsqueeze(0)  # 添加通道维度
        in_C, in_H, in_W = 1, img.size(1), img.size(2)
    else:  # 多通道图像
        in_C, in_H, in_W = img.size()

    out_H, out_W = math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)

    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for c in range(in_C):
            out_1[c, i, :] = img_aug[c, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for c in range(in_C):
            out_2[c, :, i] = out_1_aug[c, :, idx:idx + kernel_width].mv(weights_W[i])

    if img.dim() == 2:  # 单通道图像
        out_2 = out_2.squeeze(0)  # 移除通道维度

    return out_2
#实现了类似于MATLAB中 imresize 的功能，但仅支持三次样条插值（bicubic）。
# 此函数用于改变图像大小，支持抗锯齿处理。它首先计算权重和索引，然后根据这些权重对图像进行重采样。

def imresize_np(img, scale, antialiasing=True):
    """
        实现类似于MATLAB中 imresize 的功能，但仅支持三次样条插值（bicubic）。
        此函数用于改变图像大小，支持抗锯齿处理。它首先计算权重和索引，然后根据这些权重对图像进行重采样。

        参数:
        img (numpy.ndarray): 输入图像，可以是单通道 (H, W) 或多通道 (H, W, C)。
        scale (float): 缩放比例。
        antialiasing (bool): 是否进行抗锯齿处理。

        返回:
        numpy.ndarray: 调整大小后的图像。
        """
    img = torch.from_numpy(img)

    if img.dim() == 2:  # 单通道图像
        in_H, in_W = img.size()
        in_C = 1
        img = img.unsqueeze(-1)  # 添加通道维度
    else:  # 多通道图像
        in_H, in_W, in_C = img.size()

    out_H, out_W = math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)

    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for c in range(in_C):
            out_1[i, :, c] = img_aug[idx:idx + kernel_width, :, c].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for c in range(in_C):
            out_2[:, i, c] = out_1_aug[:, idx:idx + kernel_width, c].mv(weights_W[i])

    if img.dim() == 2:  # 单通道图像
        out_2 = out_2.squeeze(-1)  # 移除通道维度

    return out_2.numpy()
#类似于 imresize 函数，但是输入为 NumPy 数组而非 PyTorch 张量。此函数同样支持抗锯齿处理，并返回 NumPy 数组形式的结果

def filtering(img_gray, r, eps):
    """
        实现了一种图像滤波算法，通过对图像进行均值和方差计算，并应用加权平均来平滑图像。

        参数:
        img_gray (numpy.ndarray): 输入的灰度图像 (H, W)。
        r (int): 滤波器窗口大小的一半。
        eps (float): 平滑参数。

        返回:
        numpy.ndarray: 滤波后的图像。
        """
    img = np.copy(img_gray)
    H = 1 / np.square(r) * np.ones([r, r])
    meanI = scipy.ndimage.correlate(img, H, mode='nearest')

    var = scipy.ndimage.correlate(img * img, H, mode='nearest') - meanI * meanI
    a = var / (var + eps)
    b = meanI - a * meanI

    meana = scipy.ndimage.correlate(a, H, mode='nearest')
    meanb = scipy.ndimage.correlate(b, H, mode='nearest')
    output = meana * img + meanb
    return output
#实现了一种图像滤波算法，通过对图像进行均值和方差计算，并应用加权平均来平滑图像。

def guided_filter(img, r=5, eps=0.01):
    """
    实现了引导滤波算法，通过对图像进行均值和方差计算，并应用加权平均来平滑图像。

    参数:
    img (numpy.ndarray): 输入图像，可以是单通道 (H, W) 或多通道 (H, W, C)。
    r (int): 滤波器窗口大小的一半。
    eps (float): 平滑参数。

    返回:
    numpy.ndarray: 滤波后的图像。
    """
    if img.ndim == 2:  # 单通道图像
        return filtering(img, r, eps)
    elif img.ndim == 3:  # 多通道图像
        img_filtered = np.copy(img)
        for i in range(img.shape[2]):
            img_filtered[:, :, i] = filtering(img[:, :, i], r, eps)
        return img_filtered
    else:
        raise ValueError(f'Wrong img ndim: [{img.ndim}]. Expected 2 or 3 dimensions.')
#使用指导滤波（Guided Filter）技术来平滑图像，保持边缘细节的同时减少噪声。此函数遍历图像的每个颜色通道，并应用 filtering 函数进行滤波。

    # """
    # imresize 和 imresize_np 函数详解
    # 这两个函数的核心在于它们如何处理图像的重采样。它们都是基于三次样条插值来实现的，但 imresize 处理的是 PyTorch 张量，而 imresize_np 处理的是 NumPy 数组。这两个函数都遵循以下步骤：
    #
    # 计算权重和索引：使用 calculate_weights_indices 函数来确定重采样所需的权重和索引。
    # 边界扩展：为了处理图像边界上的像素，函数会对图像进行扩展，使其超出边界的部分也能够进行有效的重采样。
    # 逐行逐列重采样：通过对扩展后的图像应用权重，函数逐行逐列地计算新的像素值，从而生成缩放后的图像。
    # filtering 和 guided_filter 函数详解
    # filtering 函数是一个基础的滤波函数，它计算图像的局部均值和方差，并应用一个加权平均来生成平滑的结果。guided_filter 函数则是基于指导滤波算法，这是一种可以保留更多细节的平滑技术。它通过遍历图像的每个颜色通道，并对每个通道应用 filtering 函数来达到平滑效果。
    #
    # 总结
    # 这些函数主要用于图像的预处理阶段，如图像缩放、去噪和平滑等。它们可以作为图像处理流程的一部分，用于提高图像质量或准备图像以供后续处理或分析使用。
    # """