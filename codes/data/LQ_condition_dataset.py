import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import cv2

#
# class LQ_Dataset(data.Dataset):
#     def __init__(self, opt):
#         super(LQ_Dataset, self).__init__()
#         self.opt = opt
#         self.paths_LQ = None
#         self.LQ_env = None
#
#         self.LQ_env, self.paths_LQ = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
#         assert self.paths_LQ, 'Error: LQ paths are empty.'
#
#         # Initialize paired transforms for training phase only
#         if self.opt.get('phase', 'test') == 'train':
#             self.paired_transforms = PairedTransforms(use_flip=opt.get('use_flip', False),
#                                                       use_rot=opt.get('use_rot', False))
#         else:
#             self.paired_transforms = None
#
#     def __getitem__(self, index):
#         LQ_path = self.paths_LQ[index]
#         img_LQ = util.read_img(self.LQ_env, LQ_path)
#
#         # Process image shape and convert to grayscale if necessary
#         C = 1 if len(img_LQ.shape) == 2 else img_LQ.shape[2]
#         if C == 3:
#             img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2GRAY)
#
#         cond = img_LQ.copy() if self.opt['condition'] == 'image' else util.calculate_gradient(img_LQ)
#         if cond.ndim == 3 and cond.shape[2] == 3:
#             cond = cv2.cvtColor(cond, cv2.COLOR_BGR2GRAY)
#
#         # Apply paired transformations if in training phase
#         if self.paired_transforms is not None:
#             img_LQ, cond = self.paired_transforms(img_LQ, cond)
#
#         # Convert to PyTorch tensor
#         img_LQ = torch.from_numpy(np.expand_dims(img_LQ, axis=0)).float()
#         cond = torch.from_numpy(np.expand_dims(cond, axis=0)).float()
#
#         return {'LQ': img_LQ, 'LQ_path': LQ_path, 'cond': cond}
#
#     def __len__(self):
#         return len(self.paths_LQ)


class LQ_Dataset(data.Dataset):
    '''Read LQ images only in the test phase.
    LQ_Dataset 继承自 torch.utils.data.Dataset，专门用于读取低质量图像（LQ）。
    '''



    def __init__(self, opt):
        super(LQ_Dataset, self).__init__()
        self.opt = opt
        self.paths_LQ = None
        self.LQ_env = None  # environment for lmdb

        # read image list from lmdb or image files
        print(opt['data_type'], opt['dataroot_LQ'])
        self.LQ_env, self.paths_LQ = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'

        """
        参数：
        opt：包含数据集配置的字典，通常包含数据类型和数据根路径等信息。
        调用父类的构造函数。
        初始化路径和环境变量。
        通过 util.get_image_paths 函数读取图像路径列表，确保路径不为空。
        """

    def __getitem__(self, index):
        LQ_path = None

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_img(self.LQ_env, LQ_path)
        # print(f"Original LQ image shape: {img_LQ.shape}")  # 打印原始 LQ 图像形状

        # 处理图像形状
        if len(img_LQ.shape) == 2:  # 灰度图像
            H, W = img_LQ.shape
            C = 1
        elif len(img_LQ.shape) == 3:  # Color image (H, W, C)
            H, W, C = img_LQ.shape
        else:
            raise ValueError(f"Unsupported image shape: {img_LQ.shape}")

        # 将三通道转化为灰度图
        if C == 3:
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2GRAY)
            print(f"Converted to grayscale, shape: {img_LQ.shape}")  # 打印转换后的图像形状

        # condition
        if self.opt['condition'] == 'image':
            cond = img_LQ.copy()
            # print(f"Original cond tensor shape: {img_LQ.shape}")  # 打印原始 LQ 图像形状
        elif self.opt['condition'] == 'gradient':
            cond = util.calculate_gradient(img_LQ)
            # print(f"gradient,Original cond tensor shape: {img_LQ.shape}")  # 打印原始 LQ 图像形状
        else:
            raise ValueError(f"Unsupported condition: {self.opt['condition']}")

        # 确保条件数据也是灰度图
        if cond.ndim == 3 and cond.shape[2] == 3:
            cond = cv2.cvtColor(cond, cv2.COLOR_BGR2GRAY)
            print(f"Converted condition to grayscale, shape: {cond.shape}")  # 打印转换后的条件数据形状

        # 将灰度图像从 (H, W) 转换为 (1, H, W)，并将 NumPy 数组转换为 PyTorch 张量
        img_LQ = torch.from_numpy(np.expand_dims(img_LQ, axis=0)).float()  # 确保 img_LQ 为 (1, H, W)
        cond = torch.from_numpy(np.expand_dims(cond, axis=0)).float()  # 确保 cond 为 (1, H, W)
        img_LQ = img_LQ.squeeze(3)  # 删除最后一维
        cond = cond.squeeze(3)  # 删除最后一维
        # print(f"Final LQ tensor shape: {img_LQ.shape}, Final cond tensor shape: {cond.shape}")  # 打印最终张量形状

        return {'LQ': img_LQ, 'LQ_path': LQ_path, 'cond': cond}
        # """
        #     返回一个字典，包含：
        #     'LQ'：低质量图像的张量。
        #     'LQ_path'：低质量图像的路径。
        #     'cond'：条件数据（图像或梯度）。
        # """

    def __len__(self):
        return len(self.paths_LQ)
        # 返回数据集中低质量图像路径的数量，以便在数据加载时可以获取数据集的长度。