import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp


# class LQGT_dataset(data.Dataset):
#     def __init__(self, opt):
#         super(LQGT_dataset, self).__init__()
#         self.opt = opt
#         self.data_type = opt['data_type']
#         self.paths_LQ, self.paths_GT = None, None
#
#         self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])[1]
#         self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])[1]
    #
    #     # Initialize paired transforms for training phase only
    #     if self.opt.get('phase', 'test') == 'train':
    #         self.paired_transforms = PairedTransforms(use_flip=opt.get('use_flip', False),
    #                                                   use_rot=opt.get('use_rot', False))
    #     else:
    #         self.paired_transforms = None
    #
    # def __getitem__(self, index):
    #     GT_path = self.paths_GT[index]
    #     LQ_path = self.paths_LQ[index]
    #
    #     img_LQ = util.read_imgdata(LQ_path, ratio=255.0)
    #     alignratio = np.load(osp.join(self.opt['dataroot_ratio'], osp.basename(LQ_path)[:4] + "_alignratio.npy")).mean()
    #     img_GT = util.read_imgdata(GT_path, ratio=alignratio)
    #
    #     if self.opt['phase'] == 'train':
    #         # Random crop
    #         scale = self.opt['scale']
    #         GT_size = self.opt['GT_size']
    #         LQ_size = GT_size // scale
    #
    #         H, W, _ = img_LQ.shape
    #         rnd_h = random.randint(0, max(0, H - LQ_size))
    #         rnd_w = random.randint(0, max(0, W - LQ_size))
    #         img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
    #         img_GT = img_GT[rnd_h * scale:(rnd_h + LQ_size) * scale, rnd_w * scale:(rnd_w + LQ_size) * scale, :]
    #
    #         # Apply paired transformations
    #         if self.paired_transforms is not None:
    #             img_LQ, img_GT = self.paired_transforms(img_LQ, img_GT)
    #
    #     # Condition
    #     cond = img_LQ.copy() if self.opt['condition'] == 'image' else util.calculate_gradient(img_LQ)
    #
    #     # Convert to PyTorch tensor
    #     img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
    #     img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
    #     cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()
    #
    #     return {'LQ': img_LQ, 'GT': img_GT, 'cond': cond, 'LQ_path': LQ_path, 'GT_path': GT_path}
    #
    # def __len__(self):
    #     return len(self.paths_GT)



class LQGT_dataset(data.Dataset):
    """
    代码定义了一个数据集类 LQGT_dataset，用于加载低质量（LQ）图像和真实图像（GT），
    可能用于图像处理或机器学习相关任务，比如超分辨率。它使用 PyTorch 的 Dataset 类来方便高效的数据加载。
    """





    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None

        self.sizes_LQ, self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.folder_ratio = opt['dataroot_ratio']
  

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_imgdata(LQ_path, ratio=255.0)
        # print(f"Original LQ image shape: {img_LQ.shape}")  # 打印原始 LQ 图像形状

        # get GT alignratio
        filename = osp.basename(LQ_path)[:4] + "_alignratio.npy"
        ratio_path = osp.join(self.folder_ratio, filename)
        alignratio = np.load(ratio_path).astype(np.float32)
        # 确保 alignratio 是一个标量值
        if alignratio.size != 1:
            # 如果 alignratio 是一个数组，可以取其平均值或其他特定值
            alignratio = alignratio.mean()
            print(f"Converted alignratio to scalar: {alignratio}")


        alignratio = 65535
        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_imgdata(GT_path, ratio=alignratio)
        # print(f"Original GT image shape: {img_GT.shape}")  # 打印原始 GT 图像形状

        if self.opt['phase'] == 'train':
            # 处理图像形状
            if len(img_LQ.shape) == 2:  # 灰度图像
                H, W = img_LQ.shape
                C = 1
            else:  # 彩色图像
                H, W, C = img_LQ.shape

            if len(img_GT.shape) == 2:  # 灰度图像
                H_gt, W_gt = img_GT.shape
                C_gt = 1
            else:  # 彩色图像
                H_gt, W_gt, C_gt = img_GT.shape

            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))

            LQ_size = GT_size // scale

            # randomly crop
            if GT_size != 0:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                if C == 1:  # 灰度图像
                    img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size]
                else:  # 彩色图像
                    img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                if C_gt == 1:  # 灰度图像
                    img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size]
                else:  # 彩色图像
                    img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # condition
        if self.opt['condition'] == 'image':
            cond = img_LQ.copy()
        elif self.opt['condition'] == 'gradient':
            cond = util.calculate_gradient(img_LQ)
        else:
            raise ValueError(f"Unsupported condition: {self.opt['condition']}")

        # 将图像从 HWC 转换为 CHW，并加上一个单通道的维度 (1, H, W)
        if len(img_GT.shape) == 2:  # 灰度图像
            img_GT = torch.from_numpy(np.ascontiguousarray(img_GT[np.newaxis, :, :])).float()
        else:  # 彩色图像
            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        if len(img_LQ.shape) == 2:  # 灰度图像
            img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ[np.newaxis, :, :])).float()
        else:  # 彩色图像
            img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if len(cond.shape) == 2:  # 灰度图像
            cond = torch.from_numpy(np.ascontiguousarray(cond[np.newaxis, :, :])).float()
        else:  # 彩色图像
            cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path

        # print(f"Final LQ tensor shape: {img_LQ.shape}, Final GT tensor shape: {img_GT.shape}, Final cond tensor shape: {cond.shape}")  # 打印最终张量形状

        return {'LQ': img_LQ, 'GT': img_GT, 'cond': cond, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
#这个方法返回 GT 图像的数量，应该与 LQ 图像的数量匹配。
#LQGT_dataset 类旨在便捷地加载、预处理和增强配对的低质量和真实图像，以便训练机器学习模型，特别是在图像增强或超分辨率任务中。
# 随机裁剪和增强技术的使用表明，该数据集为神经网络的强健训练做好了准备。