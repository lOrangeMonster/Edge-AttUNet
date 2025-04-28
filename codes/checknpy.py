# import numpy as np

# # 加载 .npy 文件
# data = np.load(r"E:\tool\HDRUNet-main\data0\000_Train_SingleFrame_FirstStage\alignratio\0003_alignratio.npy")
#
# # 打印数据
# print(data)
import os
import numpy as np
from scipy.special import expit  # Sigmoid function
import logging
import tifffile  # 用于读取 .tif 文件
import cv2
import os
import numpy as np
from scipy.special import expit  # Sigmoid function
import logging
import tifffile  # 用于读取 .tif 文件
import imagecodecs  # 确保支持 LZW 压缩
import cv2

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_alignratio_based_on_gray_mean(lq_img_path, gt_img_path):
    """
    根据图像的平均灰度值计算 alignratio，并应用非线性函数调整。

    :param lq_img_path: LQ 图像路径
    :param gt_img_path: GT HDR 图像路径
    :return: alignratio (单一数值)
    """
    try:
        # 读取图像并转换为灰度图（对于 .tif 文件）
        lq_img = tifffile.imread(lq_img_path).astype(np.float32)
        gt_img = tifffile.imread(gt_img_path).astype(np.float32)

        if lq_img is None or gt_img is None:
            raise ValueError(f"Failed to load images from {lq_img_path} or {gt_img_path}")

        # 确保图像尺寸相同
        if lq_img.shape[:2] != gt_img.shape[:2]:
            gt_img = cv2.resize(gt_img, (lq_img.shape[1], lq_img.shape[0]))

        # 计算平均灰度值
        lq_mean = np.mean(lq_img)
        gt_mean = np.mean(gt_img)

        # 应用非线性函数调整 alignratio
        k = 0.005  # 控制曲线的陡峭程度
        x0 = 128  # 控制曲线的中心点

        # 确保所有输入都是标量
        adjusted_ratio = expit(k * (gt_mean - lq_mean - x0)) * (gt_mean / lq_mean)

        # 确保返回的是标量
        if not np.isscalar(adjusted_ratio):
            raise ValueError("Adjusted ratio should be a scalar value.")

        return float(adjusted_ratio)  # 强制转换为浮点数以确保是标量
    except Exception as e:
        logging.error(f"Error in calculate_alignratio_based_on_gray_mean: {e}")
        raise


def process_exposure_files(lq_dir, gt_dir, output_dir):
    try:
        lq_files = sorted([f for f in os.listdir(lq_dir) if f.endswith('.tif')])
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.tif')])

        if not lq_files or not gt_files:
            logging.warning("No files found in the specified directories.")
            return

        assert len(lq_files) == len(gt_files), "Number of LQ and GT images must be the same"

        for lq_file, gt_file in zip(lq_files, gt_files):
            lq_path = os.path.join(lq_dir, lq_file)
            gt_path = os.path.join(gt_dir, gt_file)

            # 计算 alignratio
            alignratio = calculate_alignratio_based_on_gray_mean(lq_path, gt_path) * 9

            # 输出 alignratio 到文件
            ratio_filename = lq_file.replace('_medium50_10.tif', '_alignratio.npy')
            ratio_path = os.path.join(output_dir, ratio_filename)
            np.save(ratio_path, alignratio)

            logging.info(f"Processed {lq_file} -> {ratio_filename}")
            logging.info(f"Alignratio: {alignratio}, Ratio type: {type(alignratio)}")
            #logging.info(f"Original image shape: {lq_img.shape}")
    except Exception as e:
        logging.error(f"Error in process_exposure_files: {e}")
        raise


if __name__ == "__main__":
    # 示例路径 train valid
    lq_dir = 'E:/tool/HDRUNet-main/data0/000_single_valid_1122/medium50_10'
    gt_dir = 'E:/tool/HDRUNet-main/data0/000_single_valid_1122/gt'
    output_dir = 'E:/tool/HDRUNet-main/data0/000_single_valid_1122/alignratio'

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 处理图像并输出 alignratio
    process_exposure_files(lq_dir, gt_dir, output_dir)