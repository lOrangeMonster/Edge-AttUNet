import os
import os.path as osp
import numpy as np
import data_io as io
import cv2
import metrics as m

ref_dir = './Validation/gt'
ref_alignratio_dir = './Validation/alignratio'
# res_dir = './Validation/medium'
# res_alignratio_dir = ref_alignratio_dir
res_dir = './Validation/results'
res_alignratio_dir = res_dir

ref_output_dir = './Validation/tone_mapped_gt'
# res_output_dir = './Validation/tone_mapped_medium'
res_output_dir = './Validation/tone_mapped_results'
"""
    ref_dir 和 ref_alignratio_dir：引用图像和对齐比率图像的目录。
    res_dir 和 res_alignratio_dir：结果图像及其对齐比率的目录。
    ref_output_dir 和 res_output_dir：存储输出图像（经过色调映射处理）的目录。
"""
# if not osp.exists(ref_output_dir):
#     os.mkdir(ref_output_dir)

if not osp.exists(res_output_dir):
    os.mkdir(res_output_dir)
"""检查结果输出目录是否存在，如果不存在则创建该目录。"""

for filename in sorted(os.listdir(ref_dir)):
    image_id = int(filename[:4])
    print(image_id)
    """
    列出参考图像目录中的文件，并按名称排序。
    提取文件名前四个字符作为 image_id（图像ID）。
    """
    hdr_image = io.imread_uint16_png(osp.join(ref_dir, "{:04d}_gt.png".format(image_id)), osp.join(ref_alignratio_dir, "{:04d}_alignratio.npy".format(image_id)))
    hdr_linear_image = hdr_image ** 2.24
    norm_perc = np.percentile(hdr_linear_image, 99)
    """
    使用自定义的 io.imread_uint16_png 函数读取HDR图像和对应的对齐比率文件。
    将图像线性化，使用幂运算 ** 2.24。
    计算HDR图像的99百分位数值，以用于后续的色调映射。
    """
    # hdr_image = (m.tanh_norm_mu_tonemap(hdr_linear_image, norm_perc) * 255.).round().astype(np.uint8)
    # cv2.imwrite(osp.join(ref_output_dir, "{:04d}_tone_mapped_gt.png".format(image_id)),  cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))

    # res_image = cv2.cvtColor(cv2.imread(osp.join(res_dir, "{:04d}_medium.png".format(image_id))), cv2.COLOR_BGR2RGB) / 255.
    res_image = io.imread_uint16_png(osp.join(res_dir, "{:04d}.png".format(image_id)), osp.join(res_alignratio_dir, "{:04d}_alignratio.npy".format(image_id)))
    res_linear_image = res_image ** 2.24
    res_image = (m.tanh_norm_mu_tonemap(res_linear_image, norm_perc) * 255.).round().astype(np.uint8)
    cv2.imwrite(osp.join(res_output_dir, "{:04d}_tone_mapped_result.png".format(image_id)),  cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))
    """
    读取结果图像并进行线性化处理。
    应用色调映射算法 tanh_norm_mu_tonemap，并将结果归一化到0-255范围，转换为8位无符号整数。
    使用OpenCV的 cv2.imwrite 将处理后的图像保存为PNG格式，并确保颜色格式转换为BGR（OpenCV默认格式）。
    """
    """整个流程是读取高动态范围图像，进行线性化处理和色调映射，然后将结果保存到指定目录。"""