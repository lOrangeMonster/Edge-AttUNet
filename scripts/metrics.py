import numpy as np
"""这段代码的功能主要是对高动态范围（HDR）图像进行色调映射，并计算图像之间的峰值信噪比（PSNR）。"""
def mu_tonemap(hdr_image, mu=5000):
    """ This function computes the mu-law tonemapped image of a given input linear HDR image.
        功能：对HDR图像应用mu-law色调映射，压缩图像值。
        输入：
        hdr_image：表示HDR图像的numpy数组（值在0到1之间）。
        mu：控制压缩程度的参数（默认值为5000）。
        返回值：经过色调映射处理的图像。
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        mu (float): Parameter controlling the compression performed during tone mapping.

    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.

    """
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value and then computes
    the mu-law tonemapped image.
        功能：在应用mu-law色调映射之前，对HDR图像进行归一化处理。
        输入：
        hdr_image：输入的HDR图像。
        norm_value：归一化因子。
        mu：压缩参数。
        返回值：经过归一化和色调映射处理的图像。
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
        mu (float): Parameter controlling the compression performed during tone mapping.

    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.

    """
    return mu_tonemap(hdr_image/norm_value, mu)

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value, afterwards bounds the
    HDR image values by applying a tanh function and afterwards computes the mu-law tonemapped image.

            功能：对HDR图像进行归一化、应用双曲正切（tanh）函数限制值范围，然后进行mu-law色调映射。
            输入：与norm_mu_tonemap相似。
            返回值：经过限制和色调映射处理的图像。

        the mu-law tonemapped image.
        Args:
            hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
            norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
            mu (float): Parameter controlling the compression performed during tone mapping.

        Returns:
            np.ndarray (): Returns the mu-law tonemapped image.

        """
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    """ This function computes Peak Signal to Noise Ratio (PSNR) between the mu-law computed images from two non-linear
    HDR images.

            功能：计算两个HDR图像之间的PSNR，先进行伽玛校正和mu-law色调映射。
            输入：
            hdr_nonlinear_ref：参考HDR图像。
            hdr_nonlinear_res：待比较的HDR图像。
            percentile：确定归一化水平。
            gamma：伽玛校正因子。
            返回值：PSNR值的平均值。

            Args:
                hdr_nonlinear_ref (np.ndarray): HDR Reference Image after gamma correction, used for the percentile norm
                hdr_nonlinear_res (np.ndarray: HDR Estimated Image after gamma correction
                percentile (float): Percentile to to use for normalization
                gamma (float): Value used to linearized the non-linear images

            Returns:
                np.ndarray (): Returns the mean mu-law PSNR value for the complete image.

            """
    hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))


def psnr(im0, im1):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        the mu-law tonemapped image.
            功能：计算两个图像之间的PSNR。
            输入：
            im0：第一幅图像。
            im1：第二幅图像。
            返回值：两幅图像的平均PSNR值。
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0

        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.

        """
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

def normalized_psnr(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.

        功能：计算两个归一化图像之间的PSNR。
        输入：
        im0：第一幅图像。
        im1：第二幅图像。
        norm：归一化值。
        返回值：归一化图像的平均PSNR值。

        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.

        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.

        """
    return psnr(im0/norm, im1/norm)