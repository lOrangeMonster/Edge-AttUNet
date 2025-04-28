import cv2
import numpy as np

def imread_uint16_png(image_path, alignratio_path):
    """ This function loads a uint16 png image from the specified path and restore its original image range with
    the ratio stored in the specified alignratio.npy respective path.
    功能
    这个函数从指定路径加载一个16位PNG图像，并使用一个对齐比例文件（.npy）恢复图像的原始范围。

    参数
    image_path (str): 指向16位PNG图像的路径。
    alignratio_path (str): 指向对应的对齐比例（.npy文件）的路径。
    返回值
    返回一个RGB格式的HDR图像，数据类型为np.float32，形状为(h, w, 3)，表示图像的高度、宽度和颜色通道。

    Args:
        image_path (str): Path to the uint16 png image
        alignratio_path (str): Path to the alignratio.npy file corresponding to the image

    Returns:
        np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    # Load the align_ratio variable and ensure is in np.float32 precision
    align_ratio = np.load(alignratio_path).astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio
    

def imwrite_uint16_png(image_path, image, alignratio_path):
    """ This function writes the hdr image as a uint16 png and stores its related align_ratio value in the specified paths.

        功能
        这个函数将HDR图像写入为16位PNG文件，并将其对应的对齐比例值保存到指定路径。

        参数
        image_path (str): 写入的PNG图像路径（必须以.png结尾）。
        image (np.ndarray): 以浮点格式表示的HDR图像。
        alignratio_path (str): 写入对齐比例值的路径（必须以.npy结尾）。
        返回值
        返回None。

        Args:
            image_path (str): Write path to the uint16 png image (needs to finish in .png, e.g. 0000.png)
            image (np.ndarray): HDR image in float format.
            alignratio_path (str): Write path to the align_ratio value (needs to finish in .npy, e.g. 0000_alignratio.npy)

        Returns:
            np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    align_ratio = (2 ** 16 - 1) / image.max()
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return None