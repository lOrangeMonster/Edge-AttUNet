import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from progress_bar import ProgressBar
"""
    这段代码是一个用于处理图像的 Python 脚本，具体功能是将指定文件夹中的图像进行裁剪，并将裁剪后的图像保存到另一个文件夹中。
    设置路径和参数：
    input_folder：原始图像所在文件夹。
    save_folder：裁剪后图像保存的文件夹。
    n_thread：并行处理的线程数。
    crop_sz：裁剪的大小（480x480）。
    step：裁剪的步幅（240）。
    thres_sz：边缘裁剪的阈值。
    compression_level：PNG 图像的压缩等级。
    检查输出文件夹：
    如果保存文件夹不存在，则创建它；如果已存在，则退出程序。
    
    收集图像文件：
    使用 os.walk 遍历输入文件夹，收集所有图像文件的路径。
    
    进度条初始化：
    创建进度条的实例 pbar。
    
    并行处理：
    使用 Pool 创建一个进程池，并对每个图像路径调用 worker 函数进行处理。
    
    等待所有进程完成：
    关闭进程池并等待所有任务完成。
"""

def main():
    """A multii-thread tool to crop sub imags."""
    # input_folder = r'E:\tool\HDRUNet-main\data0\000_Valid_SingleFrame_FirstStage\medium'
    # save_folder = r'E:\tool\HDRUNet-main\data0\000_Valid_SingleFrame_FirstStage\cutmedium'
    input_folder = r'E:\tool\HDRUNet-main\data0\000_single_valid_1122\gt'
    save_folder = r'E:\tool\HDRUNet-main\data0\000_single_valid_1122\cutgt'
    n_thread = 20
    crop_sz = 480   # crop size
    step = 240  # crop stride
    thres_sz = 48
    compression_level = 0  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)
        #for file_name in file_list:
        #    if os.path.splitext(file_name)[1] == '.png':
        #        img_list.append(os.path.join(root, file_name))

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
            args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    #img_name = '_'.join(path.split('/')[-4:])

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            cv2.imwrite(
                os.path.join(save_folder, img_name.replace('.tif', '_s{:03d}.tif'.format(index))),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()

