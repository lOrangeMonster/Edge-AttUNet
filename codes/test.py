import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

import numpy as np


# 代码是用来处理图像超分辨率（Super-Resolution, SR）任务的一部分，它从配置文件中读取选项，并根据这些选项创建数据集、数据加载器以及模型实例。
# 然后使用创建的模型对测试数据集中的每一张图片进行超分辨率处理，并将结果保存下来。
if __name__ == '__main__':
    #### options
    print("Starting main function...")  # 添加这一行
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []

        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

            # 打印原始张量形状
            if 'LQ' in data:
                logger.info(f"Original LQ shape: {data['LQ'].shape}")
            if 'GT' in data:
                logger.info(f"Original GT shape: {data['GT'].shape}")
            # 检查并调整 LQ 和 GT 的形状
            if 'LQ' in data and data['LQ'].dim() == 5:
                data['LQ'] = data['LQ'].squeeze(4)  # 删除最后一个维度
            if 'GT' in data and data['GT'].dim() == 5:
                data['GT'] = data['GT'].squeeze(4)  # 删除最后一个维度

            # 确保张量形状正确
            if 'LQ' in data:
                assert data['LQ'].dim() == 4, f"LQ tensor must be 4D, got {data['LQ'].shape}"
            if 'GT' in data:
                assert data['GT'].dim() == 4, f"GT tensor must be 4D, got {data['GT'].shape}"

            # 再次打印调整后的张量形状
            if 'LQ' in data:
                logger.info(f"Adjusted LQ shape: {data['LQ'].shape}")
            if 'GT' in data:
                logger.info(f"Adjusted GT shape: {data['GT'].shape}")

            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            sr_img = util.tensor2numpy(visuals['SR'])
            image_path, alignratio_path = util.generate_paths(dataset_dir, img_name)
            util.save_img_with_ratio(image_path, sr_img, alignratio_path)

            logger.info('{:20s}'.format(img_name))


# 测试流程
# 对于每一个测试数据加载器，进行如下操作：
# 记录开始测试的时间。
# 创建用于保存测试结果的目录。
# 初始化一个有序字典来存储测试结果，如 PSNR（峰值信噪比）。
# 遍历数据加载器中的每一个批次：
# 判断是否需要 ground truth 图像。
# 使用模型的 feed_data 方法输入数据。
# 获取图像的路径，并从中提取图像的名字。
# 使用模型的 test 方法进行前向传播。
# 获取模型输出的视觉结果。
# 将超分辨率图像转换成 numpy 数组并保存到磁盘。
# 记录处理过的图像名字。
# 日志输出
# 在处理过程中，会记录每一张图像的名字，以便跟踪进度。
