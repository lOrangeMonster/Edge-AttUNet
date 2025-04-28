import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r',encoding='utf-8') as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if opt['distortion'] == 'sr':
            dataset['scale'] = scale
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    root_base = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if opt['path']['root'] is not None:
         opt['path']['root'] = osp.join(root_base, opt['path']['root'])
    else:
        opt['path']['root'] = root_base
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    if opt['distortion'] == 'sr':
        opt['network_G']['scale'] = scale

    return opt

"""
此函数读取由 opt_path 指定的配置文件，并根据是否为训练模式 (is_train) 进行相应的路径和设置调整。主要执行的操作有：
从文件读取配置信息。
设置环境变量 CUDA_VISIBLE_DEVICES 以指定使用的 GPU 设备。
设置配置项 is_train。
处理数据集相关路径，转换相对路径为绝对路径，并检测是否使用 LMDB 数据库。
根据训练或测试模式，调整实验、模型存储、日志等路径。
如果在调试模式下，调整日志打印和检查点保存频率。
更新网络配置中的 scale 参数（如果失真类型为 sr）。
"""

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

#此函数递归地将字典转换为适合打印的日志格式的字符串，便于查看配置详情。

class NoneDict(dict):
    def __missing__(self, key):
        return None

#定义了一个继承自 dict 的子类，当访问不存在的键时返回 None 而不是抛出异常。

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
#此函数递归地将普通字典转换为 NoneDict 实例，确保后续逻辑中访问不到的键不会引发错误。

def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
#该函数检查恢复训练的状态，并更新预训练模型的路径，确保恢复训练时能够正确加载之前保存的状态。
# 此外，如果在恢复训练模式下还指定了预训练模型路径，则会发出警告，因为恢复训练将忽略这些路径。

#总结：
#这段代码的主要目的是为了动态地调整并完善训练或测试所需的配置信息，使得在不同的环境下都能够正确运行深度学习模型的训练或评估任务。
# 通过这些函数，用户可以根据实际需求灵活地更改配置选项，同时保证代码的健壮性和灵活性。