import torch
import logging

import models.modules.UNet_arch as UNet_arch
logger = logging.getLogger('base')
#这里导入了一个名为 UNet_arch 的模块，该模块包含了网络架构的定义，并创建了一个名为 'base' 的日志记录器实例。

####################
# define network
####################
#### Generator
def define_G(opt):
    #此函数接收一个字典 opt 作为参数，该字典包含了创建生成器网络所需的信息，例如网络类型、输入通道数、输出通道数等。
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    #解析配置选项:从配置选项 opt 中提取生成器网络相关的配置，并从中获取生成器的类型 which_model。
    if which_model == 'HDRUNet':
        netG = UNet_arch.HDRUNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], act_type=opt_net['act_type'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG
    #创建网络实例:根据 which_model 的值来决定创建哪种类型的生成器网络。
    # 如果 which_model 是 'HDRUNet'，则从 UNet_arch 模块中导入 HDRUNet 类，并使用给定的参数创建一个实例。
    # 如果 which_model 不是 'HDRUNet'，则抛出一个异常，表明不支持指定的生成器类型。
    #函数最后返回创建好的生成器网络实例。

"""

示例配置选项
为了更好地理解这段代码的工作原理，以下是一个可能的配置选项 opt 的示例：
opt = {
    'network_G': {
        'which_model_G': 'HDRUNet',
        'in_nc': 3,  # 输入通道数
        'out_nc': 3, # 输出通道数
        'nf': 64,    # 特征通道数
        'act_type': 'relu' # 激活函数类型
    }
}
当使用上述配置选项调用 define_G(opt) 函数时，将会创建一个 HDRUNet 网络实例，具有 3 个输入通道、3 个输出通道、64 个特征通道，并使用 ReLU 作为激活函数。
"""