import logging
logger = logging.getLogger('base')
#这里导入了 Python 的标准库模块 logging，用于提供日志记录的功能。
def create_model(opt):
    model = opt['model']
    
    if model == 'base':
        pass
    elif model == 'condition':
        from .Generation_condition import GenerationModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    #此函数接收一个字典 opt 作为参数，该字典包含了创建模型所需的信息，例如模型的类型。
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
"""
根据 opt 中的 'model' 键值来决定应该创建哪种类型的模型。如果模型类型是 'base'，则直接跳过不做任何处理（通常意味着默认行为或者预留的接口）。
如果是 'condition'，则从相对模块路径 .
Generation_condition 导入名为 GenerationModel 的类，并使用别名 M。
如果提供的模型类型不在预期范围内，则抛出 NotImplementedError 异常。
"""