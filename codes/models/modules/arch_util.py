import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
        """
        用途：初始化网络中的权重。
        参数：
        net_l：要初始化的网络或网络列表。
        scale：权重初始化的缩放因子，默认为1。
        实现：遍历网络的所有模块，根据模块类型使用不同的初始化方式。
        对于 nn.Conv2d 和 nn.Linear 使用 Kaiming 正态分布初始化，并乘以 scale；对于 nn.BatchNorm2d 使用常数初始化。
        """
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

        # 用途：创建由多个相同类型的层组成的序列。
        # 参数：
        # block：要重复的层类型。
        # n_layers：重复的次数。
        # 实现：创建一个包含 n_layers 个 block 类型的 nn.Sequential 对象。
        # 类定义

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
        # 用途：定义一个没有批量归一化的残差块。
        # 成员变量：
        # conv1 和 conv2：两个卷积层。
        # 初始化：使用 initialize_weights 函数初始化 conv1 和 conv2。
        # 前向传播：输入经过卷积和 ReLU 激活，然后与输入相加形成残差连接。
class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift
        # 用途：定义一个空间频率转换层（Spatial Modulation of Activations），用于条件特征变换。
        # 成员变量：
        # SFT_scale_conv0 和 SFT_scale_conv1：用于计算尺度因子的卷积层。
        # SFT_shift_conv0 和 SFT_shift_conv1：用于计算位移因子的卷积层。
        # 前向传播：输入特征图乘以计算出的尺度因子加上位移因子。
class ResBlock_with_SFT(nn.Module):
    def __init__(self, nf=64):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.sft1 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft1(x)
        fea = F.relu(self.conv1(fea), inplace=True)
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])

        # 用途：定义一个带有 SFT 层的残差块。
        # 成员变量：
        # conv1 和 conv2：两个卷积层。
        # sft1 和 sft2：两个 SFT 层。
        # 初始化：使用 initialize_weights 函数初始化 conv1 和 conv2。
        # 前向传播：输入先通过 SFT 层进行变换，再经过卷积和 ReLU 激活，然后再次通过 SFT 层，最后与输入相加形成残差连接。


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

    # 用途：使用光流场对图像或特征图进行变形。
    # 参数：
    # x：待变形的图像或特征图。
    # flow：光流场，用于描述每个像素的位移。
    # interp_mode：插值模式，默认为双线性插值。
    # padding_mode：填充模式，默认为零填充。
    # 实现：首先根据输入图像的尺寸创建网格，然后根据光流场更新网格坐标，并将其映射到 [-1, 1] 范围内，最后使用 F.grid_sample 对输入图像进行变形。

    # 段代码提供了用于构建深度学习模型的一些基础组件，特别是针对图像处理任务。
    # 通过定义不同的网络层和模块，如残差块、SFT 层等，可以灵活地构建复杂模型。
    # 此外，flow_warp 函数提供了使用光流场对图像进行变形的能力，这对于视频处理、图像修复等领域非常有用。
    # 这些组件可以作为更复杂模型的一部分，应用于多种图像处理任务中。
