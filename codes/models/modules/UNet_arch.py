import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.arch_util as arch_util
from ..CBAM import CBAM

class HDRUNet(nn.Module):
    # 此处有修改通道数
    def __init__(self, in_nc=1, out_nc=1, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.SFT_layer1 = arch_util.SFTLayer()
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        # 在每个下采样后添加CBAM
        self.cbam1 = CBAM(channel_in=nf)
        self.cbam2 = CBAM(channel_in=nf)
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        # 在每个上采样前添加CBAM
        self.cbam3 = CBAM(channel_in=nf)
        self.cbam4 = CBAM(channel_in=nf)
        self.SFT_layer2 = arch_util.SFTLayer()
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        # 此处有修改通道数
        cond_in_nc=1
        cond_nf=64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 3, 2, 1))

        self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), 
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 3, 1, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, out_nc, 1),
                                     )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        """
        参数说明：
        in_nc：输入图像的颜色通道数，默认为3（RGB）。
        out_nc：输出图像的颜色通道数，默认也为3。
        nf：网络中的特征图数量，默认为64。
        act_type：激活函数类型，默认为ReLU。
        成员变量初始化：
        conv_first：第一个卷积层，用于从输入图像提取初始特征。
        SFT_layer1 和 SFT_layer2：空间频率变换层（Spatial Frequency Transform Layer），用于条件性特征变换。
        HR_conv1 和 HR_conv2：高分辨率卷积层。
        down_conv1 和 down_conv2：下采样卷积层。
        recon_trunk1, recon_trunk2, recon_trunk3：残差块序列，用于特征重建。
        up_conv1 和 up_conv2：上采样卷积层，使用像素shuffle操作。
        conv_last：最后的卷积层，用于生成最终输出。
        cond_first, CondNet1, CondNet2, CondNet3：条件网络，用于处理条件输入，生成不同层次的条件特征。
        mask_est：用于估计掩码的网络分支。
        激活函数：
        根据 act_type 参数选择合适的激活函数实例。
        """

    def forward(self, x):
        mask = self.mask_est(x[0])

        cond = self.cond_first(x[1])
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1 = self.cbam1(fea1)  # 应用CBAM
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        fea2 = self.cbam2(fea2)  # 应用CBAM
        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2

        # 在上采样之前应用CBAM
        out = self.cbam3(out)  # 应用CBAM
        out = self.act(self.up_conv1(out)) + fea1
        out, _ = self.recon_trunk3((out, cond2))

        # 再次在上采样之前应用CBAM
        out = self.cbam4(out)  # 应用CBAM
        out = self.act(self.up_conv2(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask * x[0] + out
        return out
    """
    输入参数：
    x：一个包含两个元素的列表，x[0] 是输入图像，x[1] 是条件输入。
    处理流程：
    掩码估计：
    使用 mask_est 网络分支从输入图像估计掩码。
    条件特征提取：
    使用 cond_first 层处理条件输入，然后通过 CondNet1, CondNet2, CondNet3 生成不同层次的条件特征。
    特征提取与下采样：
    通过 conv_first 层提取输入图像的特征。
    应用 SFT 层和 HR 卷积层，然后通过 SFT 层结合条件特征。
    经过两次下采样卷积，每次下采样后通过残差块进行特征重建。
    特征重建与上采样：
    使用上采样卷积层逐步恢复特征分辨率，并与之前保存的特征相加。
    再次通过残差块进行特征重建，并通过 SFT 层结合条件特征。
    最终输出：
    通过最后一个卷积层生成最终输出。
    将输出与输入图像按掩码进行融合，生成最终结果。
    """

    #总结
    #这个 HDRUNet 类实现了一个具有条件输入的 U-Net 结构，用于图像处理任务，特别是涉及 HDR 图像的处理。
    #通过使用 SFT 层和多尺度条件特征，模型能够在处理过程中有效地利用条件信息。此外，掩码估计分支允许模型在输出时考虑输入图像的重要区域。
    #这种设计使得模型能够在保持图像细节的同时，有效处理复杂的图像修复或增强任务。