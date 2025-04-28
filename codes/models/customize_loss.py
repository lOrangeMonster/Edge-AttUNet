import torch
import torch.nn as nn
import torch.nn.functional as F

class tanh_L1Loss(nn.Module):
    def __init__(self, edge_loss_weight=0.3):
        super(tanh_L1Loss, self).__init__()
        self.edge_loss_weight = edge_loss_weight  # 边缘损失的权重
        self.tanh_L1Loss_origin = tanh_L1Loss_origin()  # 初始化原有的损失函数
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, y):
        # 确保 x 和 y 的形状相同
        if x.shape != y.shape:
            raise ValueError(f"Shapes of x and y do not match. x shape: {x.shape}, y shape: {y.shape}")

        # 计算原始的 tanh_L1Loss_origin
        l1_loss_origin = self.tanh_L1Loss_origin(x, y)
        # 计算梯度图
        edges_x = torch.sqrt(torch.clamp(F.conv2d(torch.tanh(x), self.sobel_x, padding=1) ** 2 +
                                         F.conv2d(torch.tanh(x), self.sobel_y, padding=1) ** 2, min=1e-6))
        edges_y = torch.sqrt(torch.clamp(F.conv2d(torch.tanh(y), self.sobel_x, padding=1) ** 2 +
                                         F.conv2d(torch.tanh(y), self.sobel_y, padding=1) ** 2, min=1e-6))

        # print("Max value of x before tanh:", x.max())
        # print("Min value of x before tanh:", x.min())
        # print("Max value of y before tanh:", y.max())
        # print("Min value of y before tanh:", y.min())

        # 计算边缘损失作为 L1 距离
        edge_loss = torch.mean(torch.abs(edges_x - edges_y))

        # 组合两种损失
        total_loss = (1-self.edge_loss_weight)*l1_loss_origin + self.edge_loss_weight * edge_loss
        # print("L1 Loss:", l1_loss_origin.item())
        # print("Edge Loss:", edge_loss.item())
        # print("Total Loss:", total_loss.item())
        return total_loss


class tanh_L1Loss_origin(nn.Module):

    def __init__(self):
        super(tanh_L1Loss_origin, self).__init__()

    def forward(self, x, y):
        # 打印 x 和 y 的形状
        # print("Shape of x:", x.shape)
        # print("Shape of y:", y.shape)

        # 确保 x 和 y 的形状相同
        if x.shape != y.shape:
            raise ValueError(f"Shapes of x and y do not match. x shape: {x.shape}, y shape: {y.shape}")

        # 计算损失
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss
    """
    用途：计算两个输入张量 x 和 y 经过 tanh 变换后的 L1 距离（绝对误差均值）。
    成员变量：无。
    初始化方法：调用父类 nn.Module 的初始化方法。
    前向传播方法：
    输入：
    x：第一个输入张量。
    y：第二个输入张量。
    处理：
    对 x 和 y 分别应用 tanh 函数。
    计算 tanh(x) 和 tanh(y) 之间的逐元素绝对差。
    对得到的结果求平均值作为损失值。
    输出：
    返回计算得到的损失值。
    """


    # def __init__(self):
    #     super(tanh_L1Loss, self).__init__()
    # def forward(self, x, y):
    #     # 打印 x 和 y 的形状
    #     # print("Shape of x:", x.shape)
    #     # print("Shape of y:", y.shape)
    #
    #     # 确保 x 和 y 的形状相同
    #     if x.shape != y.shape:
    #         raise ValueError(f"Shapes of x and y do not match. x shape: {x.shape}, y shape: {y.shape}")
    #
    #     # 计算损失
    #     loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
    #     return loss

class tanh_L2Loss(nn.Module):
    def __init__(self):
        super(tanh_L2Loss, self).__init__()
    def forward(self, x, y):
        # 打印 x 和 y 的形状
        print("Shape of x:", x.shape)
        print("Shape of y:", y.shape)

        # 确保 x 和 y 的形状相同
        if x.shape != y.shape:
            raise ValueError(f"Shapes of x and y do not match. x shape: {x.shape}, y shape: {y.shape}")

        # 计算损失
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        return loss

    """
    用途：计算两个输入张量 x 和 y 经过 tanh 变换后的 L2 距离（平方误差均值）。
    成员变量：无。
    初始化方法：调用父类 nn.Module 的初始化方法。
    前向传播方法：
    输入：
    x：第一个输入张量。
    y：第二个输入张量。
    处理：
    对 x 和 y 分别应用 tanh 函数。
    计算 tanh(x) 和 tanh(y) 之间的逐元素差的平方。
    对得到的结果求平均值作为损失值。
    输出：
    返回计算得到的损失值。
    """

    """
    这两个类定义了两种不同的损失函数，它们都是基于 tanh 激活函数的输出来进行损失计算的。
    tanh 函数是一种非线性激活函数，它的输出范围是 [-1, 1]，因此在某些场景下可能比普通的线性变换更适合用于计算损失。
    例如，在图像处理中，像素值通常被归一化到 [0, 1] 或者 [-1, 1] 的范围内，使用 tanh 函数可以更好地适应这种数据范围。
    """