import numpy as np  # 添加这一行以导入 numpy 并使用别名 np
from torchvision.transforms import functional as F
import random
from PIL import Image

class PairedTransform:
    def __init__(self, degrees=15, translate=None, scale=(0.9, 1.1)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, img_LQ, img_GT):
        angle = random.uniform(-self.degrees, self.degrees)
        if self.translate is not None:
            max_dx = self.translate[0] * img_LQ.width
            max_dy = self.translate[1] * img_LQ.height
            translations = (int(random.uniform(-max_dx, max_dx)),
                            int(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        scale_factor = random.uniform(self.scale[0], self.scale[1])
        shear = 0  # 如果需要添加剪切变换，可以在这里设置

        # 确保输入是 PIL 图像
        if isinstance(img_LQ, np.ndarray):
            img_LQ = Image.fromarray(np.uint8(img_LQ))
        if isinstance(img_GT, np.ndarray):
            img_GT = Image.fromarray(np.uint8(img_GT))

        img_LQ = F.affine(img_LQ, angle, translations, scale_factor, shear, fillcolor=0)
        img_GT = F.affine(img_GT, angle, translations, scale_factor, shear, fillcolor=0)

        return img_LQ, img_GT