"""
测试时增强（TTA）模块
对输入图像生成多个几何扰动视图，平均 softmax 概率以提高鲁棒性。
"""

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import affine


class TTAPredictor:
    """
    测试时增强推理器
    参数：
        num_views: 视图数量（默认12）
        scale_range: (min, max) 随机缩放范围
        translate: 平移像素绝对值（例如2表示±2px）
        rotation: 旋转角度范围（例如10表示±10°）
        horizontal_flip: 是否水平翻转（手写数字禁止）
    """

    def __init__(
        self,
        num_views: int = 12,
        scale_range: tuple = (0.9, 1.1),
        translate: int = 2,
        rotation: float = 10.0,
        horizontal_flip: bool = False,
    ):
        self.num_views = num_views
        self.scale_range = scale_range
        self.translate = translate
        self.rotation = rotation
        self.horizontal_flip = horizontal_flip

    @torch.no_grad()
    def predict(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        对批量图像执行 TTA 推理，返回平均类别概率。
        Args:
            model: 评估模式下的分类模型，输出 logits
            images: (B, C, H, W) 已归一化的图像张量
        Returns:
            probs: (B, num_classes) 平均 softmax 概率
        """
        model.eval()
        B = images.size(0)
        all_probs = []  # 收集每个视图的概率

        # 如果模型需要复制到同设备，已经在外部处理
        device = next(model.parameters()).device
        images = images.to(device)

        for _ in range(self.num_views):
            # 为每个视图生成不同的随机变换参数
            view = self._apply_random_transform(images)
            logits = model(view)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        # 平均所有视图的概率
        stacked = torch.stack(all_probs, dim=0)  # (views, B, num_classes)
        avg_probs = stacked.mean(dim=0)  # (B, num_classes)
        return avg_probs

    def _apply_random_transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        对批量图像应用一次随机几何变换。
        每张图像的变换参数独立随机采样。
        Args:
            images: (B, C, H, W)
        Returns:
            transformed: (B, C, H, W)
        """
        B = images.size(0)
        device = images.device
        transformed = []
        for i in range(B):
            img = images[i]  # (C, H, W)
            # 随机缩放因子
            scale = self.scale_range[0] + torch.rand(1, device=device).item() * (
                self.scale_range[1] - self.scale_range[0]
            )
            # 随机平移（像素）
            tx = int(torch.randint(-self.translate, self.translate + 1, (1,)).item())
            ty = int(torch.randint(-self.translate, self.translate + 1, (1,)).item())
            # 随机旋转角度
            angle = (torch.rand(1).item() * 2 - 1) * self.rotation

            # 使用 affine 函数：输入 PIL 或 Tensor 均可
            # 这里输入是 Tensor (C, H, W)，所以需要填充值，例如0
            transformed_img = affine(
                img,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],  # 不使用剪切
                fill=[
                    0.0
                ],  # 归一化后均值0填充（因为mean=0.5，std=0.5，像素值范围约[-1,1]，0是中间值）
            )
            transformed.append(transformed_img)

        return torch.stack(transformed, dim=0)
