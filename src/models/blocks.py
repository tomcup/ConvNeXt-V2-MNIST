"""
ConvNeXt V2 的基础构建块
包含：LayerNorm2d, DropPath, Block, ConvNeXtStage
适配 28×28 灰度输入，支持动态下采样和通道变换
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """对通道维度执行 LayerNorm (与 ConvNeXt 官方实现一致)"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # x: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        return x


class DropPath(nn.Module):
    """随机丢弃整个样本（Stochastic Depth），用于残差连接"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        return x / keep_prob * random_tensor


class Block(nn.Module):
    """
    ConvNeXt V2 基本块
    结构：Depthwise Conv → LayerNorm → 1x1 Conv (4x扩维) → GELU → 1x1 Conv (降维)
          → LayerScale → DropPath，最后残差连接
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # LayerScale（可学习的逐通道缩放因子）
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma.reshape(1, -1, 1, 1)

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """
    一个 ConvNeXt 阶段，包含可选的下采样层和多个 Block
    - 若 stride != 1 或输入输出通道不同，则先用 2×2 卷积下采样+通道变换
    - 随后堆叠 depth 个 Block（均保持 out_ch 维度）
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        kernel_size: int = 7,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # 下采样层
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=stride, bias=False),
            )
        else:
            self.downsample = nn.Identity()

        # 构建 block 序列，随机深度线性增加
        blocks = []
        for i in range(depth):
            # 线性递增的 drop path rate
            dp = drop_path_rate * i / (depth - 1) if depth > 1 else drop_path_rate
            blocks.append(Block(dim=out_ch, kernel_size=kernel_size, drop_path=dp))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x
