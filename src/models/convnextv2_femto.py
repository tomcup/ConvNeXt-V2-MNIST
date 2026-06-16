"""
ConvNeXt V2 Femto 定制版 —— 针对 28×28 单通道 MNIST
根据 default.yaml 中的模型结构实现
"""

import torch
import torch.nn as nn

from .blocks import LayerNorm2d, ConvNeXtStage


class ConvNeXtV2FemtoMNIST(nn.Module):
    """
    ConvNeXt V2 Femto 适配 28×28 灰度输入
    - Stem: Conv3x3 → LayerNorm → GELU (保持 28×28)
    - 4个阶段，2次下采样（28→14→7）
    - 头：LayerNorm → GlobalAvgPool → Dropout → Linear(384, 10)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        stem_out_channels: int = 48,
        stages_config: list = None,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        # ---------- Stem ----------
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(stem_out_channels),
            nn.GELU(),
        )

        # ---------- 构建阶段 ----------
        if stages_config is None:
            # 默认配置（来自 default.yaml）
            stages_config = [
                {
                    "stage_index": 1,
                    "in_channels": 48,
                    "out_channels": 48,
                    "depth": 2,
                    "kernel_size": 7,
                    "stride": 1,
                    "drop_path_rate": 0.0,
                },
                {
                    "stage_index": 2,
                    "in_channels": 48,
                    "out_channels": 96,
                    "depth": 2,
                    "kernel_size": 7,
                    "stride": 2,
                    "drop_path_rate": 0.05,
                },
                {
                    "stage_index": 3,
                    "in_channels": 96,
                    "out_channels": 192,
                    "depth": 4,
                    "kernel_size": 7,
                    "stride": 2,
                    "drop_path_rate": 0.1,
                },
                {
                    "stage_index": 4,
                    "in_channels": 192,
                    "out_channels": 384,
                    "depth": 2,
                    "kernel_size": 3,
                    "stride": 1,
                    "drop_path_rate": 0.15,
                },
            ]

        self.stages = nn.ModuleList()
        in_ch = stem_out_channels
        for cfg in stages_config:
            stage = ConvNeXtStage(
                in_ch=cfg["in_channels"],
                out_ch=cfg["out_channels"],
                depth=cfg["depth"],
                kernel_size=cfg["kernel_size"],
                stride=cfg["stride"],
                drop_path_rate=cfg["drop_path_rate"],
            )
            self.stages.append(stage)
            in_ch = cfg["out_channels"]  # 下一阶段的输入

        final_dim = in_ch  # 应为 384

        # ---------- 头 ----------
        self.norm = LayerNorm2d(final_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(final_dim, num_classes)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.head(x)
        return x
