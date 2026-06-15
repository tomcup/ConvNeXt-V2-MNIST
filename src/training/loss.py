"""
损失函数模块
支持硬标签（类别索引）和软标签（概率分布）的交叉熵损失。
软标签模式用于 MixUp/CutMix 等批量混合策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustLoss(nn.Module):
    """
    统一的鲁棒损失函数
    - 当 targets 为硬标签 (dtype=torch.long, shape=(batch,)) 时，
      使用标准 CrossEntropyLoss，可通过 label_smoothing 软化目标。
    - 当 targets 为软标签 (dtype=torch.float, shape=(batch, num_classes)) 时，
      直接计算交叉熵: -sum(target * log_softmax(output)) / batch_size。
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        # 仅用于硬标签的标准交叉熵损失
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 硬标签：一维整数张量
        if targets.dtype == torch.long and targets.ndim == 1:
            return self.ce(outputs, targets)
        # 软标签：二维浮点张量，形状 (batch_size, num_classes)
        elif targets.dtype in (torch.float16, torch.float32, torch.float64) and \
             targets.ndim == 2 and targets.shape == outputs.shape:
            log_probs = F.log_softmax(outputs, dim=-1)
            return -(targets * log_probs).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unsupported targets shape {targets.shape} and dtype {targets.dtype}")


def build_loss(cfg: dict) -> RobustLoss:
    """从配置字典构建损失函数"""
    label_smoothing = cfg.get("regularization", {}).get("label_smoothing", 0.0)
    return RobustLoss(label_smoothing=label_smoothing)