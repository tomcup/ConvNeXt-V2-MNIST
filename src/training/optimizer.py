"""
优化器与学习率调度器构建
支持 AdamW + 余弦退火 + 线性 warmup
"""

import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_and_scheduler(model, cfg: dict):
    """
    从配置字典构建优化器和学习率调度器。

    返回:
        (optimizer, scheduler)
    """
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    train_cfg = cfg["training"]

    # ---------- 优化器参数 ----------
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg["weight_decay"])
    betas = tuple(opt_cfg["betas"])

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    # ---------- 学习率调度参数 ----------
    min_lr = float(sched_cfg["min_lr"])
    warmup_epochs = int(sched_cfg["warmup_epochs"])
    total_epochs = int(train_cfg["epochs"])

    # 学习率乘子函数（epoch 从 0 开始）
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # 线性 warmup：从 0 线性增加到 1
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            # 余弦退火：从 1 下降到 min_lr / lr
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + (lr - min_lr) * cosine_decay) / lr

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
