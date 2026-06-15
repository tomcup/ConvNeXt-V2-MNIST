"""
模型检查点保存与加载模块
支持训练中断恢复、EMA 权重保存
"""

import os
import torch


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: object = None,
    epoch: int = 0,
    best_val_acc: float = 0.0,
    is_best: bool = False,
):
    """保存完整训练状态检查点
    Args:
        save_path: 基础保存路径（不含扩展名）
        model: 原始模型
        ema_model: EMA 模型（可选）
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前 epoch 编号
        best_val_acc: 当前最佳验证准确率
        is_best: 是否同时保存为 best 模型（覆盖 best.pth）
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
    }
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # 保存 epoch 检查点
    filepath = f"{save_path}.pth"
    torch.save(checkpoint, filepath)

    # 如果是最佳模型，额外保存一份（仅模型权重，方便推理）
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_ema.pth")
        # 推理时只需要 EMA 权重（或普通权重）
        if ema_model is not None:
            torch.save(ema_model.state_dict(), best_path)
        else:
            torch.save(model.state_dict(), best_path)


def load_checkpoint(
    load_path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: object = None,
    device: str = "cuda",
):
    """加载检查点恢复训练，返回起始 epoch 和最佳验证准确率
    Args:
        load_path: 检查点文件路径
        model: 模型
        ema_model: EMA 模型（可选）
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    Returns:
        start_epoch: 恢复后应从哪个 epoch 开始
        best_val_acc: 已记录的最佳验证准确率
    """
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    return start_epoch, best_val_acc