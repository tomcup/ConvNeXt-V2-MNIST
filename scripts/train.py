"""
训练脚本：全监督训练 ConvNeXt V2 Femto，使用动态叠加增强和 EMA。
用法：
    python scripts/train.py --config config/default.yaml [--device cuda] [--resume path/to/checkpoint.pth]
"""

import argparse
import sys
from pathlib import Path

# 确保可以导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch

from src.models.convnextv2_femto import ConvNeXtV2FemtoMNIST
from src.data_utils.dataset import PreloadedDataset
from src.data_utils.augmentations import build_train_transform, build_val_transform
from src.training.trainer import Trainer
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer_and_scheduler
from src.utils import Logger, build_logger, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt V2 Femto on augmented MNIST")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Configuration YAML')
    parser.add_argument('--device', type=str, default=None, help='Device override (e.g. cuda:0, cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 设备
    if args.device:
        cfg.setdefault('training', {})['device'] = args.device
    device = torch.device(cfg['training'].get('device', 'cuda'))
    print(f"Using device: {device}")

    # 固定随机种子
    set_seed(cfg['training'].get('seed', 42))

    # 日志
    logger = build_logger(cfg)

    # 构建数据加载器（使用预处理后的 .npy 文件）
    data_cfg = cfg['data']
    train_transform = build_train_transform(cfg)
    val_transform = build_val_transform(cfg)

    train_dataset = PreloadedDataset(
        images_path=data_cfg['train_images_path'],
        labels_path=data_cfg['train_labels_path'],
        transform=train_transform,
        soft_labels=False
    )
    val_dataset = PreloadedDataset(
        images_path=data_cfg['val_images_path'],
        labels_path=data_cfg['val_labels_path'],
        transform=val_transform,
        soft_labels=False
    )

    # 模型
    model = ConvNeXtV2FemtoMNIST().to(device)
    ema_model = ConvNeXtV2FemtoMNIST().to(device)
    ema_model.load_state_dict(model.state_dict())  # 初始化为相同权重

    # 损失函数
    loss_fn = build_loss(cfg)

    # 优化器与调度器
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # 检查点路径（可在配置中指定或使用默认）
    if args.resume:
        cfg['training']['resume_from'] = args.resume

    # 创建 Trainer 并开始训练
    trainer = Trainer(
        model=model,
        ema_model=ema_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        logger=logger,
        cfg=cfg,
    )

    trainer.train()


if __name__ == '__main__':
    main()