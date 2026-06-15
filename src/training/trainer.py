"""
训练器模块
负责完整的训练循环，包括动态采样、混合精度（可选）、EMA 更新、验证和检查点保存。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .loss import RobustLoss
from .sampler import ResumableRandomSubsetSampler
from ..utils import Logger, save_checkpoint, load_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        train_dataset,          # 完整数据集对象（mmap）
        val_dataset,            # 验证集
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: RobustLoss,
        device: torch.device,
        logger: Logger,
        cfg: dict,
    ):
        self.model = model
        self.ema_model = ema_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.cfg = cfg

        # 训练参数
        train_cfg = cfg['training']
        self.epochs = train_cfg['epochs']
        self.val_every = train_cfg.get('val_every_n_epochs', 5)
        self.gradient_clip_norm = train_cfg.get('gradient_clip_norm', None)
        self.use_compile = train_cfg.get('use_compile', False)
        self.resume_from = train_cfg.get('resume_from', None)

        # 数据加载参数
        data_cfg = cfg['data']
        self.batch_size = data_cfg['batch_size']
        self.num_workers = data_cfg['num_workers']
        self.pin_memory = data_cfg['pin_memory']
        self.train_samples_per_epoch = data_cfg['train_samples_per_epoch']

        # EMA 衰减率
        reg_cfg = cfg['regularization']
        self.ema_decay = reg_cfg['ema_decay']

        # 检查点目录
        log_cfg = cfg['logging']
        self.checkpoint_dir = log_cfg['checkpoint_dir']
        self.save_every_n_epochs = log_cfg.get('save_every_n_epochs', 10)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 训练采样器 & DataLoader
        self.train_sampler = ResumableRandomSubsetSampler(
            total_size=len(train_dataset),
            num_samples=self.train_samples_per_epoch
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2,
        )
        # 验证集 DataLoader（也复用）
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

        # 状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        
        self.use_amp = train_cfg.get('mixed_precision', False)
        self.amp_dtype = train_cfg.get('amp_dtype', 'float16')
        
        if self.amp_dtype == 'bfloat16':
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.grad_scaler.GradScaler(device=self.device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # 可选：torch.compile 加速
        if self.use_compile:
            self.model = torch.compile(self.model)

    def train(self):
        """主训练循环，支持从断点恢复"""
        # 恢复训练（如果指定）
        if self.resume_from:
            self.current_epoch, self.best_val_acc = load_checkpoint(
                self.resume_from, self.model, self.ema_model,
                self.optimizer, self.scheduler, str(self.device)
            )
            print(f"Resumed from epoch {self.current_epoch}, best_val_acc={self.best_val_acc:.4f}")

        for epoch in range(self.current_epoch, self.epochs):
            start_time = time.time()

            # 动态采样训练子集
            train_loader = self._get_train_loader(epoch)

            # 训练一个 epoch
            avg_train_loss = self._train_epoch(train_loader, epoch)

            # 验证
            if (epoch + 1) % self.val_every == 0 or epoch == self.epochs - 1:
                val_loss, val_acc = self._validate(epoch)
                self.logger.log_scalar("val/loss", val_loss, epoch)
                self.logger.log_scalar("val/accuracy", val_acc, epoch)
                print(f"Epoch {epoch+1:03d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    save_checkpoint(
                        save_path=os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}"),
                        model=self.model,
                        ema_model=self.ema_model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        best_val_acc=self.best_val_acc,
                        is_best=True,
                    )
                    print(f"  => New best model saved (acc={val_acc:.4f})")
            else:
                # 非验证轮次仍定期保存一次检查点（用于恢复）
                if (epoch + 1) % self.save_every_n_epochs == 0:
                    save_checkpoint(
                        save_path=os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}"),
                        model=self.model,
                        ema_model=self.ema_model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        best_val_acc=self.best_val_acc,
                    )

            # 学习率调度（epoch 级）
            self.scheduler.step()

            epoch_time = time.time() - start_time
            val_loss, val_acc = None, None
            if (epoch + 1) % self.val_every == 0 or epoch == self.epochs - 1:
                val_loss, val_acc = self._validate(epoch)

            # 写入 tensorboard（不打印）
            if self.logger.writer:
                self.logger.writer.add_scalar('train/loss', avg_train_loss, epoch + 1)
                self.logger.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch + 1)
                if val_loss is not None:
                    self.logger.writer.add_scalar('val/loss', val_loss, epoch + 1)
                    self.logger.writer.add_scalar('val/accuracy', val_acc, epoch + 1)

            # 构建一行简洁的日志输出
            log_parts = [f"Epoch {epoch+1:03d}",
                        f"Train Loss: {avg_train_loss:.4f}",
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"]
            if val_loss is not None:
                log_parts.append(f"Val Loss: {val_loss:.4f}")
                log_parts.append(f"Val Acc: {val_acc:.2f}%")
            log_parts.append(f"Time: {epoch_time:.1f}s")
            print(" | ".join(log_parts))


        # 训练结束，保存最终模型
        save_checkpoint(
            save_path=os.path.join(self.checkpoint_dir, "final"),
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epochs - 1,
            best_val_acc=self.best_val_acc,
        )
        self.logger.close()
        print("Training completed.")

    def _get_train_loader(self, epoch: int) -> DataLoader:
        self.train_sampler.resample()
        return self.train_loader

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(loader)
        pbar = tqdm(enumerate(loader), total=num_batches,
                    desc=f"Epoch {epoch+1:03d}", unit='batch', leave=False)
        for batch_idx, (images, targets) in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with torch.amp.autocast_mode.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            self._update_ema()
            total_loss += loss.item()

            # 更新进度条描述，显示当前 batch loss 和平局 loss
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })

        return total_loss / num_batches

    def _validate(self, epoch: int) -> tuple:
        """使用 EMA 模型进行验证，返回 (loss, accuracy)"""
        self.ema_model.eval()
        val_loader = self.val_loader

        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self.ema_model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _update_ema(self):
        """指数移动平均更新 EMA 模型"""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                if param.requires_grad:
                    ema_param.data.mul_(self.ema_decay).add_(
                        param.data, alpha=1 - self.ema_decay
                    )