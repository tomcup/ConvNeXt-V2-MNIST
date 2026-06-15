"""
训练模块初始化
集中导出 Trainer、损失函数、优化器构建函数
"""

from .trainer import Trainer
from .loss import RobustLoss, build_loss
from .optimizer import build_optimizer_and_scheduler