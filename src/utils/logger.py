"""
训练日志模块
支持 TensorBoard、Wandb 及控制台输出。
"""

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """统一的日志记录器，可同时写入 TensorBoard 和 Wandb，并输出到控制台"""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        run_name: str = "convnext_mnist",
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)

        if self.use_wandb:
            try:
                import wandb

                wandb.init(project="robust-mnist", name=run_name, dir=log_dir)
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Disabling wandb logging.")
                self.use_wandb = False

        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Logger] Started at {self.start_time}")

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量指标"""
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            self.wandb.log({tag: value}, step=step)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Step {step}: {tag} = {value:.4f}")

    def log_image(self, tag: str, image_tensor, step: int):
        """记录图像（可选）"""
        if self.use_tensorboard and self.writer:
            self.writer.add_image(tag, image_tensor, step)

    def close(self):
        """关闭资源"""
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
        print("[Logger] Closed.")


def build_logger(cfg: dict) -> Logger:
    """从配置字典构建 Logger"""
    log_cfg = cfg["logging"]
    model_name = cfg.get("model", {}).get("name", "exp")
    return Logger(
        log_dir=log_cfg["log_dir"],
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        use_wandb=log_cfg.get("use_wandb", False),
        run_name=model_name,
    )
