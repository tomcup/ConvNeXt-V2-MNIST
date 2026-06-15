"""
推理脚本：加载训练好的 EMA 模型，对测试集执行 TTA 推理，输出 CSV 提交文件。
用法：
    python scripts/infer.py --config config/default.yaml [--checkpoint path/to/best_ema.pth] [--output submission.csv]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.inference.predictor import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run inference on test set with TTA")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to training/inference configuration YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output CSV (overrides config)')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 命令行参数覆盖配置
    if args.checkpoint:
        cfg.setdefault('inference', {})['checkpoint'] = args.checkpoint
    if args.output:
        cfg.setdefault('inference', {})['output_file'] = args.output

    # 执行推理
    run_inference(cfg)


if __name__ == '__main__':
    main()