"""
错误分析脚本：使用训练好的模型对验证集/测试集（需有标签）进行推理，
保存预测错误的样本图像，并生成错误报告 CSV。
用法：
    python scripts/error_analysis.py --config config/default.yaml \
        --checkpoint checkpoints/best_ema.pth \
        --output_dir error_analysis
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from torchvision.utils import save_image

# 项目根目录导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.convnextv2_femto import ConvNeXtV2FemtoMNIST
from src.data_utils.dataset import PreloadedDataset
from src.data_utils.augmentations import build_val_transform
from src.inference.tta import TTAPredictor
import yaml


import base64
from io import BytesIO
from PIL import Image as PILImage

def create_error_html(errors, images_tensor, indices, output_path):
    """
    生成错误样本的 HTML 大表。
    errors: list of (index_in_batch, true_label, pred_label)
    images_tensor: list of tensors (1, 28, 28) uint8，与 all_images 对应
    indices: list of global indices (对应原始数据集索引)
    """
    html = ['<html><head><meta charset="utf-8"><title>错误样本分析</title>',
            '<style>table { border-collapse: collapse; } td { padding: 10px; text-align: center; border: 1px solid #ccc; } img { width: 56px; height: 56px; }</style>',
            '</head><body><table>']
    
    # 每行放8个样本，可自行调节
    num_cols = 8
    for row_start in range(0, len(errors), num_cols):
        html.append('<tr>')
        row_errors = errors[row_start:row_start + num_cols]
        for idx_in_list, true_label, pred_label in row_errors:
            img_tensor = images_tensor[idx_in_list]  # (1, 28, 28) uint8
            # 转为 PIL 并保存为 base64
            pil_img = PILImage.fromarray(img_tensor[0].numpy(), mode='L')
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            img_src = f"data:image/png;base64,{img_b64}"
            
            # 单元格内容
            html.append(f'<td><img src="{img_src}"><br>真实: {true_label}<br>预测: {pred_label}<br><small>原索引: {indices[idx_in_list]}</small></td>')
        html.append('</tr>')
    
    html.append('</table></body></html>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
        
def main():
    parser = argparse.ArgumentParser(description="Analyze model errors")
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径')
    parser.add_argument('--output_dir', default='error_analysis', help='输出目录')
    parser.add_argument('--use_tta', action='store_true', default=True, help='是否使用 TTA')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['training'].get('device', 'cuda'))
    print(f"Device: {device}")

    # 加载模型
    model = ConvNeXtV2FemtoMNIST()
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 准备验证集（有标签）
    data_cfg = cfg['data']
    val_transform = build_val_transform(cfg)
    val_dataset = PreloadedDataset(
        images_path=data_cfg['val_images_path'],
        labels_path=data_cfg['val_labels_path'],
        transform=val_transform,
        soft_labels=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # TTA 预测器（可选）
    tta = None
    if args.use_tta:
        tta_cfg = cfg['inference']['tta']
        tta = TTAPredictor(
            num_views=tta_cfg.get('num_views', 12),
            scale_range=tta_cfg.get('scale_range', (0.9, 1.1)),
            translate=tta_cfg.get('translate', 2),
            rotation=tta_cfg.get('rotation', 10.0),
            horizontal_flip=False
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / 'error_images'
    img_dir.mkdir(exist_ok=True)

    all_preds = []
    all_labels = []
    all_images = []
    indices = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
            images_gpu = images.to(device)
            if tta:
                probs = tta.predict(model, images_gpu)
            else:
                outputs = model(images_gpu)
                probs = torch.softmax(outputs, dim=-1)

            preds = probs.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            # 保存原始图像（反归一化，转为 uint8 以便保存）
            # 注意：数据加载时归一化使用了 mean=0.5, std=0.5，需要反归一化
            imgs = images.cpu() * 0.5 + 0.5  # 回到 [0,1]
            imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
            all_images.extend([imgs[i] for i in range(imgs.size(0))])
            # 记录全局索引（便于对应验证集原始索引）
            start_idx = batch_idx * args.batch_size
            indices.extend([start_idx + i for i in range(len(labels))])

    # 找出错误样本
    errors = []
    for i, (true, pred) in enumerate(zip(all_labels, all_preds)):
        if true != pred:
            errors.append((i, true, pred))

    print(f"Total samples: {len(all_labels)}, Errors: {len(errors)} "
          f"({100.0 * len(errors) / len(all_labels):.2f}%)")

    # 保存错误图像和报告
    # records = []
    # for idx, true_label, pred_label in errors:
    #     img = all_images[idx]  # (1, 28, 28) uint8
    #     # 保存图像
    #     file_name = f"idx{indices[idx]}_true{true_label}_pred{pred_label}.png"
    #     save_image(img.float() / 255.0, img_dir / file_name)  # save_image 期望 float [0,1]
    #     records.append({
    #         'original_index': indices[idx],
    #         'true_label': true_label,
    #         'predicted_label': pred_label,
    #         'image_file': file_name
    #     })

    # df = pd.DataFrame(records)
    # df.to_csv(output_dir / 'error_report.csv', index=False)
    # print(f"Saved {len(errors)} error images to {img_dir}")
    # print(f"Error report saved to {output_dir / 'error_report.csv'}")
    
    # 在错误保存后生成 HTML 表格
    create_error_html(errors, all_images, indices, output_dir / 'error_grid.html')
    print(f"HTML 错误表已生成: {output_dir / 'error_grid.html'}")


if __name__ == '__main__':
    main()