import logging
from typing import Any, Dict, List, Tuple
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils.preprocessing import (
    _count_affnist,
    _count_image_folder,
    _load_affnist,
    _load_image_folder,
    preprocess_image,
    _count_idx,
    _load_idx,
    _count_npy_directory,
    _load_npy_directory,
)


def process_sources(
    sources: List[Dict[str, Any]], output_dir: Path, img_name: str, lbl_name: str
) -> Tuple[int, Path, Path]:
    """
    处理所有数据源，生成统一的 .npy 文件。

    Args:
        sources: 数据源配置列表，每项包含 'type', 'mode', 及其他必要字段。
        output_dir: 输出目录。
        img_name: 图像文件名。
        lbl_name: 标签文件名。

    Returns:
        (total_samples, images_path, labels_path)
    """
    total = 0
    source_infos = []

    for src in sources:
        src_type = src["type"]
        mode = src.get("mode", "labeled")
        path = src.get("path", None)

        if src_type == "idx":
            imf = src["images_file"]
            lbf = src.get("labels_file")
            cnt = _count_idx(imf)

            def gen_func(imf=imf, lbf=lbf):
                return _load_idx(imf, lbf)
        elif src_type == "npy_dir":
            p = Path(path)
            split = src.get("split", "train")
            cnt = _count_npy_directory(p, split)

            def gen_func(p=p, s=split):
                return _load_npy_directory(p, s)
        elif src_type == "image_folder":
            p = Path(path)
            cnt = _count_image_folder(p)

            def gen_func(p=p):
                return _load_image_folder(p)
        elif src_type == "affnist":
            p = Path(path)
            if p.is_dir():
                mat_files = sorted(p.glob("*.mat"))
            else:
                mat_files = [p]
            cnt = _count_affnist(mat_files)

            def gen_func(mf=mat_files):
                return _load_affnist(mf)
        else:
            raise ValueError(f"Unsupported source type: {src_type}")

        source_infos.append((cnt, gen_func, mode))
        total += cnt

    logging.info(f"Total estimated samples: {total}")

    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / img_name
    lbl_path = output_dir / lbl_name

    img_mm = np.memmap(str(img_path), dtype=np.uint8, mode="w+", shape=(total, 28, 28))
    lbl_mm = np.memmap(str(lbl_path), dtype=np.int64, mode="w+", shape=(total,))

    current_idx = 0
    for cnt, gen_func, mode in source_infos:
        for img, label in gen_func():
            try:
                img_proc = preprocess_image(img)
            except Exception as e:
                logging.warning(f"Failed to preprocess image: {e}. Skipping.")
                continue
            img_mm[current_idx] = img_proc
            lbl_mm[current_idx] = label if mode == "labeled" else -1
            current_idx += 1
            if current_idx % 100000 == 0:
                logging.info(f"Processed {current_idx}/{total}")

    actual = current_idx
    logging.info(f"Actual processed samples: {actual}")

    img_mm.flush()
    lbl_mm.flush()

    img_data = np.array(img_mm[:actual])
    lbl_data = np.array(lbl_mm[:actual])
    del img_mm, lbl_mm

    np.save(str(img_path), img_data)
    np.save(str(lbl_path), lbl_data)

    logging.info("Preprocessing finished.")
    return actual, img_path, lbl_path


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/preprocessing.yaml",
        help="Path to preprocessing config file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output_dir", "processed"))
    img_name = config.get("img_name", "train_images.npy")
    lbl_name = config.get("lbl_name", "train_labels.npy")

    total_samples, images_path, labels_path = process_sources(
        config["data_sources"], output_dir, img_name, lbl_name
    )

    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Images saved to: {images_path}")
    logging.info(f"Labels saved to: {labels_path}")
