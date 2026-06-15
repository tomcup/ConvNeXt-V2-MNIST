"""
数据预处理核心模块 — 将异构 MNIST 风格数据集转换为统一的 28×28 .npy 文件。
不包含 torchvision 依赖，所有数据源均从本地读取。
"""

import gzip
import struct
from pathlib import Path
from typing import Iterator, List, Tuple, Any, Optional, Dict

import numpy as np

# 可选依赖
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


# ======================= IDX 格式（MNIST 原始格式） =======================

def _read_idx_header(filepath: str) -> Tuple[np.dtype, Tuple[int, ...]]:
    """读取 IDX 文件头，返回 (dtype, dims)"""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        dtype_code = magic // 256
        ndim = magic % 256
        dims = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        dtype = {
            0x08: np.uint8, 0x09: np.int8, 0x0B: np.int16,
            0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64
        }.get(dtype_code, np.uint8)
        return dtype, dims


def _count_idx(images_file: str) -> int:
    """IDX 文件中的图像数量"""
    _, dims = _read_idx_header(images_file)
    return dims[0]


def _load_idx(images_file: str, labels_file: Optional[str] = None) -> Iterator[Tuple[np.ndarray, int]]:
    """生成器：逐样本产生 (image, label)，无标签时 label = -1"""
    def read_full(filepath):
        opener = gzip.open if filepath.endswith('.gz') else open
        with opener(filepath, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            ndim = magic % 256
            dims = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
            dtype_code = magic // 256
            dtype = {
                0x08: np.uint8, 0x09: np.int8, 0x0B: np.int16,
                0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64
            }.get(dtype_code, np.uint8)
            data = np.frombuffer(f.read(), dtype=dtype).reshape(dims)
            return data

    images = read_full(images_file)
    labels = read_full(labels_file).ravel().astype(np.int64) if labels_file else np.full(len(images), -1, dtype=np.int64)
    for i in range(len(images)):
        yield images[i], labels[i]


# ======================= Npy 目录格式（MNIST-C 风格） =======================

def _count_npy_directory(root: Path, split: str) -> int:
    """统计 MNIST-C 风格目录中的样本总数"""
    total = 0
    for subdir in root.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        img_file = subdir / f"{split}_images.npy"
        if img_file.exists():
            arr = np.load(img_file, mmap_mode='r')
            total += len(arr)
    return total


def _load_npy_directory(root: Path, split: str) -> Iterator[Tuple[np.ndarray, int]]:
    """生成器：逐样本产生 (image, label)"""
    for subdir in root.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        img_file = subdir / f"{split}_images.npy"
        lbl_file = subdir / f"{split}_labels.npy"
        if img_file.exists() and lbl_file.exists():
            images = np.load(img_file, mmap_mode='r')
            labels = np.load(lbl_file, mmap_mode='r').ravel().astype(np.int64)
            if images.ndim == 4 and images.shape[-1] == 1:
                images = images.squeeze(-1)
            m = min(len(images), len(labels))
            for i in range(m):
                yield images[i], labels[i]


# ======================= ImageFolder 格式（图片文件夹） =======================

def _count_image_folder(root: Path) -> int:
    """统计 ImageFolder 结构下的图片数量"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    count = 0
    for class_dir in root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in exts:
                count += 1
    return count


def _load_image_folder(root: Path) -> Iterator[Tuple[np.ndarray, int]]:
    """生成器：逐张读取 ImageFolder 图片"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    for class_dir in root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in exts:
                if Image is None:
                    raise ImportError("Pillow is required for image folder datasets")
                img = Image.open(img_path).convert('L')
                yield np.array(img, dtype=np.uint8), label


# ======================= affNIST (.mat) 格式 =======================

def _count_affnist(mat_paths: List[Path]) -> int:
    """统计 affNIST .mat 文件中的样本总数"""
    if loadmat is None:
        raise ImportError("scipy is required for affNIST (.mat files)")
    total = 0
    for mp in mat_paths:
        data = loadmat(str(mp))['affNISTdata'][0, 0]
        labels = data['label_int'] if 'label_int' in data.dtype.names else data['label']
        total += len(labels.ravel())
    return total


def _load_affnist(mat_paths: List[Path]) -> Iterator[Tuple[np.ndarray, int]]:
    """生成器：逐张产生 (40,40) affNIST 图像和标签"""
    if loadmat is None:
        raise ImportError("scipy is required for affNIST (.mat files)")
    for mp in mat_paths:
        data = loadmat(str(mp))['affNISTdata'][0, 0]
        images = data['image']
        labels = data['label_int'] if 'label_int' in data.dtype.names else data['label']
        labels = labels.ravel().astype(np.int64)

        # 标准化形状为 (N, 40, 40)
        if images.ndim == 1:
            if images.shape[0] % 1600 != 0:
                raise ValueError(f"Cannot reshape flat array of size {images.shape[0]}")
            images = images.reshape(-1, 40, 40)
        elif images.ndim == 2:
            if images.shape[0] == 1600 and images.shape[1] > 0:
                images = images.T.reshape(-1, 40, 40)
            elif images.shape[1] == 1600 and images.shape[0] > 0:
                images = images.reshape(-1, 40, 40)
            elif images.shape[0] == 40 and images.shape[1] == 40:
                images = images[np.newaxis, ...]
            else:
                raise ValueError(f"Unexpected 2D image shape: {images.shape}")
        elif images.ndim != 3:
            raise ValueError(f"Unexpected image shape: {images.shape}")

        # 统一为 uint8 [0,255]
        if images.dtype == np.uint8:
            pass
        elif images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)

        for i in range(len(images)):
            yield images[i], labels[i]


# ======================= 图像预处理工具 =======================

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    将任意尺寸的灰度图像转为 28×28 uint8 数组。
    输入可以是 uint8 或 float，值域 [0,255] 或 [0,1]。
    """
    if img.dtype == np.uint8:
        pass
    elif img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    if img.shape != (28, 28):
        if Image is None:
            raise ImportError("Pillow is required to resize images")
        pil_img = Image.fromarray(img, mode='L')
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        img = np.array(pil_img, dtype=np.uint8)
    return img