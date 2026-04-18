import argparse
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


YoloBox = Tuple[int, float, float, float, float]
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"


def read_image_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def write_image_unicode(path: Path, image: np.ndarray) -> None:
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"无法编码图像: {path}")
    encoded.tofile(str(path))


def load_yolo_labels(label_path: Path) -> List[YoloBox]:
    if not label_path.exists():
        return []

    boxes: List[YoloBox] = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            boxes.append((cls, x, y, w, h))
    return boxes


def save_yolo_labels(label_path: Path, boxes: Sequence[YoloBox]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def yolo_box_to_corners(box: YoloBox, width: int, height: int) -> Tuple[int, np.ndarray]:
    cls, x, y, w, h = box
    cx = x * width
    cy = y * height
    bw = w * width
    bh = h * height
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    return cls, corners


def corners_to_yolo_box(cls: int, corners: np.ndarray, width: int, height: int) -> Optional[YoloBox]:
    xs = corners[:, 0]
    ys = corners[:, 1]

    x1 = float(np.clip(xs.min(), 0, width - 1))
    y1 = float(np.clip(ys.min(), 0, height - 1))
    x2 = float(np.clip(xs.max(), 0, width - 1))
    y2 = float(np.clip(ys.max(), 0, height - 1))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 2.0 or bh < 2.0:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return (
        cls,
        float(np.clip(cx / width, 0.0, 1.0)),
        float(np.clip(cy / height, 0.0, 1.0)),
        float(np.clip(bw / width, 0.0, 1.0)),
        float(np.clip(bh / height, 0.0, 1.0)),
    )


def transform_boxes(
    boxes: Sequence[YoloBox],
    width: int,
    height: int,
    matrix: np.ndarray,
    perspective: bool,
) -> List[YoloBox]:
    out: List[YoloBox] = []

    for box in boxes:
        cls, corners = yolo_box_to_corners(box, width, height)
        points = corners.reshape(-1, 1, 2)
        if perspective:
            transformed = cv2.perspectiveTransform(points, matrix)
        else:
            transformed = cv2.transform(points, matrix)
        transformed_corners = transformed.reshape(-1, 2)
        converted = corners_to_yolo_box(cls, transformed_corners, width, height)
        if converted is not None:
            out.append(converted)

    return out


def random_brightness(image: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.8, 1.25)
    beta = rng.uniform(-15, 15)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def rotate_small(image: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w = image.shape[:2]
    angle = rng.uniform(-12.0, 12.0)
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    out = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out, m, False


def flip_horizontal(image: np.ndarray, _: random.Random) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w = image.shape[:2]
    out = cv2.flip(image, 1)
    m = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return out, m, False


def flip_vertical(image: np.ndarray, _: random.Random) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w = image.shape[:2]
    out = cv2.flip(image, 0)
    m = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h - 1.0]], dtype=np.float32)
    return out, m, False


def slight_perspective(image: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w = image.shape[:2]
    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )

    max_shift = 0.03 * min(h, w)
    dst = src.copy()
    dst[:, 0] += np.array(
        [
            rng.uniform(-max_shift, max_shift),
            rng.uniform(-max_shift, max_shift),
            rng.uniform(-max_shift, max_shift),
            rng.uniform(-max_shift, max_shift),
        ],
        dtype=np.float32,
    )
    dst[:, 1] += np.array(
        [
            rng.uniform(-max_shift, max_shift),
            rng.uniform(-max_shift, max_shift),
            rng.uniform(-max_shift, max_shift),
            rng.uniform(-max_shift, max_shift),
        ],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        image,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return out, m, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对训练图像执行数据增强，并同步 YOLO 标签")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DATASET_DIR / "images" / "train",
        help="训练图片目录，默认 datasets/images/train",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DATASET_DIR / "labels" / "train",
        help="训练标签目录（YOLO txt），默认 datasets/labels/train",
    )
    parser.add_argument("--output-train-dir", type=Path, default=None, help="增强后图片输出目录")
    parser.add_argument("--output-label-dir", type=Path, default=None, help="增强后标签输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--include-originals", action="store_true", help="输出中包含原始样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_dir = args.train_dir
    if not train_dir.exists():
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")

    label_dir = args.label_dir
    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")

    output_train_dir = (
        args.output_train_dir if args.output_train_dir is not None else DATASET_DIR / "images" / "train_aug"
    )
    output_label_dir = (
        args.output_label_dir if args.output_label_dir is not None else DATASET_DIR / "labels" / "train_aug"
    )

    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    classes_txt = label_dir / "classes.txt"
    if classes_txt.exists():
        (output_label_dir / "classes.txt").write_text(classes_txt.read_text(encoding="utf-8"), encoding="utf-8")

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(train_dir.glob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise RuntimeError(f"未在目录中找到图片: {train_dir}")

    transforms: List[Tuple[str, Callable[[np.ndarray, random.Random], Tuple[np.ndarray, np.ndarray, bool]]]] = [
        ("flip_h", flip_horizontal),
        ("flip_v", flip_vertical),
        ("rotate", rotate_small),
        ("persp", slight_perspective),
    ]

    saved_images = 0
    saved_labels = 0

    for image_path in image_paths:
        image = read_image_unicode(image_path)
        h, w = image.shape[:2]
        stem = image_path.stem
        suffix = image_path.suffix

        label_path = label_dir / f"{stem}.txt"
        boxes = load_yolo_labels(label_path)

        if args.include_originals:
            out_image_path = output_train_dir / f"{stem}{suffix}"
            write_image_unicode(out_image_path, image)
            saved_images += 1
            if boxes:
                out_label_path = output_label_dir / f"{stem}.txt"
                save_yolo_labels(out_label_path, boxes)
                saved_labels += 1

        for idx, (tag, transform_fn) in enumerate(transforms, start=1):
            transformed, matrix, perspective = transform_fn(image, rng)
            transformed = random_brightness(transformed, rng)

            if boxes:
                new_boxes = transform_boxes(boxes, w, h, matrix, perspective)
            else:
                new_boxes = []

            aug_stem = f"{stem}_aug{idx}_{tag}"
            out_image_path = output_train_dir / f"{aug_stem}{suffix}"
            write_image_unicode(out_image_path, transformed)
            saved_images += 1

            if new_boxes:
                out_label_path = output_label_dir / f"{aug_stem}.txt"
                save_yolo_labels(out_label_path, new_boxes)
                saved_labels += 1

    print(f"处理完成: 原图数量={len(image_paths)}, 输出图片={saved_images}, 输出标签={saved_labels}")
    print(f"增强图片目录: {output_train_dir}")
    print(f"增强标签目录: {output_label_dir}")
    if classes_txt.exists():
        print(f"已复制类别文件: {output_label_dir / 'classes.txt'}")


if __name__ == "__main__":
    main()
