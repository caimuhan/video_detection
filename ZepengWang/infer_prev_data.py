#!/usr/bin/env python3
"""
对 data/prev_data 中的图片做分割推理，结果写回同目录（*_seg.png / *_compare.png）。
不修改 yolo_seg.py；复用其中的可视化与推理超参。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import yolo_seg
from yolo_seg import IMGSZ, visualize_results


def main():
    parser = argparse.ArgumentParser(description="对 prev_data 推理，输出保存在同目录")
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=yolo_seg.SCRIPT_DIR / "data" / "prev_data",
        help="待分割图片目录（默认 data/prev_data）",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=yolo_seg.SCRIPT_DIR / "runs" / "yolo_seg" / "weights" / "best.pt",
        help="训练得到的权重",
    )
    parser.add_argument("--device", type=str, default=yolo_seg.DEVICE)
    args = parser.parse_args()

    yolo_seg.sanitize_ultralytics_fonts()

    img_dir: Path = args.img_dir.resolve()
    wpath: Path = args.weights.resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"目录不存在: {img_dir}")
    if not wpath.is_file():
        raise SystemExit(f"权重不存在: {wpath}")

    model = YOLO(str(wpath))
    patterns = ("*.jpg", "*.png", "*.jpeg")
    test_images: list[Path] = []
    for pat in patterns:
        test_images.extend(sorted(img_dir.glob(pat)))
    test_images = [p for p in test_images if p.parent == img_dir]

    print(f"[推理] 权重: {wpath}")
    print(f"[推理] 目录: {img_dir}，共 {len(test_images)} 张\n")

    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=0.25,
            iou=0.5,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        image_bgr = cv2.imread(str(img_path))
        vis, count = visualize_results(image_bgr, result)

        out_seg = img_dir / f"{img_path.stem}_seg.png"
        cv2.imwrite(str(out_seg), vis)

        h0, w0 = image_bgr.shape[:2]
        w800, h800 = 800, int(800 * h0 / w0)
        compare = np.hstack(
            [
                cv2.resize(image_bgr, (w800, h800)),
                cv2.resize(vis, (w800, h800)),
            ]
        )
        out_cmp = img_dir / f"{img_path.stem}_compare.png"
        cv2.imwrite(str(out_cmp), compare)

        print(f"  {img_path.name}: {count} 个实例 → {out_seg.name}, {out_cmp.name}")

    print(f"\n[完成] 输出目录: {img_dir}")


if __name__ == "__main__":
    main()
