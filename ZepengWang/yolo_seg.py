"""
HW3: 螺丝实例分割 — 基于 YOLOv8-seg 微调方案
使用 CVAT 标注的 Ultralytics YOLO Segmentation 数据进行训练与推理。

用法:
    conda run -n cv python yolo_seg.py          # 默认合并 IMG_2374/2375/2376 训练，并对三目录推理
    conda run -n cv python yolo_seg.py --datasets IMG_2376   # 只训一个子目录
    conda run -n cv python yolo_seg.py --predict           # 仅推理（默认目录见 --img_dir）
"""

import argparse
import shutil
from pathlib import Path
import zipfile

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ============================================================
# 路径 & 超参数
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output_yolo"
# 默认合并三个已标注视频子目录训练；仅推理时仍用 --img_dir
DEFAULT_DATASETS = "IMG_2374,IMG_2375,IMG_2376"
DEFAULT_IMAGE_DIR = DATA_ROOT / "images" / "IMG_2376"
DEFAULT_ANN_ZIP = DEFAULT_IMAGE_DIR / "annotation.zip"
SEG_DATASET_ROOT = DATA_ROOT / "seg_dataset"
UNPACK_ROOT = DATA_ROOT / "_ann_unpack"

MODEL_NAME = str(SCRIPT_DIR / "models" / "yolo" / "yolo11x-seg.pt")  # 本地权重
EPOCHS = 200
IMGSZ = 1024
BATCH = 4
PATIENCE = 50
DEVICE = "3"
SPLIT_SEED = 42
DEFAULT_VAL_RATIO = 0.2


# ============================================================
# 字体修复：保留绘图，避免坏字体导致 matplotlib 崩溃
# ============================================================

def sanitize_ultralytics_fonts():
    """
    清理 Ultralytics 配置目录中的损坏字体文件，避免绘图阶段报错:
    RuntimeError: Can not load face (unknown file format)
    """
    try:
        from ultralytics.utils import USER_CONFIG_DIR
        from matplotlib import ft2font
    except Exception as e:
        print(f"[字体检查] 跳过（依赖不可用）: {e}")
        return

    bad_fonts = []
    for ttf in USER_CONFIG_DIR.glob("*.ttf"):
        try:
            ft2font.FT2Font(str(ttf))
        except Exception:
            bad_fonts.append(ttf)

    for ttf in bad_fonts:
        ttf.unlink(missing_ok=True)
        print(f"[字体检查] 已移除损坏字体: {ttf}")

    if not bad_fonts:
        print("[字体检查] 字体文件正常")


# ============================================================
# 1. 数据准备：从 annotation.zip 构建 YOLO 所需目录结构
# ============================================================

def _safe_unzip(ann_zip: Path, dst_dir: Path):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ann_zip, "r") as zf:
        for member in zf.namelist():
            target = (dst_dir / member).resolve()
            if not str(target).startswith(str(dst_dir.resolve())):
                raise RuntimeError(f"发现不安全压缩路径: {member}")
        zf.extractall(dst_dir)


def _load_annotation_meta(unpack_dir: Path):
    data_yaml = unpack_dir / "data.yaml"
    train_txt = unpack_dir / "train.txt"
    label_dir = unpack_dir / "labels" / "train"

    if not data_yaml.exists():
        raise FileNotFoundError(f"annotation 缺少 data.yaml: {data_yaml}")
    if not train_txt.exists():
        raise FileNotFoundError(f"annotation 缺少 train.txt: {train_txt}")
    if not label_dir.exists():
        raise FileNotFoundError(f"annotation 缺少 labels/train: {label_dir}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        ann_cfg = yaml.safe_load(f)
    names = ann_cfg.get("names")
    if not names:
        raise RuntimeError("annotation data.yaml 中缺少 names 配置")

    with open(train_txt, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    stems = [Path(item).stem for item in items]

    return names, stems, label_dir


def _resolve_val_count(total: int, val_count: int | None, val_ratio: float | None) -> int:
    if total < 3:
        raise RuntimeError("有效样本过少，至少需要 3 张图像用于 train/val 划分")

    if val_count is None:
        ratio = DEFAULT_VAL_RATIO if val_ratio is None else val_ratio
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"val_ratio 必须在 (0, 1) 内，当前为 {ratio}")
        val_count = round(total * ratio)

    return max(1, min(int(val_count), total - 1))


def _split_temporal_block(items: list[str], val_ratio: float | None) -> tuple[list[str], list[str]]:
    ordered = sorted(items)
    val_count = _resolve_val_count(len(ordered), None, val_ratio)
    return ordered[:-val_count], ordered[-val_count:]


def _write_split_manifest(train_entries: list[str], val_entries: list[str]):
    manifests = {
        SEG_DATASET_ROOT / "split_train.txt": train_entries,
        SEG_DATASET_ROOT / "split_val.txt": val_entries,
    }
    for path, entries in manifests.items():
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(f"{entry}\n")


def prepare_dataset(
    ann_zip: Path,
    image_dir: Path,
    val_ratio: float | None = DEFAULT_VAL_RATIO,
):
    """
    从 annotation.zip 与图像目录构建 YOLO 分割数据集:
    - 自动解压 annotation
    - 按时间顺序切块：前段 train，后段 val
    - 复制图像与标签至 data/seg_dataset
    - 生成 data/seg_dataset/data.yaml
    """
    if not ann_zip.exists():
        raise FileNotFoundError(f"标注压缩包不存在: {ann_zip}")
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")

    unpack_dir = UNPACK_ROOT / ann_zip.stem
    _safe_unzip(ann_zip, unpack_dir)
    names, stems, label_src_dir = _load_annotation_meta(unpack_dir)
    stems = sorted(set(stems))
    train_stems, val_stems = _split_temporal_block(stems, val_ratio)
    val_stem_set = set(val_stems)

    if SEG_DATASET_ROOT.exists():
        shutil.rmtree(SEG_DATASET_ROOT)
    img_train_dir = SEG_DATASET_ROOT / "images" / "train"
    img_val_dir = SEG_DATASET_ROOT / "images" / "val"
    lbl_train_dir = SEG_DATASET_ROOT / "labels" / "train"
    lbl_val_dir = SEG_DATASET_ROOT / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img_candidates = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    img_map = {p.stem: p for p in img_candidates}

    copied = {"train": 0, "val": 0}
    for stem in train_stems + val_stems:
        src_img = img_map.get(stem)
        src_label = label_src_dir / f"{stem}.txt"
        if src_img is None:
            raise FileNotFoundError(f"未找到与标注匹配的图像: {stem} (目录: {image_dir})")
        if not src_label.exists():
            raise FileNotFoundError(f"未找到标注文件: {src_label}")

        split = "val" if stem in val_stem_set else "train"
        dst_img_dir = img_val_dir if split == "val" else img_train_dir
        dst_lbl_dir = lbl_val_dir if split == "val" else lbl_train_dir
        shutil.copy2(src_img, dst_img_dir / src_img.name)
        shutil.copy2(src_label, dst_lbl_dir / src_label.name)
        copied[split] += 1

    data_cfg = {
        "path": str(SEG_DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    cfg_path = SEG_DATASET_ROOT / "data.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _write_split_manifest(train_stems, val_stems)

    print(f"[数据准备] train 图片: {copied['train']}, val 图片: {copied['val']}")
    print(f"[数据准备] 划分方式: 时间块切分（前{copied['train']}张训练，后{copied['val']}张验证）")
    print(f"[数据准备] 类别映射: {names}")
    print(f"[数据准备] 配置文件: {cfg_path}")
    return cfg_path


def prepare_merged_dataset(
    image_ann_pairs: list[tuple[Path, Path]],
    val_ratio: float | None = DEFAULT_VAL_RATIO,
):
    """
    合并多个子目录（各含 annotation.zip 与同名的 jpg/png）构建单一 YOLO 数据集。
    每个子目录按时间顺序切块：前段 train，后段 val。
    """
    if not image_ann_pairs:
        raise ValueError("至少需要一组 (图像目录, annotation.zip)")

    train_rows: list[tuple[str, Path, Path]] = []
    val_rows: list[tuple[str, Path, Path]] = []
    names_ref = None

    for image_dir, ann_zip in image_ann_pairs:
        if not ann_zip.exists():
            raise FileNotFoundError(f"标注压缩包不存在: {ann_zip}")
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")

        unpack_dir = UNPACK_ROOT / f"{image_dir.name}_unpack"
        _safe_unzip(ann_zip, unpack_dir)
        names, stems, label_src_dir = _load_annotation_meta(unpack_dir)

        if names_ref is None:
            names_ref = names
        elif names_ref != names:
            raise RuntimeError(
                f"各数据集的 data.yaml 中 names 必须一致。\n"
                f"基准: {names_ref}\n{image_dir.name}: {names}"
            )

        stems = sorted(set(stems))
        train_stems, val_stems = _split_temporal_block(stems, val_ratio)
        train_stem_set = set(train_stems)
        val_stem_set = set(val_stems)
        img_candidates = (
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpeg"))
        )
        img_map = {p.stem: p for p in img_candidates}

        for stem in stems:
            src_img = img_map.get(stem)
            src_label = label_src_dir / f"{stem}.txt"
            if src_img is None:
                raise FileNotFoundError(f"未找到与标注匹配的图像: {stem} (目录: {image_dir})")
            if not src_label.exists():
                raise FileNotFoundError(f"未找到标注文件: {src_label}")
            row = (stem, src_img, src_label)
            if stem in train_stem_set:
                train_rows.append(row)
            elif stem in val_stem_set:
                val_rows.append(row)

    n = len(train_rows) + len(val_rows)

    if SEG_DATASET_ROOT.exists():
        shutil.rmtree(SEG_DATASET_ROOT)
    img_train_dir = SEG_DATASET_ROOT / "images" / "train"
    img_val_dir = SEG_DATASET_ROOT / "images" / "val"
    lbl_train_dir = SEG_DATASET_ROOT / "labels" / "train"
    lbl_val_dir = SEG_DATASET_ROOT / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0}
    for split, rows, dst_img_dir, dst_lbl_dir in [
        ("train", train_rows, img_train_dir, lbl_train_dir),
        ("val", val_rows, img_val_dir, lbl_val_dir),
    ]:
        for _, src_img, src_label in rows:
            new_stem = f"{src_img.parent.name}__{src_img.stem}"
            shutil.copy2(src_img, dst_img_dir / f"{new_stem}{src_img.suffix}")
            shutil.copy2(src_label, dst_lbl_dir / f"{new_stem}.txt")
            copied[split] += 1

    data_cfg = {
        "path": str(SEG_DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": names_ref,
    }
    cfg_path = SEG_DATASET_ROOT / "data.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _write_split_manifest(
        [f"{src_img.parent.name}/{src_img.name}" for _, src_img, _ in train_rows],
        [f"{src_img.parent.name}/{src_img.name}" for _, src_img, _ in val_rows],
    )

    print(f"[数据准备-合并] 总样本: {n}, train: {copied['train']}, val: {copied['val']}")
    print(f"[数据准备-合并] 划分方式: 各视频按时间块切分（前80%训练，后20%验证）")
    print(f"[数据准备-合并] 类别映射: {names_ref}")
    print(f"[数据准备-合并] 配置文件: {cfg_path}")
    return cfg_path


# ============================================================
# 2. 训练
# ============================================================

def train(
    cfg_path: Path,
    init_weight: Path,
    device: str,
    epochs_override=None,
    *,
    seed: int = SPLIT_SEED,
    deterministic: bool = True,
    resume: bool = False,
):
    model = YOLO(str(init_weight))
    if resume:
        print(f"[训练] 从断点续训: {init_weight}")
        results = model.train(resume=True, device=device)
    else:
        print(f"[训练] 初始化权重: {init_weight}")
        results = model.train(
            data=str(cfg_path),
            epochs=epochs_override if epochs_override is not None else EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            project=str(SCRIPT_DIR / "runs"),
            name="yolo_seg",
            exist_ok=True,
            device=device,
            workers=4,
            amp=False,
            seed=seed,
            deterministic=deterministic,
            augment=True,
            mosaic=1.0,
            flipud=0.5,
            fliplr=0.5,
            degrees=15.0,
            scale=0.3,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
        )
    best_weight = SCRIPT_DIR / "runs" / "yolo_seg" / "weights" / "best.pt"
    print(f"[训练完成] 最佳权重: {best_weight}")
    return best_weight


# ============================================================
# 3. 推理 & 可视化
# ============================================================

def visualize_results(image_bgr, result):
    """在原图上叠加彩色半透明 mask 和轮廓，返回可视化图像。"""
    overlay = image_bgr.copy()
    vis = image_bgr.copy()

    if result.masks is None or len(result.masks) == 0:
        return vis, 0

    masks_data = result.masks.data.cpu().numpy()
    h, w = image_bgr.shape[:2]
    n = len(masks_data)

    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append((int(rgb[2]), int(rgb[1]), int(rgb[0])))

    for i, mask in enumerate(masks_data):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = mask_resized > 0.5
        overlay[binary] = colors[i]

        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, colors[i], 2)

    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w / 2000, 1.8))
    thickness = max(1, int(font_scale * 2))

    for i, mask in enumerate(masks_data):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask_resized > 0.5)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(i + 1)
            (tw, th_), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.putText(
                vis, label, (cx - tw // 2, cy + th_ // 2),
                font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA,
            )
            cv2.putText(
                vis, label, (cx - tw // 2, cy + th_ // 2),
                font, font_scale, colors[i], thickness, cv2.LINE_AA,
            )

    return vis, n


def predict(weight_path: Path, image_dir: Path, device: str):
    run_output_dir = OUTPUT_DIR / image_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weight_path))

    test_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpeg"))
    test_images = [p for p in test_images if p.parent == image_dir]

    print(f"\n[推理] 使用权重: {weight_path}")
    print(f"[推理] 图像目录: {image_dir}")
    print(f"[推理] 共 {len(test_images)} 张图片\n")

    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=0.25,
            iou=0.5,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        image_bgr = cv2.imread(str(img_path))

        vis, count = visualize_results(image_bgr, result)

        out_path = run_output_dir / f"{img_path.stem}_seg.png"
        cv2.imwrite(str(out_path), vis)

        compare = np.hstack([
            cv2.resize(image_bgr, (w := 800, h := int(800 * image_bgr.shape[0] / image_bgr.shape[1]))),
            cv2.resize(vis, (w, h)),
        ])
        cv2.imwrite(str(run_output_dir / f"{img_path.stem}_compare.png"), compare)

        print(f"  {img_path.name}: 检测到 {count} 个螺丝 → {out_path.name}")

    print(f"\n[完成] 结果保存至: {run_output_dir}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOv8-seg 螺丝实例分割")
    parser.add_argument("--predict", action="store_true", help="仅推理（跳过训练）")
    parser.add_argument("--weights", type=str, default=None, help="指定权重路径；训练时作为初始化权重，预测时作为推理权重")
    parser.add_argument("--resume", action="store_true", help="真正从 runs/yolo_seg/weights/last.pt 续训")
    parser.add_argument(
        "--datasets",
        type=str,
        default=DEFAULT_DATASETS,
        help="合并训练：逗号分隔子目录名（位于 data/images/ 下），每个需含 annotation.zip",
    )
    parser.add_argument(
        "--ann_zip",
        type=str,
        default=str(DEFAULT_ANN_ZIP),
        help="单目录训练时用：annotation.zip（默认与 --img_dir 配套；合并训练时忽略）",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=str(DEFAULT_IMAGE_DIR),
        help="抽帧图像目录：合并训练结束后会对这些目录逐一推理；仅推理时就是该目录",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=None,
        help="已弃用；当前固定按时间块切分，验证比例由 --val_ratio 控制",
    )
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="验证集比例（默认 0.2）")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="开启确定性训练")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="关闭确定性训练")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--smoke", action="store_true", help="快速冒烟训练（小 epoch 验证链路）")
    parser.add_argument("--device", type=str, default=DEVICE, help="训练/推理设备，如 2、0 或 cpu")
    args = parser.parse_args()
    sanitize_ultralytics_fonts()

    run_weight_dir = SCRIPT_DIR / "runs" / "yolo_seg" / "weights"
    best_weight = run_weight_dir / "best.pt"
    last_weight = run_weight_dir / "last.pt"
    ann_zip = Path(args.ann_zip)
    image_dir = Path(args.img_dir)

    if args.predict:
        w = Path(args.weights) if args.weights else best_weight
        if not w.exists():
            print(f"[错误] 权重文件不存在: {w}")
            return
        predict(w, image_dir=image_dir, device=args.device)
    else:
        dataset_parts = [p.strip() for p in args.datasets.split(",") if p.strip()]
        if len(dataset_parts) == 1:
            only = dataset_parts[0]
            idir = Path(only) if Path(only).is_absolute() else DATA_ROOT / "images" / only
            cfg_path = prepare_dataset(
                ann_zip=idir / "annotation.zip",
                image_dir=idir,
                val_ratio=args.val_ratio,
            )
            infer_dirs = [idir]
        else:
            pairs: list[tuple[Path, Path]] = []
            for part in dataset_parts:
                idir = Path(part) if Path(part).is_absolute() else DATA_ROOT / "images" / part
                pairs.append((idir, idir / "annotation.zip"))
            cfg_path = prepare_merged_dataset(
                pairs,
                val_ratio=args.val_ratio,
            )
            infer_dirs = [p[0] for p in pairs]

        epochs = min(5, EPOCHS) if args.smoke else None
        if args.smoke:
            print(f"[冒烟模式] 本次训练 epochs={epochs}")
        if args.resume:
            if not last_weight.exists():
                print(f"[错误] 断点续训需要 last.pt，但未找到: {last_weight}")
                return
            init_weight = last_weight
        else:
            init_weight = Path(args.weights) if args.weights else Path(MODEL_NAME)
            if not init_weight.exists():
                print(f"[错误] 初始化权重不存在: {init_weight}")
                return
        w = train(
            cfg_path,
            init_weight=init_weight,
            device=args.device,
            epochs_override=epochs,
            seed=SPLIT_SEED,
            deterministic=args.deterministic,
            resume=args.resume,
        )
        for idir in infer_dirs:
            predict(w, image_dir=idir, device=args.device)


if __name__ == "__main__":
    main()
