"""
YOLOv8 螺丝检测训练脚本
用法：python train.py
"""

import time
from pathlib import Path
from ultralytics import YOLO

# ─── 配置区（根据实际情况修改）─────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
DATA_YAML   = ROOT / "data.yaml"
CLASSES_TXT = DATASET_DIR / "labels" / "train" / "classes.txt"
MODEL       = "yolov8x.pt"                    # 预训练权重：yolov8n/s/m/l/x.pt
PROJECT     = "runs"                          # 输出根目录
NAME        = "final"                      # 本次实验名称
EPOCHS      = 200
IMGSZ       = 640
BATCH       = 4
LR0         = 0.01                            # 初始学习率
DEVICE      = 0                               # GPU id，纯 CPU 填 'cpu'
# ──────────────────────────────────────────────────────────────


def load_class_names(classes_file: Path):
    """从 classes.txt 读取类别名"""
    if not classes_file.exists():
        raise FileNotFoundError(f"找不到类别文件：{classes_file}")

    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"类别文件为空：{classes_file}")
    return names


def generate_data_yaml():
    """根据当前项目目录自动生成 YOLO 所需 data.yaml"""
    train_images = DATASET_DIR / "images" / "train"
    val_images = DATASET_DIR / "images" / "val"
    train_labels = DATASET_DIR / "labels" / "train"
    val_labels = DATASET_DIR / "labels" / "val"

    for p in [train_images, val_images, train_labels, val_labels]:
        if not p.exists():
            raise FileNotFoundError(f"找不到目录：{p}")

    names = load_class_names(CLASSES_TXT)
    names_yaml = "\n".join([f"  {idx}: {name}" for idx, name in enumerate(names)])

    yaml_text = (
        f"path: {DATASET_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_yaml}\n"
    )
    DATA_YAML.write_text(yaml_text, encoding="utf-8")
    return names


def check_env():
    """训练前环境检查"""
    import torch
    print("=" * 50)
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA 可用 : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"显存     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif DEVICE != "cpu":
        print("[提示] 未检测到 CUDA，将自动改为 CPU 训练")

    yaml_path = Path(DATA_YAML)
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到 data.yaml：{yaml_path.resolve()}")
    print(f"data.yaml: {yaml_path.resolve()}")
    print("=" * 50)


def train():
    names = generate_data_yaml()
    print(f"已生成 data.yaml：{DATA_YAML}")
    print(f"类别数：{len(names)}，类别：{', '.join(names)}")

    check_env()

    model = YOLO(MODEL)
    print(f"\n已加载预训练权重：{MODEL}")
    print(f"开始训练，共 {EPOCHS} 轮...\n")

    start = time.time()

    results = model.train(
        data      = str(DATA_YAML),
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        lr0       = LR0,
        device    = DEVICE if __import__("torch").cuda.is_available() else "cpu",
        augment   = True,       # Mosaic / 翻转 / 色调抖动
        project   = PROJECT,
        name      = NAME,
        exist_ok  = False,      # 避免覆盖已有实验，改 True 则允许覆盖
        patience  = 20,         # 20 轮 mAP 无提升则提前停止
        save      = True,
        plots     = True,       # 生成 results.png / confusion_matrix.png
        verbose   = True,
    )

    elapsed = time.time() - start
    print(f"\n训练完成，耗时 {elapsed/60:.1f} 分钟")

    # ── 打印关键指标 ──
    weight_dir = Path(PROJECT) / NAME / "weights"
    best_pt    = weight_dir / "best.pt"
    last_pt    = weight_dir / "last.pt"

    print("\n" + "=" * 50)
    print("训练结果")
    print("=" * 50)
    if best_pt.exists():
        print(f"最优权重 : {best_pt.resolve()}")
    if last_pt.exists():
        print(f"最终权重 : {last_pt.resolve()}")

    metrics = results.results_dict
    map50   = metrics.get("metrics/mAP50(B)",    0)
    map5095 = metrics.get("metrics/mAP50-95(B)", 0)
    print(f"mAP@0.5       : {map50:.4f}")
    print(f"mAP@0.5:0.95  : {map5095:.4f}")

    if map50 >= 0.85:
        print("\n[OK] 精度达标，可以直接写 run.py")
    elif map50 >= 0.70:
        print("\n[提示] 精度一般，建议查看 confusion_matrix.png，")
        print("       对混淆严重的类别补标 30~50 框后重训")
    else:
        print("\n[警告] 精度偏低，请检查：")
        print("  1. 标注框是否正确（类别有没有标错）")
        print("  2. data.yaml 里的路径是否正确")
        print("  3. 各类别标注数量是否过少（建议每类 ≥ 50 框）")

    print("=" * 50)

    # ── 快速验证：在 val 集上跑一遍 ──
    print("\n在验证集上评估最优权重...")
    best_model = YOLO(str(best_pt))
    val_results = best_model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        device=DEVICE if __import__("torch").cuda.is_available() else "cpu",
    )
    print("验证完成，详细报告见上方输出")

    return best_pt


if __name__ == "__main__":
    best_pt = train()
    print(f"\n下一步：将 {best_pt} 复制到 code/weights/best.pt，然后编写 run.py")