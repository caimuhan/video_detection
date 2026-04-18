"""
YOLO 单图推理脚本
用法示例：
python infer_image.py --image datasets/images/val/xxx.jpg --model runs/final/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="使用指定 YOLO 模型推理指定图片")
    parser.add_argument("--image", required=True, help="待推理图片路径")
    parser.add_argument("--model", required=True, help="YOLO 模型权重路径（.pt）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，默认 0.7")
    parser.add_argument("--device", default=None, help="推理设备，如 0 或 cpu；默认自动选择")
    parser.add_argument("--project", default="runs/predict", help="结果输出根目录")
    parser.add_argument("--name", default="exp", help="结果子目录名")
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存可视化结果图；不传则仅终端输出检测信息",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型：{model_path}")

    model = YOLO(str(model_path))

    predict_kwargs = {
        "source": str(image_path),
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    if args.save:
        predict_kwargs.update(
            {
                "save": True,
                "project": args.project,
                "name": args.name,
                "exist_ok": True,
            }
        )

    results = model.predict(**predict_kwargs)
    if not results:
        print("未返回任何推理结果")
        return

    result = results[0]
    boxes = result.boxes
    total = 0 if boxes is None else len(boxes)

    print("=" * 50)
    print(f"图片路径: {image_path.resolve()}")
    print(f"模型路径: {model_path.resolve()}")
    print(f"检测框数量: {total}")

    if boxes is not None and total > 0:
        names = result.names
        class_ids = boxes.cls.tolist()
        class_count = {}
        for cid in class_ids:
            cid_int = int(cid)
            class_name = names.get(cid_int, str(cid_int)) if isinstance(names, dict) else str(cid_int)
            class_count[class_name] = class_count.get(class_name, 0) + 1

        print("各类别数量:")
        for class_name, count in class_count.items():
            print(f"  {class_name}: {count}")

    if args.save:
        save_dir = Path(args.project) / args.name
        print(f"结果已保存到: {save_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
