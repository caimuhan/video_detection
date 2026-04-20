import argparse
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from call_infer_image import process_image_dir
from read_vedio import extract_frames_from_video


TYPE_ORDER = ["Type_1", "Type_2", "Type_3", "Type_4", "Type_5"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab4 视频螺丝计数统一入口")
    parser.add_argument("--data_dir", required=True, help="测试视频文件夹路径")
    parser.add_argument("--output_path", required=True, help="输出 result.npy 路径")
    parser.add_argument("--output_time_path", required=True, help="输出 time.txt 路径")
    parser.add_argument("--mask_output_path", required=True, help="掩膜图输出目录")

    # 可选参数：便于本地调参，默认不影响作业要求的命令格式
    parser.add_argument("--model_path", default=None, help="YOLO 权重路径，默认自动查找")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", default=None, help="推理设备，如 0 或 cpu")
    parser.add_argument("--ransac_thresh", type=float, default=30.0, help="RANSAC 阈值")
    parser.add_argument("--dedup_dist", type=float, default=160.0, help="去重距离阈值")
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace, code_dir: Path) -> Path:
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        return model_path

    candidates = [
        code_dir / "weights" / "best.pt",
        code_dir.parent / "runs" / "detect" / "runs" / "final" / "weights" / "best.pt",
        code_dir.parent / "runs" / "final" / "weights" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError("未找到模型权重，请通过 --model_path 指定")


def collect_videos(data_dir: Path) -> List[Path]:
    videos = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(videos)


def draw_mask_overlay(image_path: Path, detections: List[Dict[str, object]], save_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片用于掩膜绘制: {image_path}")

    overlay = image.copy()
    for det in detections:
        center = det.get("center_xy", [0, 0])
        class_name = str(det.get("class_name", "unknown"))
        cx = int(round(float(center[0])))
        cy = int(round(float(center[1])))

        cv2.circle(overlay, (cx, cy), 18, (0, 255, 0), thickness=-1)
        cv2.putText(
            overlay,
            class_name,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    blended = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), blended)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    code_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir 不存在或不是目录: {data_dir}")

    model_path = resolve_model_path(args, code_dir)
    videos = collect_videos(data_dir)
    if not videos:
        raise RuntimeError(f"未在目录中找到视频文件: {data_dir}")

    output_path = Path(args.output_path)
    output_time_path = Path(args.output_time_path)
    mask_output_dir = Path(args.mask_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_time_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    temp_frames_root = code_dir / ".tmp_frames"
    temp_frames_root.mkdir(parents=True, exist_ok=True)

    result_dict: Dict[str, List[int]] = {}

    for video_path in videos:
        video_name = video_path.stem
        extract_frames_from_video(str(video_path), str(temp_frames_root))

        frame_dir = temp_frames_root / video_name
        if not frame_dir.exists():
            raise RuntimeError(f"视频抽帧目录不存在: {frame_dir}")

        process_result = process_image_dir(
            image_dir=str(frame_dir),
            model=str(model_path),
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=False,
            ransac_thresh=args.ransac_thresh,
            dedup_dist=args.dedup_dist,
        )

        class_count = process_result.get("screw_count_by_class", {})
        result_dict[video_name] = [int(class_count.get(t, 0)) for t in TYPE_ORDER]
        print(f"视频 {video_name} 处理完成，结果: {class_count}")
        image_names = process_result.get("image_names", [])
        detections_by_image = process_result.get("detections_by_image", {})
        if image_names:
            pick_name = image_names[len(image_names) // 2]
            pick_image_path = frame_dir / pick_name
            pick_dets = detections_by_image.get(pick_name, [])
            mask_path = mask_output_dir / f"{video_name}_mask.png"
            draw_mask_overlay(pick_image_path, pick_dets, mask_path)

    np.save(str(output_path), result_dict)
    print(result_dict)
    print(f"抽帧结果已保留在: {temp_frames_root.resolve()}")
    elapsed = time.time() - start_time
    output_time_path.write_text(f"{elapsed:.6f}", encoding="utf-8")


if __name__ == "__main__":
    main()
