"""批量检测并通过匹配+RANSAC估计单应矩阵，维护全局螺丝列表。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from infer_image import infer_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量检测并通过匹配+RANSAC估计单应矩阵，维护全局螺丝列表")
    parser.add_argument("--image-dir", required=True, help="包含多个图片的目录路径")
    parser.add_argument("--model", required=True, help="YOLO 模型权重路径（.pt）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", default=None, help="推理设备，如 0 或 cpu")
    parser.add_argument("--save", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--project", default="runs/predict", help="结果输出根目录")
    parser.add_argument("--name", default="exp_call", help="结果子目录名")
    parser.add_argument("--ransac-thresh", type=float, default=30.0, help="RANSAC重投影阈值，默认30像素")
    parser.add_argument("--dedup-dist", type=float, default=160.0, help="全局去重阈值，默认160像素")
    parser.add_argument("--out-json", default=None, help="可选：将结果写入 JSON 文件")
    return parser.parse_args()


def collect_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(image_paths)


def to_np_point(det: Dict[str, Any]) -> np.ndarray:
    center = det["center_xy"]
    return np.array([float(center[0]), float(center[1])], dtype=np.float64)


def build_hungarian_matches(
    prev_dets: List[Dict[str, Any]],
    curr_dets: List[Dict[str, Any]],
) -> List[Tuple[int, int, float]]:
    matches: List[Tuple[int, int, float]] = []
    classes = sorted({str(d["class_name"]) for d in prev_dets} & {str(d["class_name"]) for d in curr_dets})

    for class_name in classes:
        prev_idx = [i for i, d in enumerate(prev_dets) if str(d["class_name"]) == class_name]
        curr_idx = [j for j, d in enumerate(curr_dets) if str(d["class_name"]) == class_name]
        if not prev_idx or not curr_idx:
            continue

        cost = np.zeros((len(prev_idx), len(curr_idx)), dtype=np.float64)
        for r, i in enumerate(prev_idx):
            p = to_np_point(prev_dets[i])
            for c, j in enumerate(curr_idx):
                q = to_np_point(curr_dets[j])
                cost[r, c] = float(np.linalg.norm(p - q))

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            matches.append((prev_idx[r], curr_idx[c], float(cost[r, c])))

    return matches


def apply_homography(point_xy: np.ndarray, h: np.ndarray) -> np.ndarray:
    p = np.array([point_xy[0], point_xy[1], 1.0], dtype=np.float64)
    p2 = h @ p
    if abs(p2[2]) < 1e-9:
        return np.array([point_xy[0], point_xy[1]], dtype=np.float64)
    return np.array([p2[0] / p2[2], p2[1] / p2[2]], dtype=np.float64)


def add_to_global_list(
    global_screws: List[Dict[str, Any]],
    mapped_xy: np.ndarray,
    det: Dict[str, Any],
    first_seen_image: str,
    dedup_dist: float,
) -> bool:
    for item in global_screws:
        existing = np.array(item["ref_center_xy"], dtype=np.float64)
        if float(np.linalg.norm(existing - mapped_xy)) < dedup_dist:
            return False

    global_screws.append(
        {
            "ref_center_xy": [round(float(mapped_xy[0]), 2), round(float(mapped_xy[1]), 2)],
            "class_id": int(det["class_id"]),
            "class_name": str(det["class_name"]),
            "first_seen_image": first_seen_image,
        }
    )
    return True


def process_image_dir(
    image_dir: str,
    model: str,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Any = None,
    save: bool = False,
    project: str = "runs/predict",
    name: str = "exp_call",
    ransac_thresh: float = 30.0,
    dedup_dist: float = 170.0,
) -> Dict[str, Any]:
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists() or not image_dir_path.is_dir():
        raise FileNotFoundError(f"图片目录不存在或不是目录: {image_dir_path}")

    image_paths = collect_images(image_dir_path)
    if not image_paths:
        raise RuntimeError(f"目录中没有可处理图片: {image_dir_path}")

    detections_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for image_path in image_paths:
        detections = infer_image(
            image=str(image_path),
            model=model,
            conf=conf,
            iou=iou,
            device=device,
            save=save,
            project=project,
            name=name,
        )
        detections_by_image[image_path.name] = detections

    first_image_name = image_paths[0].name
    global_screws: List[Dict[str, Any]] = []
    pairwise_info: List[Dict[str, Any]] = []

    # 每帧到第一帧坐标系的累计单应矩阵，第一帧为单位阵
    cumulative_h_to_first: Dict[str, np.ndarray] = {first_image_name: np.eye(3, dtype=np.float64)}

    for det in detections_by_image[first_image_name]:
        add_to_global_list(
            global_screws=global_screws,
            mapped_xy=to_np_point(det),
            det=det,
            first_seen_image=first_image_name,
            dedup_dist=dedup_dist,
        )

    for idx in range(1, len(image_paths)):
        prev_name = image_paths[idx - 1].name
        curr_name = image_paths[idx].name
        prev_dets = detections_by_image[prev_name]
        curr_dets = detections_by_image[curr_name]

        pair_record: Dict[str, Any] = {
            "pair": [prev_name, curr_name],
            "matched_pairs": 0,
            "inliers": 0,
            "H_curr_to_prev": None,
            "status": "ok",
            "reason": "",
        }

        matches = build_hungarian_matches(prev_dets, curr_dets)
        pair_record["matched_pairs"] = len(matches)

        if len(matches) < 4:
            pair_record["status"] = "skipped"
            pair_record["reason"] = "匹配对不足4，无法估计单应矩阵"
            pairwise_info.append(pair_record)
            continue

        prev_pts = np.array([to_np_point(prev_dets[i]) for i, _, _ in matches], dtype=np.float64).reshape(-1, 1, 2)
        curr_pts = np.array([to_np_point(curr_dets[j]) for _, j, _ in matches], dtype=np.float64).reshape(-1, 1, 2)

        h_curr_to_prev, mask = cv2.findHomography(curr_pts, prev_pts, cv2.RANSAC, ransac_thresh)
        if h_curr_to_prev is None or mask is None:
            pair_record["status"] = "skipped"
            pair_record["reason"] = "RANSAC失败，未得到有效单应矩阵"
            pairwise_info.append(pair_record)
            continue

        inliers = int(mask.ravel().sum())
        pair_record["inliers"] = inliers
        pair_record["H_curr_to_prev"] = [[round(float(v), 6) for v in row] for row in h_curr_to_prev.tolist()]

        if prev_name not in cumulative_h_to_first:
            pair_record["status"] = "skipped"
            pair_record["reason"] = "上一帧缺少到首帧的累计单应矩阵，当前帧跳过"
            pairwise_info.append(pair_record)
            continue

        h_curr_to_first = cumulative_h_to_first[prev_name] @ h_curr_to_prev
        cumulative_h_to_first[curr_name] = h_curr_to_first

        new_added = 0
        for det in curr_dets:
            mapped = apply_homography(to_np_point(det), h_curr_to_first)
            if add_to_global_list(
                global_screws=global_screws,
                mapped_xy=mapped,
                det=det,
                first_seen_image=curr_name,
                dedup_dist=dedup_dist,
            ):
                new_added += 1

        pair_record["new_added_to_global"] = new_added
        pairwise_info.append(pair_record)

    screw_count_by_class: Dict[str, int] = {}
    for item in global_screws:
        class_name = str(item["class_name"])
        screw_count_by_class[class_name] = screw_count_by_class.get(class_name, 0) + 1

    return {
        "image_count": len(image_paths),
        "image_names": [p.name for p in image_paths],
        "total_unique_screws": len(global_screws),
        "screw_count_by_class": screw_count_by_class,
        "global_screws": global_screws,
        "pairwise_homography": pairwise_info,
        "detections_by_image": detections_by_image,
    }


def main() -> None:
    args = parse_args()
    result = process_image_dir(
        image_dir=args.image_dir,
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        project=args.project,
        name=args.name,
        ransac_thresh=args.ransac_thresh,
        dedup_dist=args.dedup_dist,
    )

    print(f"已处理图片数量: {result['image_count']}")
    
    print("各螺丝类别数量:")
    print(json.dumps(result["screw_count_by_class"], ensure_ascii=False, indent=2))

    # print("总列表（基准坐标+类别+首次出现帧）:")
    # print(json.dumps(result["global_screws"], ensure_ascii=False, indent=2))

    # print("每对相邻帧的单应矩阵与内点统计:")
    # print(json.dumps(result["pairwise_homography"], ensure_ascii=False, indent=2))
    print(f"总螺丝数量（去重后）: {result['total_unique_screws']}")
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存到: {out_path.resolve()}")


if __name__ == "__main__":
    main()
