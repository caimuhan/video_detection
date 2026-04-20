import os
import cv2
import argparse
from pathlib import Path
import numpy as np


def create_stitcher(mode_name: str = "scans"):
    """
    创建 OpenCV Stitcher
    mode:
        - scans: 更适合平移扫描、文档/墙面/地面等连续视频拼接
        - panorama: 更适合原地转动拍全景
    """
    mode_name = mode_name.lower()
    if mode_name == "panorama":
        mode = cv2.Stitcher_PANORAMA
    else:
        mode = cv2.Stitcher_SCANS

    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(mode)
    else:
        # 兼容老版本 OpenCV
        stitcher = cv2.createStitcher(False if mode_name == "scans" else True)

    return stitcher


def resize_keep_ratio(img, target_long_side=1200):
    # target_long_side <= 0 表示不缩放，保留原始分辨率
    if target_long_side is None or target_long_side <= 0:
        return img

    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return img

    scale = target_long_side / float(long_side)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sharpness_score(img):
    """
    用拉普拉斯方差衡量清晰度，越大越清晰
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_difference(img1, img2, size=(320, 180)):
    """
    衡量两帧差异，避免抽取太多几乎一样的相邻帧
    """
    a = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = np.mean(cv2.absdiff(a, b))
    return diff


def extract_frames(
    video_path,
    step=15,
    max_frames=80,
    target_long_side=1200,
    blur_thresh=80.0,
    diff_thresh=6.0,
):
    """
    从视频中抽帧：
    - 每 step 帧取一次
    - 跳过过模糊帧
    - 跳过和上一个保留帧过于相似的帧
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] 视频路径: {video_path}")
    print(f"[INFO] 总帧数: {total_frames}, FPS: {fps:.2f}")
    print(f"[INFO] 每 {step} 帧抽取一次，最多保留 {max_frames} 帧")

    frames = []
    last_kept = None
    frame_idx = 0
    kept_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame_small = resize_keep_ratio(frame, target_long_side)

            score = sharpness_score(frame_small)
            if score < blur_thresh:
                frame_idx += 1
                continue

            if last_kept is not None:
                diff = frame_difference(frame_small, last_kept)
                if diff < diff_thresh:
                    frame_idx += 1
                    continue

            frames.append(frame_small)
            last_kept = frame_small
            kept_count += 1

            print(
                f"[INFO] 保留第 {frame_idx} 帧 | sharpness={score:.2f} | 当前保留 {kept_count} 帧"
            )

            if kept_count >= max_frames:
                print("[INFO] 已达到最大保留帧数，停止抽帧")
                break

        frame_idx += 1

    cap.release()

    if len(frames) < 2:
        raise RuntimeError("可用于拼接的帧太少，至少需要 2 帧。")

    print(f"[INFO] 最终用于拼接的帧数: {len(frames)}")
    return frames


def stitch_images(images, mode="scans"):
    """
    使用 OpenCV Stitcher 进行拼接
    """
    stitcher = create_stitcher(mode)

    status, pano = stitcher.stitch(images)

    # OpenCV 通常 0 代表成功
    if status != cv2.Stitcher_OK:
        raise RuntimeError(
            f"拼接失败，状态码: {status}\n"
            f"常见原因：\n"
            f"1. 相邻帧重叠太少\n"
            f"2. 相邻帧太像/太多，导致误匹配\n"
            f"3. 视频视差太大，不适合直接单应拼接\n"
            f"4. 模式不对，可尝试 scans / panorama 切换"
        )

    return pano


def unsharp_mask(img, sigma=1.2, amount=1.0):
    """
    轻量反锐化蒙版，增强细节但尽量避免过度噪声。
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def main():
    script_dir = Path(__file__).resolve().parent
    default_video = script_dir / "data" / "IMG_2375.MOV"
    default_output = script_dir / "results" / "scans.jpg"

    parser = argparse.ArgumentParser(description="对连续视频做 OpenCV 拼接，输出大图")
    parser.add_argument(
        "--video",
        type=str,
        default=str(default_video),
        help="输入视频路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="输出拼接图路径",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="每隔多少帧抽一帧",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=50,
        help="最多用于拼接的帧数",
    )
    parser.add_argument(
        "--resize_long",
        type=int,
        default=0,
        help="抽帧后缩放到的最长边；<=0 表示保留原分辨率",
    )
    parser.add_argument(
        "--blur_thresh",
        type=float,
        default=80.0,
        help="清晰度阈值，越大越严格",
    )
    parser.add_argument(
        "--diff_thresh",
        type=float,
        default=10.0,
        help="与上一保留帧的差异阈值，越大保留帧越少",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="scans",
        choices=["scans", "panorama"],
        help="拼接模式：scans 或 panorama",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="对拼接结果做轻量锐化，提升细节清晰度",
    )

    args = parser.parse_args()

    try:
        frames = extract_frames(
            video_path=args.video,
            step=args.step,
            max_frames=args.max_frames,
            target_long_side=args.resize_long,
            blur_thresh=args.blur_thresh,
            diff_thresh=args.diff_thresh,
        )

        pano = stitch_images(frames, mode=args.mode)

        if args.sharpen:
            pano = unsharp_mask(pano, sigma=1.2, amount=1.0)

        ok = cv2.imwrite(args.output, pano)
        if not ok:
            raise RuntimeError(f"拼接成功，但保存失败: {args.output}")

        print(f"[INFO] 拼接完成，已保存到: {args.output}")
        print(f"[INFO] 输出尺寸: {pano.shape[1]} x {pano.shape[0]}")

    except Exception as e:
        print("[ERROR]", e)


if __name__ == "__main__":
    main()