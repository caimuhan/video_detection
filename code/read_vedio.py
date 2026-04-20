import cv2
import os
import argparse
import sys
from pathlib import Path

def extract_frames_from_video(video_path, output_root):
    """
    对单个视频文件提取帧。
    
    Args:
        video_path (str): 视频文件路径。
        output_root (str): 总输出根目录。
    """
    # 根据视频文件名创建专属输出子目录
    video_name = Path(video_path).stem  # 不含扩展名
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频文件 '{video_path}'，已跳过。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"警告：视频 '{video_path}' 帧率无效，使用默认 30 fps。")
        fps = 30.0

    frame_interval = int(round(fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n处理视频：{video_path}")
    print(f"  帧率: {fps:.2f} fps, 总帧数: {total_frames}, 保存间隔: {frame_interval} 帧")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == 0 or frame_idx % frame_interval == 0:
            filename = f"frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            # 可取消下行注释以显示详细保存信息
            # print(f"    已保存: {filepath}")

        frame_idx += 1

    cap.release()
    print(f"  完成，共保存 {saved_count} 张图片到 '{output_dir}'")

def process_folder(input_dir, output_root="frames", extensions=(".mov",)):
    """
    处理文件夹内所有指定扩展名的视频文件。

    Args:
        input_dir (str): 输入文件夹路径。
        output_root (str): 输出根目录。
        extensions (tuple): 要处理的视频扩展名元组。
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"错误：'{input_dir}' 不是一个有效文件夹。")
        sys.exit(1)

    # 收集所有匹配扩展名的视频文件
    video_files = []
    for ext in extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))

    # 去重（避免同一文件因大小写重复）
    video_files = list(set(video_files))

    if not video_files:
        print(f"警告：在文件夹 '{input_dir}' 中未找到任何 .mov 文件。")
        return

    print(f"找到 {len(video_files)} 个视频文件，开始处理...")
    for video_file in sorted(video_files):
        extract_frames_from_video(str(video_file), output_root)

    print(f"\n全部处理完成！所有截图保存在目录：'{output_root}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量处理文件夹内所有 .mov 视频：第一帧必存，此后每隔1秒截取一帧。"
    )
    parser.add_argument("input_dir", help="包含 .mov 视频的文件夹路径")
    parser.add_argument("-o", "--output", default="frames", help="输出图片的总目录（默认：frames）")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".mov"], 
                        help="要处理的视频扩展名（默认：.mov），可指定多个，如 .mov .mp4")
    args = parser.parse_args()

    # 将扩展名统一为小写开头（如用户输入 .MOV 也转为 .mov）
    exts = tuple(ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in args.extensions)
    process_folder(args.input_dir, args.output, exts)