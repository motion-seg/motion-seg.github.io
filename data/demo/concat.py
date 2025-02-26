import cv2
import numpy as np
import os

def concatenate_videos_horizontally(video_path1, video_path2, output_path):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    video1 = cv2.VideoCapture(video_path1)
    video2 = cv2.VideoCapture(video_path2)

    if not video1.isOpened():
        print(f"Error: Unable to open video {video_path1}")
        return
    if not video2.isOpened():
        print(f"Error: Unable to open video {video_path2}")
        return

    # 获取视频参数
    fps = video1.get(cv2.CAP_PROP_FPS)
    width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_height = min(height1, height2)
    # 计算调整后的宽度（保持宽高比）
    if height1 != 0:
        width1_adj = int(width1 * (new_height / height1))
    else:
        width1_adj = width1
    if height2 != 0:
        width2_adj = int(width2 * (new_height / height2))
    else:
        width2_adj = width2

    total_width = width1_adj + width2_adj

    # 使用H.264编解码器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, new_height))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            break

        # 调整帧尺寸到新的宽度和高度
        frame1_resized = cv2.resize(frame1, (width1_adj, new_height))
        frame2_resized = cv2.resize(frame2, (width2_adj, new_height))

        # 水平拼接
        concatenated_frame = np.hstack((frame1_resized, frame2_resized))

        out.write(concatenated_frame)

    video1.release()
    video2.release()
    out.release()
    print(f"{output_path}")

# 使用示例
video_path1 = "static/videos/davis-mask/horsejump-high/video/original_rgb.mp4"
video_path2 = "static/videos/davis-mask/horsejump-high/video/mask_rgb_color.mp4"
output_path = "/Users/huangnan/4_3d_track/page/motion-seg.github.io/data/demo/horsejump-high_concat.mp4"

concatenate_videos_horizontally(video_path1, video_path2, output_path)