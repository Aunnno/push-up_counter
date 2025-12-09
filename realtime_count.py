import argparse
import cv2
import os
from modules import PushupDetector, PushupCounter


def main():
    parser = argparse.ArgumentParser(description='Realtime push-up counter (camera or video)')
    parser.add_argument('--video', '-v', default=None, help='Path to input video. If omitted, use webcam or selected video mode')
    parser.add_argument('--output', '-o', default=None, help='Path to save annotated output video (optional)')
    parser.add_argument('--model', '-m', default='models/pushup_model.pkl', help='Path to classifier model')
    parser.add_argument('--yolo', '-y', default='models/yolo11x-pose.pt', help='Path to YOLO pose model')
    parser.add_argument('--camera-max-index', type=int, default=4, help='Max camera index to try when no --video provided')
    parser.add_argument('--mode', choices=['camera', 'video', '1', '2'], default=None,
                        help="Choose mode: 'camera' (1) or 'video' (2). If omitted, you'll be prompted.")
    args = parser.parse_args()

    # 选择运行模式：摄像头 或 本地视频
    mode = args.mode
    if mode is None:
        # 交互式选择
        try:
            choice = input("请选择模式: 1) 摄像头  2) 本地视频 [默认1]: ").strip()
        except Exception:
            choice = ''
        if choice in ('2', 'video', 'v'):
            mode = 'video'
        else:
            mode = 'camera'

    # 处理 video 模式
    if mode in ('2', 'video'):
        if args.video:
            source = args.video
        else:
            # 询问用户输入路径或使用回退文件
            try:
                vidpath = input("请输入视频路径（回车使用 videos/n1.mp4）: ").strip()
            except Exception:
                vidpath = ''
            if vidpath:
                source = vidpath
            else:
                source = 'videos/n1.mp4'

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频文件: {source}")
            return

    else:
        # camera 模式：尝试打开本地摄像头索引 0..camera_max_index
        cap = None
        source = None
        for i in range(0, args.camera_max_index + 1):
            try_cap = cv2.VideoCapture(i)
            if try_cap.isOpened():
                cap = try_cap
                source = i
                print(f"已打开摄像头索引: {i}")
                break
            try_cap.release()

        if cap is None:
            # 回退到示例视频（如果存在）
            fallback = 'videos/n1.mp4'
            if os.path.exists(fallback):
                print(f"未找到摄像头，回退使用本地视频: {fallback}")
                source = fallback
                cap = cv2.VideoCapture(source)
            else:
                print("未找到可用摄像头，且未提供回退视频，退出。")
                return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # check for NaN
        fps = 30

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        # 常见摄像头未提供分辨率，使用默认值
        w, h = 640, 480

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, max(1, int(fps)), (w, h))

    detector = PushupDetector(args.model, args.yolo)
    counter = PushupCounter(detector)

    print("按 'q' 退出，按 's' 保存当前帧（仅保存到输出视频，如果提供）")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count, annotated = counter.update(frame)

        # 显示窗口
        cv2.imshow('Pushup Counter - Live', annotated)

        if writer is not None:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and writer is not None:
            # 保存当前帧也已经由 writer 持续写入；这里另外打印提示
            print(f"已保存一帧到输出文件: {args.output}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n总计: {counter.count} 个俯卧撑")


if __name__ == '__main__':
    main()
