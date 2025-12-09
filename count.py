from modules import PushupDetector, PushupCounter

if __name__ == "__main__":
    detector = PushupDetector('models/pushup_model.pkl', 'models/yolo11x-pose.pt')
    counter = PushupCounter(detector)
    
    total = counter.process_video(
        video_path='videos/n1.mp4',
        output_path='videos/output.mp4'
    )
    
    print(f"检测到 {total} 个俯卧撑")