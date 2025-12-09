from modules import VideoAnnotationTool

if __name__ == "__main__":
    annotation = VideoAnnotationTool("videos/n1.mp4","models/yolo11x-pose.pt")
    annotation.run()