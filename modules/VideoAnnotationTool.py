from ultralytics import YOLO
import json
import cv2
import os

class VideoAnnotationTool:
    def __init__(self, video_path, yolo_model_path, output_dir='data', annotation_filename='annotations.json'):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.yolo_model = YOLO(yolo_model_path)
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置标注文件完整路径
        self.annotation_file = os.path.join(self.output_dir, annotation_filename)
        
        self.annotations = []
        self.paused = False
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stats = {'ready': 0, 'bottom': 0}
        
        self.load_existing_annotations()
        
    def load_existing_annotations(self):
        self.existing_count = 0
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    self.existing_count = len(existing)
                    print(f"✓ 文件中已有 {self.existing_count} 条标注")
            except:
                print("⚠ 加载标注失败，将创建新文件")
        else:
            print(f"ℹ 将创建新文件: {self.annotation_file}")
    
    def draw_skeleton(self, frame, keypoints):
        skeleton = [
            [5,7],[7,9],[6,8],[8,10],[5,6],
            [5,11],[6,12],[11,12],[11,13],
            [13,15],[12,14],[14,16]
        ]
        kpts = keypoints[:, :2].astype(int)
        for conn in skeleton:
            cv2.line(frame, tuple(kpts[conn[0]]), tuple(kpts[conn[1]]), (0,255,0), 2)
        for pt in kpts:
            cv2.circle(frame, tuple(pt), 4, (0,0,255), -1)
        return frame
    
    def run(self):
        print("\n" + "="*60)
        print("视频标注工具")
        print("操作: 1=Ready, 2=Bottom, 空格=暂停, S=保存, Q=退出")
        print("="*60 + "\n")
        
        try:
            while self.cap.isOpened():
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("视频播放完毕")
                        self.save_annotations()
                        break
                    
                    self.current_frame += 1
                    results = self.yolo_model(frame, verbose=False)
                    display_frame = frame.copy()
                    
                    if len(results[0].keypoints) > 0:
                        keypoints = results[0].keypoints.xy[0].cpu().numpy()
                        display_frame = self.draw_skeleton(display_frame, keypoints)
                        self.current_keypoints = keypoints
                    else:
                        self.current_keypoints = None
                    
                    info = [
                        f"Video: {self.video_name} | Frame: {self.current_frame}/{self.total_frames}",
                        f"Ready: {self.stats['ready']} | Bottom: {self.stats['bottom']}"
                    ]
                    
                    for i, text in enumerate(info):
                        cv2.putText(display_frame, text, (10, 30+i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    cv2.imshow('Annotation Tool', display_frame)
                else:
                    cv2.imshow('Annotation Tool', display_frame)
                
                key = cv2.waitKey(30 if not self.paused else 0) & 0xFF
                
                if key == ord('q'):
                    print("\n退出")
                    self.save_annotations()
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                elif key == ord('1'):
                    self.add_annotation('ready')
                elif key == ord('2'):
                    self.add_annotation('bottom')
                elif key == ord('s'):
                    self.save_annotations()
            
            self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"ERROR: {e}")
            self.save_annotations()
            self.cap.release()
            cv2.destroyAllWindows()
    
    def add_annotation(self, label):
        if hasattr(self, 'current_keypoints') and self.current_keypoints is not None:
            self.annotations.append({
                'video': self.video_name,
                'keypoints': self.current_keypoints.tolist(),
                'label': label
            })
            self.stats[label] += 1
            print(f"{self.video_name} - 已标注 [{label}] | 总计: {self.stats}")
        else:
            print("未检测到姿态")
    
    def save_annotations(self):
        existing = []
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                pass
        
        merged = existing + self.annotations
        
        with open(self.annotation_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 已保存 {len(merged)} 条标注 (新增 {len(self.annotations)} 条)")

