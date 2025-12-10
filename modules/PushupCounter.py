import cv2

class PushupCounter:
    """计数模块"""
    
    def __init__(self, detector, confidence=0.6):
        self.detector = detector
        self.confidence = confidence
        self.reset()
    
    def reset(self):
        self.count = 0
        self.last_state = None
        self.in_bottom = False
    
    def update(self, frame):
        state, confidence, annotated_frame = self.detector.detect(frame)
        
        if state and confidence > self.confidence:
            if state == 'bottom' and not self.in_bottom:
                self.in_bottom = True
            elif state == 'ready' and self.in_bottom:
                self.count += 1
                self.in_bottom = False
            
            self.last_state = state
        
        cv2.putText(annotated_frame, f"Count: {self.count}", (10,100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)
        
        return self.count, annotated_frame
    
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        self.reset()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            count, annotated = self.update(frame)
            
            cv2.imshow('Pushup Counter', annotated)
            if output_path:
                out.write(annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n总计: {self.count} 个俯卧撑")
        return self.count