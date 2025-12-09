import cv2
from modules import PushupDetector

if __name__ == "__main__":
    detector = PushupDetector('models/pushup_model.pkl', 'models/yolo11x-pose.pt')
    
    cap = cv2.VideoCapture('videos/n1.mp4')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        state, confidence, annotated = detector.detect(frame)
        
        cv2.imshow('Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()