import numpy as np
import cv2
import joblib
from ultralytics import YOLO

class PushupDetector:
    """识别模块"""
    
    def __init__(self, model_path, yolo_path='yolov8x-pose.pt'):
        self.yolo = YOLO(yolo_path)
        self.classifier = joblib.load(model_path)

    def extract_features(self, keypoints):
        """仅使用角度特征，返回 numpy 数组。
        特征顺序: [left_elbow, right_elbow, avg_elbow, left_shoulder, right_shoulder, torso_angle]
        """
        if keypoints is None:
            return np.zeros(6, dtype=float)

        if keypoints.shape[1] == 3:
            kpts = keypoints[:, :2]
        else:
            kpts = keypoints

        def angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos = np.dot(v1, v2) / denom
            return np.degrees(np.arccos(np.clip(cos, -1, 1)))

        # 防护：如果关键点不完整，填充为0
        def safe(idx):
            try:
                return kpts[idx]
            except Exception:
                return np.array([0.0, 0.0])

        left_elbow = angle(safe(5), safe(7), safe(9))
        right_elbow = angle(safe(6), safe(8), safe(10))
        avg_elbow = (left_elbow + right_elbow) / 2.0
        left_shoulder = angle(safe(11), safe(5), safe(7))
        right_shoulder = angle(safe(12), safe(6), safe(8))

        # 躯干角：肩中心 - 髋 - 膝中点
        shoulder_center = (safe(5) + safe(6)) / 2.0
        hip_center = (safe(11) + safe(12)) / 2.0
        knee_center = (safe(13) + safe(14)) / 2.0
        torso_angle = angle(shoulder_center, hip_center, knee_center)

        return np.array([left_elbow, right_elbow, avg_elbow, left_shoulder, right_shoulder, torso_angle])
    
    def detect(self, frame):
        results = self.yolo(frame, verbose=False)

        # 兼容不同 ultralytics 版本的返回格式
        try:
            kp_container = results[0].keypoints
        except Exception:
            kp_container = None

        if not kp_container:
            # 返回原图，state/confidence 为 None/0
            return None, 0.0, frame

        # 取第一个检测到的人体关键点（如果有多个，选择置信度最高或第一个）
        try:
            keypoints = kp_container.xy[0].cpu().numpy()
        except Exception:
            # 备用：尝试直接将 keypoints 转为 numpy
            try:
                keypoints = np.array(kp_container)
            except Exception:
                return None, 0.0, frame

        features = self.extract_features(keypoints)

        try:
            state = self.classifier.predict([features])[0]
            confidence = float(self.classifier.predict_proba([features])[0].max())
        except Exception:
            state = None
            confidence = 0.0

        annotated_frame = self._draw_skeleton(frame.copy(), keypoints, state, confidence)

        return state, confidence, annotated_frame
    
    def _draw_skeleton(self, frame, keypoints, state, confidence):
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
        
        color = (0,255,0) if confidence > 0.8 else (0,165,255)
        text = f"{state.upper()}: {confidence:.0%}"
        cv2.putText(frame, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        return frame