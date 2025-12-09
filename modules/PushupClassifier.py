import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class PushupClassifier:
    def extract_features(self, keypoints):
        """仅使用角度特征，返回与 Trainer/Detector 一致的向量。"""
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

        shoulder_center = (safe(5) + safe(6)) / 2.0
        hip_center = (safe(11) + safe(12)) / 2.0
        knee_center = (safe(13) + safe(14)) / 2.0
        torso_angle = angle(shoulder_center, hip_center, knee_center)

        return np.array([left_elbow, right_elbow, avg_elbow, left_shoulder, right_shoulder, torso_angle])

    def predict(self, keypoints):
        features = self.extract_features(keypoints).reshape(1, -1)
        state = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = float(max(proba))
        return state, confidence
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        print(f"模型已加载: {filepath}")

def real_time_inference(video_path, model_path='pushup_model.pkl', yolo_path='yolov8n-pose.pt'):
    """模块层级的实时推理小函数（独立于类）。"""
    import cv2
    from ultralytics import YOLO

    print("加载模型...")
    classifier = PushupClassifier()
    classifier.load_model(model_path)
    yolo = YOLO(yolo_path)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)

        try:
            kp_container = results[0].keypoints
        except Exception:
            kp_container = None

        if kp_container:
            try:
                keypoints = kp_container.xy[0].cpu().numpy()
            except Exception:
                try:
                    keypoints = np.array(kp_container)
                except Exception:
                    keypoints = None

            if keypoints is not None:
                state, confidence = classifier.predict(keypoints)

                skeleton = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],
                        [11,12],[11,13],[13,15],[12,14],[14,16]]
                kpts = keypoints[:, :2].astype(int)
                for conn in skeleton:
                    cv2.line(frame, tuple(kpts[conn[0]]), tuple(kpts[conn[1]]), (0,255,0), 2)

                color = (0,255,0) if confidence > 0.8 else (0,165,255)
                text = f"{state.upper()}: {confidence:.0%}"
                cv2.putText(frame, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier = PushupClassifier()
