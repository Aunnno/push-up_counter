import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class PushupTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            oob_score=True
        )
    
    def extract_features(self, keypoints):
        """仅使用角度特征。

        返回顺序与 PushupDetector 保持一致：
        [left_elbow, right_elbow, avg_elbow, left_shoulder, right_shoulder, torso_angle]
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
    
    def train(self, annotation_file):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        X = [self.extract_features(np.array(d['keypoints'])) for d in data]
        y = [d['label'] for d in data]
        X, y = np.array(X), np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        print(f"训练准确率: {self.model.score(X_train, y_train):.4f}")
        print(f"测试准确率: {self.model.score(X_test, y_test):.4f}")
        print(f"OOB准确率: {self.model.oob_score_:.4f}\\n")
        print(classification_report(y_test, self.model.predict(X_test)))
        
        return self.model.score(X_test, y_test)
    
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"模型已保存: {path}")