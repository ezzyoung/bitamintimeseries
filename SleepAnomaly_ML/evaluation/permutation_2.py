import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# ✅ MiniROCKETWrapper 정의
class MiniRocketWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=1000):
        self.num_kernels = num_kernels
        self.transformer = MiniRocketMultivariate(num_kernels=self.num_kernels)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

# ✅ 경로
model_path = "C:/BitaminDirectory/Sleep_Anomaly_Detection/trained_models/minirocket_rf_pipeline.pkl"
split_dir = "C:/BitaminDirectory/Sleep_Anomaly_Detection/Database/processed/split"

# ✅ 데이터 로딩
def load_val_data(split_dir):
    def partial_load(filename):
        path = os.path.join(split_dir, filename)
        return np.load(path, mmap_mode="r")
    
    X_val = np.concatenate([partial_load("X_val_0.npy"), partial_load("X_val_1.npy")])
    y_val = np.concatenate([partial_load("y_val_0.npy"), partial_load("y_val_1.npy")])
    return X_val, y_val

# ✅ 모델 로드
print("📦 모델 불러오는 중...")
model = joblib.load(model_path)
rf_model = model.named_steps["randomforestclassifier"]

# ✅ 데이터 로드
print("📂 검증 데이터 로딩 중...")
X_val, y_val = load_val_data(split_dir)
X_val = np.transpose(X_val, (0, 2, 1))  # (N, T, C)
X_val_feat = model.named_steps["minirocketwrapper"].transform(X_val)

# ✅ 라벨별 분리
X_0, y_0 = X_val_feat[y_val == 0], y_val[y_val == 0]
X_1, y_1 = X_val_feat[y_val == 1], y_val[y_val == 1]

# ✅ Permutation Importance
print("🔍 라벨 0 (Normal) 중요도 계산 중...")
result_0 = permutation_importance(rf_model, X_0, y_0, n_repeats=10, random_state=42, n_jobs=-1)

print("🔍 라벨 1 (Anomaly) 중요도 계산 중...")
result_1 = permutation_importance(rf_model, X_1, y_1, n_repeats=10, random_state=42, n_jobs=-1)

# ✅ 시각화
plt.figure(figsize=(12, 5))
plt.plot(result_0.importances_mean, label="Label 0 (Normal)", color='blue', alpha=0.7)
plt.plot(result_1.importances_mean, label="Label 1 (Anomaly)", color='red', alpha=0.7)
plt.title("Permutation Importance by Class")
plt.xlabel("Kernel Index")
plt.ylabel("Mean Importance")
plt.legend()
plt.tight_layout()
plt.show()
