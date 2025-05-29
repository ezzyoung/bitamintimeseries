import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# ✅ 저장된 모델을 불러오기 위한 MiniRocketWrapper 정의 필요
class MiniRocketWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=1000):
        self.num_kernels = num_kernels
        self.transformer = MiniRocketMultivariate(num_kernels=self.num_kernels)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

# ✅ 모델 및 데이터 경로
model_path = "C:/BitaminDirectory/Sleep_Anomaly_Detection/trained_models/minirocket_rf_pipeline.pkl"
split_dir = "C:/BitaminDirectory/Sleep_Anomaly_Detection/Database/processed/split"

# ✅ 데이터 로드 함수
def load_val_data(split_dir):
    def partial_load(filename):
        return np.load(f"{split_dir}/{filename}", mmap_mode="r")
    X_val = np.concatenate([partial_load("X_val_0.npy"), partial_load("X_val_1.npy")])
    y_val = np.concatenate([partial_load("y_val_0.npy"), partial_load("y_val_1.npy")])
    return X_val, y_val

# ✅ 실행
model = joblib.load(model_path)
X_val, y_val = load_val_data(split_dir)
X_val = np.transpose(X_val, (0, 2, 1))  # (N, T, C)

# 🎯 MiniROCKET features 추출
X_val_feat = model.named_steps['minirocketwrapper'].transform(X_val)
rf_model = model.named_steps['randomforestclassifier']

# 🔍 Permutation Importance
result = permutation_importance(rf_model, X_val_feat, y_val, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
indices = np.argsort(importances)[::-1][:20]

# 📈 시각화
plt.figure(figsize=(10, 5))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [f"Kernel {i}" for i in indices], rotation=45)
plt.title("Top 20 Important Features (Permutation Importance)")
plt.tight_layout()
plt.show()
