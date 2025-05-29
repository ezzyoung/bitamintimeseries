import joblib
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import os
import time

from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# ✅ 경고 보기 설정
warnings.filterwarnings("default")

# ✅ 사용자 정의 MiniROCKET 래퍼
class MiniRocketWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=1000):
        self.num_kernels = num_kernels
        self.transformer = MiniRocketMultivariate(num_kernels=self.num_kernels)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

# ✅ 경로 설정
BASE_DIR = "C:/BitaminDirectory/Sleep_Anomaly_Detection"
model_path = os.path.join(BASE_DIR, "trained_models", "minirocket_rf_pipeline.pkl")
split_dir = os.path.join(BASE_DIR, "Database", "processed", "split")
output_dir = os.path.join(BASE_DIR, "results", "time_importance")
os.makedirs(output_dir, exist_ok=True)

# ✅ 데이터 로드 함수
def load_val_data(split_dir):
    def partial_load(filename):
        path = os.path.join(split_dir, filename)
        print(f"📂 loading: {path}")
        return np.load(path, mmap_mode="r")
    X_val = np.concatenate([partial_load("X_val_0.npy"), partial_load("X_val_1.npy")])
    y_val = np.concatenate([partial_load("y_val_0.npy"), partial_load("y_val_1.npy")])
    print(f"✅ X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    return X_val, y_val

# ✅ 중요도 계산 함수 (transpose 제거됨)
def compute_temporal_importance(X_sample, model, window_size=30):
    X_sample = X_sample[None, :, :]  # (1, T, C) → MiniROCKET 예상 입력
    try:
        original_prob = model.predict_proba(X_sample)[0][1]
    except Exception as e:
        print("❌ 초기 예측 실패:", e)
        print(f"입력 shape: {X_sample.shape}, dtype: {X_sample.dtype}")
        return np.zeros(X_sample.shape[1])

    T = X_sample.shape[1]
    importance = np.zeros(T)

    print("🔍 중요도 계산 중...")
    for start in range(0, T, window_size):
        end = min(start + window_size, T)
        X_masked = copy.deepcopy(X_sample)
        X_masked[0, start:end, :] = 0
        try:
            prob = model.predict_proba(X_masked)[0][1]
            importance[start:end] = abs(original_prob - prob)
            print(f"⏳ {start:5d}-{end:5d} | Δprob = {abs(original_prob - prob):.5f}")
        except Exception as e:
            print(f"❌ 예측 실패 at {start}-{end}: {e}")
    return importance

# ✅ 시각화 및 저장 함수
def save_time_importance_plot(importance, idx):
    plt.figure(figsize=(12, 3))
    plt.plot(importance, label="Time Importance")
    plt.title(f"Time-wise Importance (Sample #{idx})")
    plt.xlabel("Time Index")
    plt.ylabel("Impact on Prediction")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"time_importance_sample{idx}.png")
    plt.savefig(save_path)
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.close()

# ✅ 실행
if __name__ == "__main__":
    try:
        print("📦 모델 불러오는 중...")
        model = joblib.load(model_path)

        print("📂 검증 데이터 로딩 중...")
        X_val, y_val = load_val_data(split_dir)
        print(f"✅ 전체 shape: {X_val.shape}")  # (N, C, T)

        idx = 0  # 분석할 샘플 인덱스
        X_sample = np.transpose(X_val[idx], (1, 0))  # (T, C) ← MiniROCKET용

        print(f"🔍 Sample #{idx} 분석 시작")
        start_time = time.time()

        importance = compute_temporal_importance(X_sample, model)
        save_time_importance_plot(importance, idx)

        print(f"🎯 완료! 총 소요 시간: {time.time() - start_time:.2f}초")

    except Exception as e:
        print("❌ 전체 오류 발생:", e)
