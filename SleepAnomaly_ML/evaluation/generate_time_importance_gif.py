import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import copy
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.panel.rocket import MiniRocketMultivariate

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
output_dir = os.path.join(BASE_DIR, "results", "time_importance", "frames")
os.makedirs(output_dir, exist_ok=True)

gif_output_path = os.path.join(BASE_DIR, "results", "time_importance", "time_importance.gif")

# ✅ 데이터 로드 함수
def load_val_data(split_dir):
    def partial_load(filename):
        return np.load(os.path.join(split_dir, filename), mmap_mode="r")
    X_val = np.concatenate([partial_load("X_val_0.npy"), partial_load("X_val_1.npy")])
    y_val = np.concatenate([partial_load("y_val_0.npy"), partial_load("y_val_1.npy")])
    return X_val, y_val

# ✅ 중요도 계산 함수
def compute_temporal_importance(X_sample, model, window_size=30):
    X_sample = X_sample[None, :, :]  # (1, T, C)
    original_prob = model.predict_proba(X_sample)[0][1]
    T = X_sample.shape[1]
    importance = np.zeros(T)
    for start in range(0, T, window_size):
        end = min(start + window_size, T)
        X_masked = copy.deepcopy(X_sample)
        X_masked[0, start:end, :] = 0
        prob = model.predict_proba(X_masked)[0][1]
        importance[start:end] = abs(original_prob - prob)
    return importance

# ✅ 프레임 저장 함수
def save_frame_plot(importance, idx, save_path):
    plt.figure(figsize=(12, 3))
    plt.plot(importance, label="Time Importance", color="steelblue")
    plt.title(f"Sample #{idx} - Time-wise Importance")
    plt.xlabel("Time Index")
    plt.ylabel("Impact on Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ✅ 메인 실행
if __name__ == "__main__":
    print("📦 모델 로드 중...")
    model = joblib.load(model_path)

    print("📂 데이터 로드 중...")
    X_val, y_val = load_val_data(split_dir)
    X_val = np.transpose(X_val, (0, 2, 1))  # (N, T, C)

    sample_count = 10  # GIF에 포함할 샘플 수
    frame_paths = []

    print("🎞️ 프레임 생성 중...")
    for idx in tqdm(range(sample_count)):
        X_sample = X_val[idx]
        importance = compute_temporal_importance(X_sample, model)

        frame_path = os.path.join(output_dir, f"frame_{idx:02d}.png")
        save_frame_plot(importance, idx, frame_path)
        frame_paths.append(frame_path)

    print("🎬 GIF 생성 중...")
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(gif_output_path, images, duration=1.5)  # 1.5초 간격

    print(f"✅ GIF 저장 완료: {gif_output_path}")
