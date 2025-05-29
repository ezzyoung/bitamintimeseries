
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model.AnomalyTransformer import AnomalyTransformer
from utils import detect_anomalies


import os

# 현재 파일 기준으로 프로젝트 루트 경로 계산
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# .npy 파일 경로 설정
X_val_path = os.path.join(BASE_DIR, "Database", "processed", "split", "X_val.npy")
y_val_path = os.path.join(BASE_DIR, "Database", "processed", "split", "y_val.npy")


# ✅ 데이터 로딩
X_val = np.load(X_val_path)
y_val = np.load(y_val_path)

X_val = X_val.transpose(0, 2, 1)  # → (B, T, C)

# Tensor 변환
X_tensor = torch.tensor(X_val, dtype=torch.float32)
y_tensor = torch.tensor(y_val, dtype=torch.long)
val_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=1)

# ✅ 모델 초기화 (GitHub 모델 구조 기준)
model = AnomalyTransformer(
    win_size=X_tensor.shape[1],  # T
    enc_in=X_tensor.shape[2],   # C
    c_out=1,
    e_layers=3,
    d_model=64,
    n_heads=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 학습된 가중치 로드 (필요 시)
# model.load_state_dict(torch.load("your_checkpoint.pth"))

# ✅ 이상치 탐지 실행
preds, scores, threshold = detect_anomalies(model, val_loader, device)

# ✅ 결과 확인
print(f"🔍 Threshold: {threshold:.4f}")
print(f"📊 예측값 (0=정상, 1=이상): {preds[:20]}")
print(f"✅ 실제값 (GT):             {y_val[:20]}")
