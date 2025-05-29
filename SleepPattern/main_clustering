import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import umap
import markov_clustering as mc
from collections import Counter

# === 수면 지표 계산 함수 ===
def extract_sleep_features(label, n_classes=6):
    features = {}
    label = label[label < n_classes]

    features['total_sleep_time'] = np.sum(label != 5)
    sleep_onset_indices = np.where((label == 1) | (label == 2))[0]
    features['sleep_latency'] = sleep_onset_indices[0] if len(sleep_onset_indices) > 0 else len(label)

    wake_indices = np.where(label == 5)[0]
    if len(wake_indices) == 0:
        features['wake_count'] = 0
    else:
        wake_transitions = np.diff(wake_indices) > 1
        features['wake_count'] = 1 + np.sum(wake_transitions)

    features['rem_ratio'] = np.sum(label == 4) / len(label)
    features['deep_sleep_ratio'] = np.sum(label == 3) / len(label)

    trans_matrix = np.zeros((n_classes, n_classes))
    for i in range(len(label) - 1):
        a, b = label[i], label[i+1]
        trans_matrix[a, b] += 1
    flat_trans = trans_matrix.flatten()
    features['transition_entropy'] = entropy(flat_trans + 1e-8)

    transitions = [(label[i], label[i+1]) for i in range(len(label) - 1)]
    nrem_rem_cycles = sum((a in [2, 3]) and (b == 4) for a, b in transitions)
    features['cycle_count'] = nrem_rem_cycles

    return features

# === 데이터 로딩 및 Feature 추출 ===
annotations_path = "./sleep-cassette/annotations/"
label_files = [f for f in os.listdir(annotations_path) if f.endswith(".npy")]

feature_list = []
file_ids = []

for file in tqdm(label_files):
    label = np.load(os.path.join(annotations_path, file))
    if len(label) < 10:
        continue
    features = extract_sleep_features(label)
    feature_list.append(features)
    file_ids.append(file)

df_features = pd.DataFrame(feature_list)
df_features["filename"] = file_ids

# === Autoencoder 정의 ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# === 데이터 준비
X = df_features.drop(columns=["filename"]).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_x)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# === Autoencoder 학습
input_dim = X.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(100):
    for batch in dataloader:
        inputs = batch[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === 재구성 오차 기반 중요도 계산
model.eval()
with torch.no_grad():
    reconstructed = model(tensor_x).numpy()

recon_error = np.mean((X_scaled - reconstructed) ** 2, axis=0)
autoencoder_importance = pd.DataFrame({
    "feature": df_features.drop(columns=["filename"]).columns,
    "importance": recon_error
}).sort_values(by="importance", ascending=False)

print(f"feature importance", autoencoder_importance)

# === 가중치 시각화 ===
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(autoencoder_importance["feature"], autoencoder_importance["importance"])
plt.xlabel("Importance")
plt.title("Autoencoder Feature Importance")
plt.gca().invert_yaxis()
plt.show()

print("feature_importance:")
print(autoencoder_importance)

# === 전체 feature에 가중치 곱한 후 정규화
weights = autoencoder_importance.set_index("feature")["importance"]
X_weighted = X * weights.values
X_weighted_scaled = StandardScaler().fit_transform(X_weighted)


# === 클러스터링 (UMAP + MCL)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding_2d = reducer.fit_transform(X_weighted_scaled)
umap_graph = reducer.graph_

result = mc.run_mcl(umap_graph, inflation=1.4)
clusters = mc.get_clusters(result)

cluster_labels = np.zeros(len(file_ids), dtype=int)
for cluster_id, indices in enumerate(clusters):
    for idx in indices:
        cluster_labels[idx] = cluster_id

df_features["cluster_autoencoder"] = cluster_labels

print(f"Cluster numbers:", len(np.unique(cluster_labels)))

# === 수면 질 점수 계산
weight_dict_autoencoder = weights.to_dict()
df_features["sleep_score_autoencoder"] = df_features.apply(
    lambda row: sum(row[feat] * weight_dict_autoencoder[feat] for feat in weight_dict_autoencoder), axis=1
)
df_features["sleep_score_autoencoder"] = df_features["sleep_score_autoencoder"].rank(pct=True)

# === 클러스터별 평균 수면 질 점수 출력
cluster_scores_autoencoder = df_features.groupby("cluster_autoencoder")["sleep_score_autoencoder"].mean().reset_index()
cluster_scores_autoencoder = cluster_scores_autoencoder.sort_values(by="sleep_score_autoencoder", ascending=False)
print("Cluster scores based on Autoencoder importance:")
print(cluster_scores_autoencoder)


feature_means_by_cluster = df_features.groupby("cluster_autoencoder")[list(weight_dict_autoencoder.keys())].mean().reset_index()

print("클러스터별 평균 수면 지표:")
print(feature_means_by_cluster.to_string(index=False))

# ==== 클러스터별 수면 시간 비율
def compute_stage_ratio(label):
    counter = Counter(label)
    total = sum(counter.values())
    return np.array([counter.get(i, 0) / total for i in range(5)])

# 클러스터별로 비율 계산
cluster_ratios = {}

for cluster_id in sorted(df_features["cluster_autoencoder"].unique()):
    files = df_features[df_features["cluster_autoencoder"] == cluster_id]["filename"]
    ratios = []

    for fname in files:
        path = os.path.join(annotations_path, fname)
        if not os.path.exists(path):
            continue
        label = np.load(path)
        label = label[label < 5]  # stage 0~4만
        if len(label) < 10:
            continue
        ratios.append(compute_stage_ratio(label))

    if ratios:
        cluster_ratios[cluster_id] = np.mean(ratios, axis=0)

# 결과 정리
stage_labels = ["W", "N1", "N2", "N3", "REM"]
ratio_df = pd.DataFrame.from_dict(cluster_ratios, orient="index", columns=stage_labels)
ratio_df.index.name = "cluster_autoencoder"

print("📊 클러스터별 수면 단계 비율 (%):")
print((ratio_df * 100).round(2))


# === 저장
df_features.to_csv("auto_based_sleep_clustering.csv", index=False)
