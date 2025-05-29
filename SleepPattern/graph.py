import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

df = pd.read_csv("auto_based_sleep_clustering.csv")
annotations_path = "./sleep-cassette/annotations/"
n_classes = 5  # W, N1, N2, N3, REM

def compute_ratio(label):
    counter = Counter(label)
    total = sum(counter.values())
    return np.array([counter.get(i, 0) / total for i in range(n_classes)])

# 1. 클러스터별 비율 평균 계산
cluster_ratios = {}
cluster_samples = {}

for cluster_id in sorted(df['cluster_autoencoder'].unique()):
    cluster_files = df[df['cluster_autoencoder'] == cluster_id]['filename'].tolist()
    ratio_list = []
    valid_files = []

    for fname in cluster_files:
        path = os.path.join(annotations_path, fname)
        if not os.path.exists(path):
            continue
        label = np.load(path)
        label = label[label < n_classes]
        if len(label) < 10:
            continue
        ratio = compute_ratio(label)
        ratio_list.append(ratio)
        valid_files.append((fname, ratio, label))

    if not ratio_list:
        continue

    mean_ratio = np.mean(ratio_list, axis=0)
    cluster_ratios[cluster_id] = mean_ratio

    # 2. 대표 샘플 = 평균 비율과 가장 가까운 샘플
    best_file = None
    min_dist = float('inf')
    best_label = None

    for fname, ratio, label in valid_files:
        dist = np.linalg.norm(ratio - mean_ratio)
        if dist < min_dist:
            best_file = fname
            best_label = label
            min_dist = dist

    cluster_samples[cluster_id] = (best_file, best_label)

# 3. 시각화 (Hypnogram 구조)
stage_labels = ["W", "N1", "N2", "N3", "REM"]

plt.figure(figsize=(12, 2.5 * len(cluster_samples)))

for i, (cluster_id, (fname, label)) in enumerate(cluster_samples.items()):
    ax = plt.subplot(len(cluster_samples), 1, i + 1)
    ax.step(range(len(label)), label, where="mid")
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(stage_labels)
    ax.invert_yaxis()
    ax.set_title(f"Cluster {cluster_id} - Representative: {fname}")
    ax.set_xlabel("Epoch (30s each)")
    ax.set_ylabel("Stage")

plt.tight_layout()
plt.show()
