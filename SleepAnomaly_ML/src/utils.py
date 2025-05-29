# src/utils.py

import torch
import numpy as np

def compute_kl_divergence(series, prior):
    eps = 1e-5
    series = series.clamp(min=eps)
    prior = prior.clamp(min=eps)
    kl = series * torch.log(series / prior)
    score = kl.sum(-1).mean(-1).mean(0)  # shape: (layers,) → 평균
    return score.cpu().numpy()

def detect_anomalies(model, dataloader, device, threshold=None):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            series, prior, _ = model(batch_x, return_attention=True)
            kl_score = compute_kl_divergence(series, prior)
            scores.append(kl_score.mean())  # sample-level score

    scores = np.array(scores)

    if threshold is None:
        threshold = np.percentile(scores, 95)  # top 5% 기준

    preds = (scores > threshold).astype(int)
    return preds, scores, threshold
