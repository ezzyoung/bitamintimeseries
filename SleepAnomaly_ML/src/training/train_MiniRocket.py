import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ 사용자 정의 래퍼: transformer도 pipeline에 넣기 위함
class MiniRocketWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=1000):
        self.num_kernels = num_kernels
        self.transformer = MiniRocketMultivariate(num_kernels=self.num_kernels)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

# ✅ 경로 설정
BASE_DIR = "c:/BitaminDirectory/Sleep_Anomaly_Detection"
split_dir = os.path.join(BASE_DIR, "Database", "processed", "split")
model_save_path = os.path.join(BASE_DIR, "trained_models", "minirocket_rf_pipeline.pkl")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# ✅ 데이터 로드
def load_data(split_dir, train_ratio=0.6, val_ratio=0.6):
    def partial_load(filename, ratio):
        path = os.path.join(split_dir, filename)
        arr = np.load(path, mmap_mode="r")
        sample_count = max(1, int(arr.shape[0] * ratio))
        return arr[:sample_count]

    X_train = np.concatenate([
        partial_load("X_train_0.npy", train_ratio),
        partial_load("X_train_1.npy", train_ratio)
    ])
    y_train = np.concatenate([
        partial_load("y_train_0.npy", train_ratio),
        partial_load("y_train_1.npy", train_ratio)
    ])
    X_val = np.concatenate([
        partial_load("X_val_0.npy", val_ratio),
        partial_load("X_val_1.npy", val_ratio)
    ])
    y_val = np.concatenate([
        partial_load("y_val_0.npy", val_ratio),
        partial_load("y_val_1.npy", val_ratio)
    ])
    return X_train, y_train, X_val, y_val

# ✅ 학습 및 평가
def run_minirocket_rf(X_train, y_train, X_val, y_val):
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))

    pipeline = make_pipeline(
        MiniRocketWrapper(num_kernels=1000),
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    )

    pipeline.fit(X_train, y_train)

    # ✅ 저장
    joblib.dump(pipeline, model_save_path)
    print(f"\n💾 모델 저장 완료: {model_save_path}")

    # ✅ 학습 평가
    y_train_pred = pipeline.predict(X_train)
    y_train_score = pipeline.predict_proba(X_train)[:, 1]

    print("\n✅ [Train Set] 평가 결과")
    print(classification_report(y_train, y_train_pred, digits=4))
    print(f"AUC       : {roc_auc_score(y_train, y_train_score):.4f}")
    print(f"F1-score  : {f1_score(y_train, y_train_pred):.4f}")
    print(f"Precision : {precision_score(y_train, y_train_pred):.4f}")
    print(f"Recall    : {recall_score(y_train, y_train_pred):.4f}")

    # ✅ 검증 평가
    y_val_pred = pipeline.predict(X_val)
    y_val_score = pipeline.predict_proba(X_val)[:, 1]

    print("\n📊 [Validation Set] 평가 결과")
    print(classification_report(y_val, y_val_pred, digits=4))
    print(f"AUC       : {roc_auc_score(y_val, y_val_score):.4f}")
    print(f"F1-score  : {f1_score(y_val, y_val_pred):.4f}")
    print(f"Precision : {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall    : {recall_score(y_val, y_val_pred):.4f}")

# ✅ 실행
if __name__ == "__main__":
    print("📂 데이터 로드 중...")
    X_train, y_train, X_val, y_val = load_data(split_dir, train_ratio=0.6, val_ratio=0.6)
    print(f"✅ 학습 샘플 수: {len(X_train)}, 검증 샘플 수: {len(X_val)}")
    print("🚀 MiniRocket + RandomForest 학습 시작")
    run_minirocket_rf(X_train, y_train, X_val, y_val)
