

import os
import pickle
import json
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import joblib


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "naive_bayes.pkl")
METRICS_SAVE_PATH = os.path.join(METRICS_DIR, "naive_bayes.json")


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("  Gaussian Naive Bayes Training")
    print("=" * 60)

    #  Load Data
    print("[load] Loading data...")
    try:
        X_train = pd.read_csv(X_TRAIN_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
        y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    except FileNotFoundError as e:
        print(f"[error] Data file not found: {e}")
        return


    print("[scale] Loading scaler and transforming data...")
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except FileNotFoundError:
        print(f"[error] Scaler not found at {SCALER_PATH}. Run preprocess.py first.")
        return

    #  Initialize and Train Model
    print("[train] Training Gaussian Naive Bayes...")
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    #  Predict
    print("[predict] Generating predictions...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    #  Calculate Metrics
    print("[metrics] Calculating evaluation metrics...")
    metrics = {
        "model": "Naive Bayes (Gaussian)",
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    # Print Metrics
    print("-" * 30)
    for k, v in metrics.items():
        if k == "model": continue
        print(f"{k.capitalize():<15}: {v:.4f}")
    print("-" * 30)

    #  Save Model
    print(f"[save] Saving model to {MODEL_SAVE_PATH}...")
    joblib.dump(model, MODEL_SAVE_PATH)

    #  Save Metrics
    print(f"[save] Saving metrics to {METRICS_SAVE_PATH}...")
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("=" * 60)
    print("  Training complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
