import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path("phase20_branch_cv_outputs")
PHASE6_OOF = "phase06_multichannel_joint_outputs/phase06_multichannel_oof_predictions.csv"
OUT_DIR = Path("phase10_error_specialist_router_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5
LR = 8e-4
WEIGHT_DECAY = 1e-4

# routing thresholds
ROUTE_LOW = 0.20
ROUTE_HIGH = 0.80

PRECISION_FLOOR = 0.90
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Log Loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Threshold": float(thr),
    }

def build_threshold_sweep(y_true, y_prob):
    return pd.DataFrame([compute_metrics(y_true, y_prob, thr=t) for t in THRESHOLDS])

def build_tab_features(url_df: pd.DataFrame, html_df: pd.DataFrame, main_prob: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "row_index": html_df["row_index"].values,
        "url": html_df["url"].values,
        "y_true": html_df["y_true"].astype(int).values,
        "html_prob": html_df["html_prob"].astype(float).values,
        "url_textcnn_prob": url_df["url_textcnn_prob"].astype(float).values,
        "url_rf_prob": url_df["url_rf_prob"].astype(float).values,
        "main_prob": main_prob.astype(float),
    })

    probs = ["html_prob", "url_textcnn_prob", "url_rf_prob", "main_prob"]

    df["d_url_html"] = np.abs(df["url_textcnn_prob"] - df["html_prob"])
    df["d_rf_html"] = np.abs(df["url_rf_prob"] - df["html_prob"])
    df["d_main_html"] = np.abs(df["main_prob"] - df["html_prob"])
    df["d_main_url"] = np.abs(df["main_prob"] - df["url_textcnn_prob"])

    df["mean_prob"] = df[probs].mean(axis=1)
    df["max_prob"] = df[probs].max(axis=1)
    df["min_prob"] = df[probs].min(axis=1)
    df["std_prob"] = df[probs].std(axis=1)
    df["range_prob"] = df["max_prob"] - df["min_prob"]

    df["main_conf"] = np.abs(df["main_prob"] - 0.5)
    df["html_conf"] = np.abs(df["html_prob"] - 0.5)
    df["url_conf"] = np.abs(df["url_textcnn_prob"] - 0.5)
    df["rf_conf"] = np.abs(df["url_rf_prob"] - 0.5)

    df["main_uncertain"] = ((df["main_prob"] >= ROUTE_LOW) & (df["main_prob"] <= ROUTE_HIGH)).astype(float)
    df["html_vs_url_conflict"] = ((df["html_prob"] > 0.7) & (df["url_textcnn_prob"] < 0.3)).astype(float)
    df["main_vs_html_conflict"] = ((df["main_prob"] > 0.7) & (df["html_prob"] < 0.3) | (df["main_prob"] < 0.3) & (df["html_prob"] > 0.7)).astype(float)

    return df

TAB_COLS = [
    "html_prob", "url_textcnn_prob", "url_rf_prob", "main_prob",
    "d_url_html", "d_rf_html", "d_main_html", "d_main_url",
    "mean_prob", "max_prob", "min_prob", "std_prob", "range_prob",
    "main_conf", "html_conf", "url_conf", "rf_conf",
    "main_uncertain", "html_vs_url_conflict", "main_vs_html_conflict",
]

class SpecialistDataset(Dataset):
    def __init__(self, X_tab, X_url, X_html, y):
        self.X_tab = X_tab.astype(np.float32)
        self.X_url = X_url.astype(np.float32)
        self.X_html = X_html.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X_tab[i], dtype=torch.float32),
            torch.tensor(self.X_url[i], dtype=torch.float32),
            torch.tensor(self.X_html[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )

class ErrorSpecialist(nn.Module):
    def __init__(self, tab_dim, url_dim, html_dim):
        super().__init__()

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.url_proj = nn.Sequential(
            nn.Linear(url_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.html_proj = nn.Sequential(
            nn.Linear(html_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential
