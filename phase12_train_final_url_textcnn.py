import os
import json
import random
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

# ============================================================
# CONFIG
# ============================================================
CFG = {
    "data_path": "final_multimodal_dataset.parquet",
    "url_column": "url",
    "label_column": "label",

    "cache_dir": "cache_domurlbert_new/cache_domurlbert_new",
    "meta_file": "meta.json",
    "emb_file": "embeddings.npy",

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    "batch_size": 128,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,

    "num_filters": 128,
    "kernels": (3, 5, 7),
    "dropout": 0.5,
}

OUT_DIR = Path("phase12_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(CFG["device"])


# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])


# ============================================================
# LOADERS
# ============================================================
def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_domurlbert_cache(cache_dir: str, n_expected: int):
    meta_path = os.path.join(cache_dir, CFG["meta_file"])
    emb_path = os.path.join(cache_dir, CFG["emb_file"])

    meta = load_meta(meta_path)
    n_cache = int(meta["n_samples"])
    L = int(meta["max_tokens"])
    H = int(meta["hidden_size"])
    dtype = np.float16 if meta["dtype"] == "float16" else np.float32

    if n_cache != n_expected:
        raise ValueError(f"Cache n_samples={n_cache} but dataset has {n_expected}")

    emb_mm = np.memmap(emb_path, dtype=dtype, mode="r", shape=(n_cache, L, H))
    return meta, emb_mm

def load_dataset():
    df = pd.read_parquet(CFG["data_path"]).copy()
    urls = df[CFG["url_column"]].astype(str).values
    y = (df[CFG["label_column"]].astype(str).str.lower() == "phish").astype(int).values
    return df, urls, y


# ============================================================
# DATASET
# ============================================================
class CachedEmbeddingDataset(Dataset):
    def __init__(self, emb_mm, labels):
        self.emb_mm = emb_mm
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.emb_mm[i], copy=True)).float()
        y = torch.tensor(int(self.labels[i]), dtype=torch.long)
        idx = torch.tensor(int(i), dtype=torch.long)
        return x, y, idx


# ============================================================
# MODEL
# ============================================================
class DomURLBERTTextCNN(nn.Module):
    def __init__(self, hidden_size, num_filters=128, kernels=(3, 5, 7), dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = num_filters * len(kernels)
        self.fc = nn.Linear(self.feature_dim, 2)

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            feats.append(torch.max(h, dim=2).values)

        z = torch.cat(feats, dim=1)
        z = self.dropout(z)
        logits = self.fc(z)

        if return_features:
            return logits, z
        return logits


# ============================================================
# TRAIN / PREDICT
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X, y, _ in loader:
        X = X.to(DEVICE).float()
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict_with_features(model, loader):
    model.eval()
    probs, preds, trues, feats, row_ids = [], [], [], [], []

    for X, y, idx in loader:
        X = X.to(DEVICE).float()
        logits, z = model(X, return_features=True)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()

        probs.extend(prob.tolist())
        preds.extend(pred.tolist())
        trues.extend(y.numpy().tolist())
        feats.append(z.cpu().numpy())
        row_ids.extend(idx.numpy().tolist())

    feats = np.concatenate(feats, axis=0)
    return np.array(trues), np.array(preds), np.array(probs), feats, np.array(row_ids)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Matthews CC": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "Log Loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("Using device:", DEVICE)

    df, urls, labels = load_dataset()
    meta, emb_mm = load_domurlbert_cache(CFG["cache_dir"], n_expected=len(df))
    hidden_size = int(meta["hidden_size"])

    print(f"Rows: {len(df)}")
    print(f"Embedding cache: N={meta['n_samples']} L={meta['max_tokens']} H={meta['hidden_size']}")

    ds = CachedEmbeddingDataset(emb_mm, labels)
    dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=CFG["num_workers"])
    dl_eval = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"])

    model = DomURLBERTTextCNN(
        hidden_size=hidden_size,
        num_filters=CFG["num_filters"],
        kernels=CFG["kernels"],
        dropout=CFG["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, CFG["epochs"] + 1):
        loss = train_epoch(model, dl, optimizer, criterion)
        print(f"Epoch {epoch:02d} | train_loss={loss:.4f}")

    y_true, y_pred, y_prob, feats, row_ids = predict_with_features(model, dl_eval)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\nFINAL URL FULL-DATA METRICS")
    for k, v in metrics.items():
        if k != "Confusion Matrix":
            print(f"{k:<20} {v:.4f}")

    probs_full = np.zeros(len(df), dtype=np.float32)
    preds_full = np.zeros(len(df), dtype=np.int32)
    feats_full = np.zeros((len(df), feats.shape[1]), dtype=np.float32)

    probs_full[row_ids] = y_prob
    preds_full[row_ids] = y_pred
    feats_full[row_ids] = feats

    np.save(OUT_DIR / "final_url_probs.npy", probs_full)
    np.save(OUT_DIR / "final_url_preds.npy", preds_full)
    np.save(OUT_DIR / "final_url_features.npy", feats_full)

    out_df = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "url": urls,
        "y_true": labels,
        "final_url_prob": probs_full,
        "final_url_pred": preds_full,
    })
    out_df.to_csv(OUT_DIR / "final_url_predictions.csv", index=False)

    with open(OUT_DIR / "final_url_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "final_url_probs.npy")
    print(OUT_DIR / "final_url_preds.npy")
    print(OUT_DIR / "final_url_features.npy")
    print(OUT_DIR / "final_url_predictions.csv")
    print("Feature shape:", feats_full.shape)

if __name__ == "__main__":
    main()