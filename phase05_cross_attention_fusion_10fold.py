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

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR = Path("phase05_cross_attention_fusion_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 25
PATIENCE = 5
LR = 7e-4
WEIGHT_DECAY = 1e-4

PRECISION_FLOOR = 0.90
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)

# ============================================================
# METRICS
# ============================================================
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
    rows = [compute_metrics(y_true, y_prob, thr=t) for t in THRESHOLDS]
    return pd.DataFrame(rows)

# ============================================================
# FEATURES
# ============================================================
def build_tab_features(url_df: pd.DataFrame, html_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "row_index": html_df["row_index"].values,
        "url": html_df["url"].values,
        "y_true": html_df["y_true"].astype(int).values,
        "html_prob": html_df["html_prob"].astype(float).values,
        "url_textcnn_prob": url_df["url_textcnn_prob"].astype(float).values,
        "url_rf_prob": url_df["url_rf_prob"].astype(float).values,
    })

    probs = ["html_prob", "url_textcnn_prob", "url_rf_prob"]

    df["d_url_html"] = np.abs(df["url_textcnn_prob"] - df["html_prob"])
    df["d_rf_html"] = np.abs(df["url_rf_prob"] - df["html_prob"])
    df["d_url_rf"] = np.abs(df["url_textcnn_prob"] - df["url_rf_prob"])

    df["mean_prob"] = df[probs].mean(axis=1)
    df["max_prob"] = df[probs].max(axis=1)
    df["min_prob"] = df[probs].min(axis=1)
    df["std_prob"] = df[probs].std(axis=1)
    df["range_prob"] = df["max_prob"] - df["min_prob"]

    df["html_conf"] = np.abs(df["html_prob"] - 0.5)
    df["url_conf"] = np.abs(df["url_textcnn_prob"] - 0.5)
    df["rf_conf"] = np.abs(df["url_rf_prob"] - 0.5)

    df["html_gt_url"] = (df["html_prob"] > df["url_textcnn_prob"]).astype(float)
    df["url_gt_html"] = (df["url_textcnn_prob"] > df["html_prob"]).astype(float)
    df["rf_gt_html"] = (df["url_rf_prob"] > df["html_prob"]).astype(float)

    df["html_strong_phish"] = (df["html_prob"] > 0.80).astype(float)
    df["url_strong_phish"] = (df["url_textcnn_prob"] > 0.80).astype(float)
    df["rf_strong_phish"] = (df["url_rf_prob"] > 0.80).astype(float)

    df["html_strong_benign"] = (df["html_prob"] < 0.20).astype(float)
    df["url_strong_benign"] = (df["url_textcnn_prob"] < 0.20).astype(float)
    df["rf_strong_benign"] = (df["url_rf_prob"] < 0.20).astype(float)

    df["html_vs_url_conflict"] = (
        (df["html_prob"] > 0.7) & (df["url_textcnn_prob"] < 0.3)
    ).astype(float)

    df["html_vs_rf_conflict"] = (
        (df["html_prob"] > 0.7) & (df["url_rf_prob"] < 0.3)
    ).astype(float)

    return df

TAB_COLS = [
    "html_prob",
    "url_textcnn_prob",
    "url_rf_prob",
    "d_url_html",
    "d_rf_html",
    "d_url_rf",
    "mean_prob",
    "max_prob",
    "min_prob",
    "std_prob",
    "range_prob",
    "html_conf",
    "url_conf",
    "rf_conf",
    "html_gt_url",
    "url_gt_html",
    "rf_gt_html",
    "html_strong_phish",
    "url_strong_phish",
    "rf_strong_phish",
    "html_strong_benign",
    "url_strong_benign",
    "rf_strong_benign",
    "html_vs_url_conflict",
    "html_vs_rf_conflict",
]

# ============================================================
# DATASET
# ============================================================
class CrossAttentionDataset(Dataset):
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

# ============================================================
# MODEL
# ============================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, tab_dim, url_dim, html_dim, d_model=64, n_heads=4):
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
            nn.Linear(128, d_model),
            nn.ReLU()
        )

        self.html_proj = nn.Sequential(
            nn.Linear(html_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, d_model),
            nn.ReLU()
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.10,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(32 + d_model * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_url, x_html):
        t = self.tab_proj(x_tab)      # [B, 32]
        u = self.url_proj(x_url)      # [B, d]
        h = self.html_proj(x_html)    # [B, d]

        seq = torch.stack([u, h], dim=1)  # [B, 2, d]

        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm1(seq + attn_out)

        ffn_out = self.ffn(seq)
        seq = self.norm2(seq + ffn_out)

        u_att = seq[:, 0, :]
        h_att = seq[:, 1, :]
        pooled = seq.mean(dim=1)
        diff = torch.abs(u_att - h_att)

        z = torch.cat([t, u_att, h_att, pooled, diff], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

# ============================================================
# TRAIN / PREDICT
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x_tab, x_url, x_html, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_url = x_url.to(DEVICE)
        x_html = x_html.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tab, x_url, x_html)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x_tab.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, trues = [], []
    for x_tab, x_url, x_html, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_url = x_url.to(DEVICE)
        x_html = x_html.to(DEVICE)

        logits = model(x_tab, x_url, x_html)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(probs)

# ============================================================
# MAIN
# ============================================================
def main():
    all_preds = []
    fold_metrics = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 05 CROSS-ATTENTION FUSION FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_url_feat = np.load(fold_dir / "train_url_features.npy").astype(np.float32)
        test_url_feat = np.load(fold_dir / "test_url_features.npy").astype(np.float32)
        train_html_feat = np.load(fold_dir / "train_html_features.npy").astype(np.float32)
        test_html_feat = np.load(fold_dir / "test_html_features.npy").astype(np.float32)

        train_df = build_tab_features(train_url, train_html)
        test_df = build_tab_features(test_url, test_html)

        X_tab_train = train_df[TAB_COLS].values.astype(np.float32)
        X_tab_test = test_df[TAB_COLS].values.astype(np.float32)

        y_train = train_df["y_true"].astype(int).values
        y_test = test_df["y_true"].astype(int).values

        scaler_tab = StandardScaler()
        scaler_url = StandardScaler()
        scaler_html = StandardScaler()

        X_tab_train = scaler_tab.fit_transform(X_tab_train)
        X_tab_test = scaler_tab.transform(X_tab_test)

        X_url_train = scaler_url.fit_transform(train_url_feat)
        X_url_test = scaler_url.transform(test_url_feat)

        X_html_train = scaler_html.fit_transform(train_html_feat)
        X_html_test = scaler_html.transform(test_html_feat)

        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(
            idx, test_size=0.1, stratify=y_train, random_state=SEED
        )

        ds_train = CrossAttentionDataset(X_tab_train[tr_idx], X_url_train[tr_idx], X_html_train[tr_idx], y_train[tr_idx])
        ds_val = CrossAttentionDataset(X_tab_train[val_idx], X_url_train[val_idx], X_html_train[val_idx], y_train[val_idx])
        ds_test = CrossAttentionDataset(X_tab_test, X_url_test, X_html_test, y_test)

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = CrossAttentionFusion(
            tab_dim=X_tab_train.shape[1],
            url_dim=X_url_train.shape[1],
            html_dim=X_html_train.shape[1],
            d_model=64,
            n_heads=4
        ).to(DEVICE)

        n_neg = int((y_train[tr_idx] == 0).sum())
        n_pos = int((y_train[tr_idx] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_prauc = -1.0
        wait = 0

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, dl_train, optimizer, criterion)
            y_val_true, y_val_prob = predict_probs(model, dl_val)
            prauc = average_precision_score(y_val_true, y_val_prob)
            print(f"[Fold {fold}] epoch {epoch:02d} | loss={loss:.4f} | val_pr_auc={prauc:.4f}")

            if prauc > best_val_prauc:
                best_val_prauc = prauc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"[Fold {fold}] early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        y_test_true, y_test_prob = predict_probs(model, dl_test)
        metrics = compute_metrics(y_test_true, y_test_prob, thr=0.5)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

        pred_df = test_df[["row_index", "url", "y_true"]].copy()
        pred_df["cross_attention_prob"] = y_test_prob
        pred_df["cross_attention_pred"] = (y_test_prob >= 0.5).astype(int)
        pred_df["fold"] = fold
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase05_cross_attention_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase05_cross_attention_fold_metrics.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase05_cross_attention_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 05 CROSS-ATTENTION FUSION SUMMARY")
    print("=" * 70)
    print(summary)

    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["cross_attention_prob"].astype(float).values
    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase05_cross_attention_threshold_sweep.csv", index=False)

    best_mcc = sweep.sort_values(["MCC", "F1", "Recall"], ascending=[False, False, False]).iloc[0]
    best_f1 = sweep.sort_values(["F1", "MCC", "Recall"], ascending=[False, False, False]).iloc[0]
    filtered = sweep[sweep["Precision"] >= PRECISION_FLOOR].copy()
    best_low_fn = None if len(filtered) == 0 else filtered.sort_values(
        ["FN", "MCC", "F1"], ascending=[True, False, False]
    ).iloc[0]

    print("\nBest threshold by MCC")
    print(best_mcc.to_string())

    print("\nBest threshold by F1")
    print(best_f1.to_string())

    if best_low_fn is not None:
        print(f"\nBest threshold by lowest FN with precision >= {PRECISION_FLOOR:.2f}")
        print(best_low_fn.to_string())

    with open(OUT_DIR / "phase05_cross_attention_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase05_cross_attention_oof_predictions.csv")
    print(OUT_DIR / "phase05_cross_attention_fold_metrics.csv")
    print(OUT_DIR / "phase05_cross_attention_summary.csv")
    print(OUT_DIR / "phase05_cross_attention_threshold_sweep.csv")

if __name__ == "__main__":
    main()