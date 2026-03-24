import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)

# ============================================================
# CONFIG
# ============================================================
HTML_PRED_PATH = "phase7_outputs/phase7_all_predictions.csv"
HTML_FEAT_PATH = "phase7_outputs/oof_html_phase7_features.npy"
URL_OOF_PATH   = "domurlbert_textcnn_rf_url_only_oof_predictions.csv"

OUT_DIR = Path("phase10_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
N_SPLITS = 10
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
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
        "Threshold": thr
    }

def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.01):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t

# ============================================================
# LOAD + BUILD FUSION TABLE
# ============================================================
def load_and_merge():
    html_df = pd.read_csv(HTML_PRED_PATH)
    url_df = pd.read_csv(URL_OOF_PATH)
    html_feats = np.load(HTML_FEAT_PATH)

    required_html = {"row_index", "y_true", "y_prob"}
    required_url = {"row_index", "y_true", "textcnn_prob", "rf_prob"}

    if not required_html.issubset(html_df.columns):
        raise ValueError(f"HTML prediction file missing columns: {required_html - set(html_df.columns)}")
    if not required_url.issubset(url_df.columns):
        raise ValueError(f"URL OOF file missing columns: {required_url - set(url_df.columns)}")

    html_df = html_df.sort_values("row_index").reset_index(drop=True)
    url_df = url_df.sort_values("row_index").reset_index(drop=True)

    if len(html_df) != len(url_df) or len(html_df) != len(html_feats):
        raise ValueError("Length mismatch among HTML preds, URL preds, and HTML features")

    if not np.array_equal(html_df["row_index"].values, url_df["row_index"].values):
        raise ValueError("row_index mismatch between HTML and URL files")

    if not np.array_equal(html_df["y_true"].values, url_df["y_true"].values):
        raise ValueError("y_true mismatch between HTML and URL files")

    fusion_df = pd.DataFrame({
        "row_index": html_df["row_index"].values,
        "y_true": html_df["y_true"].astype(int).values,
        "html_prob": html_df["y_prob"].astype(float).values,
        "url_textcnn_prob": url_df["textcnn_prob"].astype(float).values,
        "url_rf_prob": url_df["rf_prob"].astype(float).values,
    })

    fusion_df["d_url_html"] = np.abs(fusion_df["url_textcnn_prob"] - fusion_df["html_prob"])
    fusion_df["d_rf_html"] = np.abs(fusion_df["url_rf_prob"] - fusion_df["html_prob"])
    fusion_df["d_url_rf"] = np.abs(fusion_df["url_textcnn_prob"] - fusion_df["url_rf_prob"])

    fusion_df["mean_prob"] = fusion_df[["html_prob", "url_textcnn_prob", "url_rf_prob"]].mean(axis=1)
    fusion_df["max_prob"] = fusion_df[["html_prob", "url_textcnn_prob", "url_rf_prob"]].max(axis=1)
    fusion_df["min_prob"] = fusion_df[["html_prob", "url_textcnn_prob", "url_rf_prob"]].min(axis=1)

    return fusion_df, html_feats

# ============================================================
# DATASET
# ============================================================
class FusionMLPDataset(Dataset):
    def __init__(self, X_tab, X_html, y):
        self.X_tab = X_tab.astype(np.float32)
        self.X_html = X_html.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X_tab[i], dtype=torch.float32),
            torch.tensor(self.X_html[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32)
        )

# ============================================================
# MODEL
# ============================================================
class GatedFusionMLP(nn.Module):
    def __init__(self, tab_dim, html_dim):
        super().__init__()

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.html_proj = nn.Sequential(
            nn.Linear(html_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(32 + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_html):
        t = self.tab_proj(x_tab)      # [B, 32]
        h = self.html_proj(x_html)    # [B, 64]

        gate = self.gate(torch.cat([t, h], dim=1))   # [B, 1]
        h_gated = gate * h

        z = torch.cat([t, h, h_gated], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

# ============================================================
# TRAIN / PREDICT
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x_tab, x_html, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_html = x_html.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tab, x_html)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x_tab.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, trues = [], []

    for x_tab, x_html, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_html = x_html.to(DEVICE)

        logits = model(x_tab, x_html)
        prob = torch.sigmoid(logits).cpu().numpy()

        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())

    return np.array(trues), np.array(probs)

# ============================================================
# CV: LOGISTIC REGRESSION BASELINE
# ============================================================
def run_lr_baseline(X_tab, X_html, y):
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION FUSION BASELINE")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.zeros(len(y), dtype=np.float32)

    X_full = np.concatenate([X_tab, X_html], axis=1)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_full, y), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_full[tr_idx])
        Xte = scaler.transform(X_full[te_idx])

        clf = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED
        )
        clf.fit(Xtr, y[tr_idx])
        oof_prob[te_idx] = clf.predict_proba(Xte)[:, 1]

    thr = find_best_threshold(y, oof_prob)
    metrics = compute_metrics(y, oof_prob, thr=thr)

    for k, v in metrics.items():
        print(f"{k:<18} {v:.4f}")

    return oof_prob, metrics

# ============================================================
# CV: GATED MLP FUSION
# ============================================================
def run_gated_mlp(X_tab, X_html, y):
    print("\n" + "=" * 70)
    print("GATED MLP MULTIMODAL FUSION")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.zeros(len(y), dtype=np.float32)

    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_tab, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}")

        # scale tab features + html features separately
        scaler_tab = StandardScaler()
        scaler_html = StandardScaler()

        Xtab_tr = scaler_tab.fit_transform(X_tab[tr_idx])
        Xtab_te = scaler_tab.transform(X_tab[te_idx])

        Xhtml_tr = scaler_html.fit_transform(X_html[tr_idx])
        Xhtml_te = scaler_html.transform(X_html[te_idx])

        # split train into train/val
        inner_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=RANDOM_SEED)
        inner_train_idx, val_idx = next(inner_skf.split(Xtab_tr, y[tr_idx]))

        ds_train = FusionMLPDataset(Xtab_tr[inner_train_idx], Xhtml_tr[inner_train_idx], y[tr_idx][inner_train_idx])
        ds_val   = FusionMLPDataset(Xtab_tr[val_idx], Xhtml_tr[val_idx], y[tr_idx][val_idx])
        ds_test  = FusionMLPDataset(Xtab_te, Xhtml_te, y[te_idx])

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = GatedFusionMLP(
            tab_dim=X_tab.shape[1],
            html_dim=X_html.shape[1]
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        best_state = None
        best_val_prauc = -1.0
        wait = 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, dl_train, optimizer, criterion)
            y_val_true, y_val_prob = predict_probs(model, dl_val)
            val_prauc = average_precision_score(y_val_true, y_val_prob)

            print(f"  Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_pr_auc={val_prauc:.4f}")

            if val_prauc > best_val_prauc:
                best_val_prauc = val_prauc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)

        y_te_true, y_te_prob = predict_probs(model, dl_test)
        oof_prob[te_idx] = y_te_prob

        thr_fold = find_best_threshold(y_te_true, y_te_prob)
        m = compute_metrics(y_te_true, y_te_prob, thr=thr_fold)
        fold_metrics.append(m)

        print(f"  Fold Accuracy: {m['Accuracy']:.4f} | PR-AUC: {m['PR-AUC']:.4f} | MCC: {m['MCC']:.4f}")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    thr = find_best_threshold(y, oof_prob)
    metrics = compute_metrics(y, oof_prob, thr=thr)

    print("\nFINAL GATED MLP OOF METRICS")
    for k, v in metrics.items():
        print(f"{k:<18} {v:.4f}")

    return oof_prob, metrics

# ============================================================
# MAIN
# ============================================================
def main():
    print("Using device:", DEVICE)

    fusion_df, html_feats = load_and_merge()
    y = fusion_df["y_true"].astype(int).values

    # tabular fusion inputs
    tab_cols = [
        "html_prob",
        "url_textcnn_prob",
        "url_rf_prob",
        "d_url_html",
        "d_rf_html",
        "d_url_rf",
        "mean_prob",
        "max_prob",
        "min_prob",
    ]
    X_tab = fusion_df[tab_cols].values.astype(np.float32)
    X_html = html_feats.astype(np.float32)

    print("Fusion rows:", len(fusion_df))
    print("Tab feature dim:", X_tab.shape[1])
    print("HTML feature dim:", X_html.shape[1])

    # save raw fusion table
    fusion_df.to_csv(OUT_DIR / "phase10_fusion_input_table.csv", index=False)

    # LR baseline
    lr_oof, lr_metrics = run_lr_baseline(X_tab, X_html, y)

    # Gated MLP
    mlp_oof, mlp_metrics = run_gated_mlp(X_tab, X_html, y)

    # save OOF probs
    out = fusion_df[["row_index", "y_true"]].copy()
    out["lr_fusion_prob"] = lr_oof
    out["mlp_fusion_prob"] = mlp_oof
    out.to_csv(OUT_DIR / "phase10_fusion_oof_predictions.csv", index=False)

    with open(OUT_DIR / "phase10_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "lr_fusion": lr_metrics,
            "gated_mlp_fusion": mlp_metrics
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase10_fusion_input_table.csv")
    print(OUT_DIR / "phase10_fusion_oof_predictions.csv")
    print(OUT_DIR / "phase10_metrics_summary.json")


if __name__ == "__main__":
    main()