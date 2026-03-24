import json
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)

# ============================================================
# PATHS
# ============================================================
DATA_PATH = "final_multimodal_dataset.parquet"

FINAL_URL_PRED_PATH = "phase12_outputs/final_url_predictions.csv"
FINAL_URL_FEAT_PATH = "phase12_outputs/final_url_features.npy"

FINAL_HTML_PRED_PATH = "phase13_outputs/final_html_predictions.csv"
FINAL_HTML_FEAT_PATH = "phase13_outputs/final_html_features.npy"

OUT_DIR = Path("phase14_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
LR = 8e-4
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
        "Threshold": float(thr),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

# ============================================================
# LOAD + MERGE
# ============================================================
def load_all():
    data_df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    url_pred = pd.read_csv(FINAL_URL_PRED_PATH).sort_values("row_index").reset_index(drop=True)
    html_pred = pd.read_csv(FINAL_HTML_PRED_PATH).sort_values("row_index").reset_index(drop=True)

    url_feat = np.load(FINAL_URL_FEAT_PATH).astype(np.float32)
    html_feat = np.load(FINAL_HTML_FEAT_PATH).astype(np.float32)

    if len(data_df) != len(url_pred) or len(data_df) != len(html_pred):
        raise ValueError("Length mismatch")
    if len(data_df) != len(url_feat) or len(data_df) != len(html_feat):
        raise ValueError("Embedding length mismatch")

    y = (data_df["label"].astype(str).str.lower() == "phish").astype(int).values

    fusion_df = pd.DataFrame({
        "row_index": np.arange(len(data_df)),
        "url": data_df["url"].astype(str).values,
        "label": data_df["label"].astype(str).values,
        "y_true": y,
        "final_html_prob": html_pred["final_html_prob"].astype(float).values,
        "final_url_prob": url_pred["final_url_prob"].astype(float).values,
    })

    # use final URL prob as main URL signal and keep simple RF-like proxy if missing
    fusion_df["d_url_html"] = np.abs(fusion_df["final_url_prob"] - fusion_df["final_html_prob"])
    fusion_df["mean_prob"] = fusion_df[["final_url_prob", "final_html_prob"]].mean(axis=1)
    fusion_df["max_prob"] = fusion_df[["final_url_prob", "final_html_prob"]].max(axis=1)
    fusion_df["min_prob"] = fusion_df[["final_url_prob", "final_html_prob"]].min(axis=1)
    fusion_df["std_prob"] = fusion_df[["final_url_prob", "final_html_prob"]].std(axis=1)
    fusion_df["url_conf"] = np.abs(fusion_df["final_url_prob"] - 0.5)
    fusion_df["html_conf"] = np.abs(fusion_df["final_html_prob"] - 0.5)
    fusion_df["range_prob"] = fusion_df["max_prob"] - fusion_df["min_prob"]

    cross_cols = [c for c in data_df.columns if c.startswith("cross_")]
    cross_X = data_df[cross_cols].astype(np.float32).values if len(cross_cols) > 0 else None

    return fusion_df, url_feat, html_feat, cross_X

# ============================================================
# DATASET
# ============================================================
class FinalFusionDataset(Dataset):
    def __init__(self, x_tab, x_url, x_html, x_cross, y):
        self.x_tab = x_tab.astype(np.float32)
        self.x_url = x_url.astype(np.float32)
        self.x_html = x_html.astype(np.float32)
        self.x_cross = None if x_cross is None else x_cross.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.x_cross is None:
            xc = torch.zeros(1, dtype=torch.float32)
        else:
            xc = torch.tensor(self.x_cross[i], dtype=torch.float32)

        return (
            torch.tensor(self.x_tab[i], dtype=torch.float32),
            torch.tensor(self.x_url[i], dtype=torch.float32),
            torch.tensor(self.x_html[i], dtype=torch.float32),
            xc,
            torch.tensor(self.y[i], dtype=torch.float32)
        )

# ============================================================
# MODEL
# ============================================================
class FusionV3(nn.Module):
    def __init__(self, tab_dim, url_dim, html_dim, cross_dim):
        super().__init__()

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.url_proj = nn.Sequential(
            nn.Linear(url_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.html_proj = nn.Sequential(
            nn.Linear(html_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.cross_proj = nn.Sequential(
            nn.Linear(cross_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) if cross_dim > 0 else None

        self.interaction = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.url_to_html = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.html_to_url = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.gate = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        in_dim = 32 + 64 + 64 + 64 + (32 if cross_dim > 0 else 0)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_url, x_html, x_cross):
        t = self.tab_proj(x_tab)
        u = self.url_proj(x_url)
        h = self.html_proj(x_html)

        inter = torch.cat([u, h, torch.abs(u - h), u * h], dim=1)
        inter = self.interaction(inter)

        u_new = u + self.html_to_url(inter)
        h_new = h + self.url_to_html(inter)

        gate = self.gate(torch.cat([u_new, h_new], dim=1))
        fused = gate * u_new + (1.0 - gate) * h_new

        parts = [t, u_new, h_new, fused]
        if self.cross_proj is not None:
            parts.append(self.cross_proj(x_cross))

        z = torch.cat(parts, dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

# ============================================================
# TRAIN / PREDICT
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x_tab, x_url, x_html, x_cross, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_url = x_url.to(DEVICE)
        x_html = x_html.to(DEVICE)
        x_cross = x_cross.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tab, x_url, x_html, x_cross)
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

    for x_tab, x_url, x_html, x_cross, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_url = x_url.to(DEVICE)
        x_html = x_html.to(DEVICE)
        x_cross = x_cross.to(DEVICE)

        logits = model(x_tab, x_url, x_html, x_cross)
        prob = torch.sigmoid(logits).cpu().numpy()

        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())

    return np.array(trues), np.array(probs)

# ============================================================
# MAIN
# ============================================================
def main():
    print("Using device:", DEVICE)

    fusion_df, url_feat, html_feat, cross_X = load_all()
    y = fusion_df["y_true"].astype(int).values

    tab_cols = [
        "final_html_prob",
        "final_url_prob",
        "d_url_html",
        "mean_prob",
        "max_prob",
        "min_prob",
        "std_prob",
        "url_conf",
        "html_conf",
        "range_prob",
    ]
    tab_X = fusion_df[tab_cols].values.astype(np.float32)

    if cross_X is None:
        cross_dim = 0
        cross_X = np.zeros((len(y), 0), dtype=np.float32)
    else:
        cross_dim = cross_X.shape[1]

    print("Rows:", len(y))
    print("Tab dim:", tab_X.shape[1])
    print("URL dim:", url_feat.shape[1])
    print("HTML dim:", html_feat.shape[1])
    print("Cross dim:", cross_dim)

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    tr_inner, val_idx = train_test_split(
        tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=RANDOM_SEED
    )

    scaler_tab = StandardScaler()
    scaler_url = StandardScaler()
    scaler_html = StandardScaler()
    scaler_cross = StandardScaler() if cross_dim > 0 else None

    Xtab_tr = scaler_tab.fit_transform(tab_X[tr_inner])
    Xtab_val = scaler_tab.transform(tab_X[val_idx])
    Xtab_te = scaler_tab.transform(tab_X[te_idx])

    Xurl_tr = scaler_url.fit_transform(url_feat[tr_inner])
    Xurl_val = scaler_url.transform(url_feat[val_idx])
    Xurl_te = scaler_url.transform(url_feat[te_idx])

    Xhtml_tr = scaler_html.fit_transform(html_feat[tr_inner])
    Xhtml_val = scaler_html.transform(html_feat[val_idx])
    Xhtml_te = scaler_html.transform(html_feat[te_idx])

    if cross_dim > 0:
        Xcross_tr = scaler_cross.fit_transform(cross_X[tr_inner])
        Xcross_val = scaler_cross.transform(cross_X[val_idx])
        Xcross_te = scaler_cross.transform(cross_X[te_idx])
    else:
        Xcross_tr = np.zeros((len(tr_inner), 0), dtype=np.float32)
        Xcross_val = np.zeros((len(val_idx), 0), dtype=np.float32)
        Xcross_te = np.zeros((len(te_idx), 0), dtype=np.float32)

    ds_train = FinalFusionDataset(Xtab_tr, Xurl_tr, Xhtml_tr, Xcross_tr, y[tr_inner])
    ds_val = FinalFusionDataset(Xtab_val, Xurl_val, Xhtml_val, Xcross_val, y[val_idx])
    ds_test = FinalFusionDataset(Xtab_te, Xurl_te, Xhtml_te, Xcross_te, y[te_idx])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionV3(
        tab_dim=tab_X.shape[1],
        url_dim=url_feat.shape[1],
        html_dim=html_feat.shape[1],
        cross_dim=cross_dim
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

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_pr_auc={val_prauc:.4f}")

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    y_test_true, y_test_prob = predict_probs(model, dl_test)
    metrics = compute_metrics(y_test_true, y_test_prob, thr=0.5)

    print("\nFINAL PHASE 14 TEST METRICS")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<18} {v:.4f}")
        else:
            print(f"{k:<18} {v}")

    out = fusion_df.iloc[te_idx][["row_index", "url", "label", "y_true"]].copy()
    out["phase14_prob"] = y_test_prob
    out["phase14_pred"] = (y_test_prob >= 0.5).astype(int)
    out.to_csv(OUT_DIR / "phase14_test_predictions.csv", index=False)

    fps = out[(out["y_true"] == 0) & (out["phase14_pred"] == 1)].copy()
    fns = out[(out["y_true"] == 1) & (out["phase14_pred"] == 0)].copy()

    fps.to_csv(OUT_DIR / "phase14_false_positives.csv", index=False)
    fns.to_csv(OUT_DIR / "phase14_false_negatives.csv", index=False)

    with open(OUT_DIR / "phase14_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase14_test_predictions.csv")
    print(OUT_DIR / "phase14_false_positives.csv")
    print(OUT_DIR / "phase14_false_negatives.csv")
    print(OUT_DIR / "phase14_metrics.json")

if __name__ == "__main__":
    main()