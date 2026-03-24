import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)

# ============================================================
# PATHS
# ============================================================
HTML_PRED_PATH = "phase7_outputs/phase7_all_predictions.csv"
HTML_FEAT_PATH = "phase7_outputs/oof_html_phase7_features.npy"

URL_OOF_PATH = "domurlbert_textcnn_rf_url_only_oof_predictions.csv"
URL_FEAT_PATH = "oof_url_textcnn_features.npy"

OUT_DIR = Path("phase11_lite_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
N_SPLITS = 10
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
LR = 8e-4
WEIGHT_DECAY = 1e-4

THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)
PRECISION_FLOOR = 0.90

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

def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t

def threshold_sweep_table(y_true, y_prob):
    rows = []
    for t in THRESHOLDS:
        rows.append(compute_metrics(y_true, y_prob, thr=t))
    return pd.DataFrame(rows)

# ============================================================
# LOAD + MERGE
# ============================================================
def load_and_merge():
    html_df = pd.read_csv(HTML_PRED_PATH).sort_values("row_index").reset_index(drop=True)
    url_df = pd.read_csv(URL_OOF_PATH).sort_values("row_index").reset_index(drop=True)

    html_feats = np.load(HTML_FEAT_PATH).astype(np.float32)
    url_feats = np.load(URL_FEAT_PATH).astype(np.float32)

    required_html = {"row_index", "url", "label", "y_true", "y_prob"}
    required_url = {"row_index", "url", "y_true", "textcnn_prob", "rf_prob"}

    if not required_html.issubset(html_df.columns):
        raise ValueError(f"HTML file missing columns: {required_html - set(html_df.columns)}")
    if not required_url.issubset(url_df.columns):
        raise ValueError(f"URL file missing columns: {required_url - set(url_df.columns)}")

    if len(html_df) != len(url_df):
        raise ValueError("HTML and URL prediction tables have different lengths")
    if len(html_df) != len(html_feats) or len(html_df) != len(url_feats):
        raise ValueError("Prediction/feature length mismatch")

    if not np.array_equal(html_df["row_index"].values, url_df["row_index"].values):
        raise ValueError("row_index mismatch")
    if not np.array_equal(html_df["url"].astype(str).values, url_df["url"].astype(str).values):
        raise ValueError("URL mismatch")
    if not np.array_equal(html_df["y_true"].values, url_df["y_true"].values):
        raise ValueError("y_true mismatch")

    fusion_df = pd.DataFrame({
        "row_index": html_df["row_index"].values,
        "url": html_df["url"].values,
        "label": html_df["label"].values,
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

    return fusion_df, html_feats, url_feats

# ============================================================
# DATASET
# ============================================================
class FusionDataset(Dataset):
    def __init__(self, x_tab, x_url, x_html, y):
        self.x_tab = x_tab.astype(np.float32)
        self.x_url = x_url.astype(np.float32)
        self.x_html = x_html.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x_tab[i], dtype=torch.float32),
            torch.tensor(self.x_url[i], dtype=torch.float32),
            torch.tensor(self.x_html[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32)
        )

# ============================================================
# MODEL
# ============================================================
class FusionLite(nn.Module):
    def __init__(self, tab_dim, url_dim, html_dim):
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

        # simpler, safer interaction than broken Phase 11
        self.interaction = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_url, x_html):
        t = self.tab_proj(x_tab)
        u = self.url_proj(x_url)
        h = self.html_proj(x_html)

        inter = torch.cat([u, h, torch.abs(u - h), u * h], dim=1)
        inter = self.interaction(inter)

        z = torch.cat([t, u, h, inter], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

# ============================================================
# TRAIN / PREDICT
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

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

        total_loss += loss.item() * x_tab.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, trues = [], []

    for x_tab, x_url, x_html, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_url = x_url.to(DEVICE)
        x_html = x_html.to(DEVICE)

        logits = model(x_tab, x_url, x_html)
        prob = torch.sigmoid(logits).cpu().numpy()

        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())

    return np.array(trues), np.array(probs)

# ============================================================
# MAIN
# ============================================================
def main():
    print("Using device:", DEVICE)

    fusion_df, html_X, url_X = load_and_merge()
    y = fusion_df["y_true"].astype(int).values

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
    tab_X = fusion_df[tab_cols].values.astype(np.float32)

    print("Fusion rows:", len(y))
    print("Tab feature dim:", tab_X.shape[1])
    print("URL feature dim:", url_X.shape[1])
    print("HTML feature dim:", html_X.shape[1])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(tab_X, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}")

        scaler_tab = StandardScaler()
        scaler_url = StandardScaler()
        scaler_html = StandardScaler()

        Xtab_tr = scaler_tab.fit_transform(tab_X[tr_idx])
        Xtab_te = scaler_tab.transform(tab_X[te_idx])

        Xurl_tr = scaler_url.fit_transform(url_X[tr_idx])
        Xurl_te = scaler_url.transform(url_X[te_idx])

        Xhtml_tr = scaler_html.fit_transform(html_X[tr_idx])
        Xhtml_te = scaler_html.transform(html_X[te_idx])

        inner_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=RANDOM_SEED)
        inner_train_idx, val_idx = next(inner_skf.split(Xtab_tr, y[tr_idx]))

        ds_train = FusionDataset(
            Xtab_tr[inner_train_idx],
            Xurl_tr[inner_train_idx],
            Xhtml_tr[inner_train_idx],
            y[tr_idx][inner_train_idx]
        )
        ds_val = FusionDataset(
            Xtab_tr[val_idx],
            Xurl_tr[val_idx],
            Xhtml_tr[val_idx],
            y[tr_idx][val_idx]
        )
        ds_test = FusionDataset(
            Xtab_te,
            Xurl_te,
            Xhtml_te,
            y[te_idx]
        )

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = FusionLite(
            tab_dim=tab_X.shape[1],
            url_dim=url_X.shape[1],
            html_dim=html_X.shape[1]
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

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    best_thr = find_best_threshold(y, oof_prob)
    final_metrics = compute_metrics(y, oof_prob, thr=best_thr)

    print("\n" + "=" * 70)
    print("FINAL PHASE 11-LITE METRICS")
    print("=" * 70)
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k:<18} {v:.4f}")
        else:
            print(f"{k:<18} {v}")

    out_df = fusion_df.copy()
    out_df["phase11_lite_prob"] = oof_prob
    out_df["phase11_lite_pred"] = (oof_prob >= best_thr).astype(int)
    out_df.to_csv(OUT_DIR / "phase11_lite_oof_predictions.csv", index=False)

    fps = out_df[(out_df["y_true"] == 0) & (out_df["phase11_lite_pred"] == 1)].copy()
    fns = out_df[(out_df["y_true"] == 1) & (out_df["phase11_lite_pred"] == 0)].copy()

    fps.to_csv(OUT_DIR / "phase11_lite_false_positives.csv", index=False)
    fns.to_csv(OUT_DIR / "phase11_lite_false_negatives.csv", index=False)

    sweep_df = threshold_sweep_table(y, oof_prob)
    sweep_df.to_csv(OUT_DIR / "phase11_lite_threshold_sweep.csv", index=False)

    best_mcc = sweep_df.sort_values(["MCC", "F1", "Recall"], ascending=[False, False, False]).iloc[0]
    best_f1 = sweep_df.sort_values(["F1", "MCC", "Recall"], ascending=[False, False, False]).iloc[0]
    filtered = sweep_df[sweep_df["Precision"] >= PRECISION_FLOOR].copy()
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

    print("\nFive false negative samples:")
    preview_cols = ["url", "label", "y_true", "phase11_lite_prob", "phase11_lite_pred"]
    print(fns[preview_cols].head(5).to_string(index=False))

    with open(OUT_DIR / "phase11_lite_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_metrics": final_metrics,
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
            "fusion_rows": int(len(y)),
            "tab_feature_dim": int(tab_X.shape[1]),
            "url_feature_dim": int(url_X.shape[1]),
            "html_feature_dim": int(html_X.shape[1]),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase11_lite_oof_predictions.csv")
    print(OUT_DIR / "phase11_lite_false_positives.csv")
    print(OUT_DIR / "phase11_lite_false_negatives.csv")
    print(OUT_DIR / "phase11_lite_threshold_sweep.csv")
    print(OUT_DIR / "phase11_lite_metrics.json")

if __name__ == "__main__":
    main()