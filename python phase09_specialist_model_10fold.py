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
PHASE06_OOF_PATH = Path("phase06_multichannel_joint_outputs/phase06_multichannel_oof_predictions.csv")
OUT_DIR = Path("phase09_specialist_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 25
PATIENCE = 5
LR = 8e-4
WEIGHT_DECAY = 1e-4

MAIN_THRESHOLD = 0.60     # best MCC threshold from Phase 6
UNCERTAIN_LOW = 0.20
UNCERTAIN_HIGH = 0.80
SPECIALIST_POS_WEIGHT_MULT = 3.0
SPECIALIST_UNCERTAIN_WEIGHT = 2.0
SPECIALIST_ERROR_WEIGHT = 4.0

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
# TAB FEATURES
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

    df["html_vs_url_conflict"] = ((df["html_prob"] > 0.7) & (df["url_textcnn_prob"] < 0.3)).astype(float)
    df["html_vs_rf_conflict"] = ((df["html_prob"] > 0.7) & (df["url_rf_prob"] < 0.3)).astype(float)

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
class SpecialistDataset(Dataset):
    def __init__(self, X, y, sample_weights):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = sample_weights.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
            torch.tensor(self.w[i], dtype=torch.float32),
        )

# ============================================================
# MODEL
# ============================================================
class SpecialistMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ============================================================
# TRAIN / PREDICT
# ============================================================
def weighted_bce_loss(logits, targets, weights, pos_weight=None):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    return (loss * weights).mean()

def train_epoch(model, loader, optimizer, pos_weight):
    model.train()
    total = 0.0
    for X, y, w in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        w = w.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = weighted_bce_loss(logits, y, w, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, trues = [], []
    for X, y, _ in loader:
        X = X.to(DEVICE)
        logits = model(X)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(probs)

# ============================================================
# HARD-CASE WEIGHTS
# ============================================================
def build_specialist_weights(main_prob, y_true, main_thr):
    main_pred = (main_prob >= main_thr).astype(int)

    is_error = (main_pred != y_true).astype(np.float32)
    is_uncertain = ((main_prob >= UNCERTAIN_LOW) & (main_prob <= UNCERTAIN_HIGH)).astype(np.float32)
    is_positive = (y_true == 1).astype(np.float32)

    weights = np.ones_like(main_prob, dtype=np.float32)
    weights += is_error * (SPECIALIST_ERROR_WEIGHT - 1.0)
    weights += is_uncertain * (SPECIALIST_UNCERTAIN_WEIGHT - 1.0)
    weights += is_positive * (SPECIALIST_POS_WEIGHT_MULT - 1.0)

    return weights

# ============================================================
# MAIN
# ============================================================
def main():
    phase06_oof = pd.read_csv(PHASE06_OOF_PATH).sort_values("row_index").reset_index(drop=True)

    all_preds = []
    fold_metrics = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 09 SPECIALIST FOLD {fold}/{N_SPLITS}")
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

        # attach main model probs from phase 6 OOF for training rows
        train_main = phase06_oof[phase06_oof["row_index"].isin(train_df["row_index"])].sort_values("row_index").reset_index(drop=True)
        test_main = phase06_oof[phase06_oof["row_index"].isin(test_df["row_index"])].sort_values("row_index").reset_index(drop=True)

        if not np.array_equal(train_main["row_index"].values, train_df["row_index"].values):
            raise ValueError(f"Train row_index mismatch on fold {fold}")
        if not np.array_equal(test_main["row_index"].values, test_df["row_index"].values):
            raise ValueError(f"Test row_index mismatch on fold {fold}")

        train_df["main_prob"] = train_main["multichannel_prob"].astype(float).values
        test_df["main_prob"] = test_main["multichannel_prob"].astype(float).values

        # richer specialist features
        X_train = np.concatenate([
            train_df[TAB_COLS + ["main_prob"]].values.astype(np.float32),
            train_url_feat,
            train_html_feat,
        ], axis=1)

        X_test = np.concatenate([
            test_df[TAB_COLS + ["main_prob"]].values.astype(np.float32),
            test_url_feat,
            test_html_feat,
        ], axis=1)

        y_train = train_df["y_true"].astype(int).values
        y_test = test_df["y_true"].astype(int).values
        main_prob_train = train_df["main_prob"].values.astype(np.float32)
        main_prob_test = test_df["main_prob"].values.astype(np.float32)

        sample_weights = build_specialist_weights(main_prob_train, y_train, MAIN_THRESHOLD)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(
            idx, test_size=0.1, stratify=y_train, random_state=SEED
        )

        ds_train = SpecialistDataset(X_train[tr_idx], y_train[tr_idx], sample_weights[tr_idx])
        ds_val = SpecialistDataset(X_train[val_idx], y_train[val_idx], np.ones_like(y_train[val_idx], dtype=np.float32))
        ds_test = SpecialistDataset(X_test, y_test, np.ones_like(y_test, dtype=np.float32))

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = SpecialistMLP(in_dim=X_train.shape[1]).to(DEVICE)

        n_neg = int((y_train[tr_idx] == 0).sum())
        n_pos = int((y_train[tr_idx] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_state = None
        best_val_prauc = -1.0
        wait = 0

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, dl_train, optimizer, pos_weight)
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
        y_test_true, specialist_prob = predict_probs(model, dl_test)

        # routing: use specialist only in uncertainty band, otherwise keep main
        final_prob = main_prob_test.copy()
        mask_uncertain = (main_prob_test >= UNCERTAIN_LOW) & (main_prob_test <= UNCERTAIN_HIGH)
        final_prob[mask_uncertain] = specialist_prob[mask_uncertain]

        metrics = compute_metrics(y_test_true, final_prob, thr=MAIN_THRESHOLD)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

        pred_df = test_df[["row_index", "url", "y_true"]].copy()
        pred_df["main_prob"] = main_prob_test
        pred_df["specialist_prob"] = specialist_prob
        pred_df["used_specialist"] = mask_uncertain.astype(int)
        pred_df["final_prob"] = final_prob
        pred_df["final_pred"] = (final_prob >= MAIN_THRESHOLD).astype(int)
        pred_df["fold"] = fold
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"UsedSpecialist={mask_uncertain.mean():.3f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase09_specialist_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase09_specialist_fold_metrics.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase09_specialist_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 09 SPECIALIST SUMMARY")
    print("=" * 70)
    print(summary)

    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["final_prob"].astype(float).values
    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase09_specialist_threshold_sweep.csv", index=False)

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

    with open(OUT_DIR / "phase09_specialist_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase09_specialist_oof_predictions.csv")
    print(OUT_DIR / "phase09_specialist_fold_metrics.csv")
    print(OUT_DIR / "phase09_specialist_summary.csv")
    print(OUT_DIR / "phase09_specialist_threshold_sweep.csv")

if __name__ == "__main__":
    main()
