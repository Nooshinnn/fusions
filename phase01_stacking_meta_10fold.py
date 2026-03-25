import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR = Path("phase01_stacking_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLP_BATCH_SIZE = 128
MLP_EPOCHS = 30
MLP_PATIENCE = 5
MLP_LR = 1e-3
MLP_WD = 1e-4

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
# META FEATURES
# ============================================================
def build_meta_features(url_df: pd.DataFrame, html_df: pd.DataFrame) -> pd.DataFrame:
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

    df["all_agree_high"] = (
        (df["html_prob"] > 0.7) &
        (df["url_textcnn_prob"] > 0.7) &
        (df["url_rf_prob"] > 0.7)
    ).astype(float)

    df["all_agree_low"] = (
        (df["html_prob"] < 0.3) &
        (df["url_textcnn_prob"] < 0.3) &
        (df["url_rf_prob"] < 0.3)
    ).astype(float)

    df["html_vs_url_conflict"] = (
        (df["html_prob"] > 0.7) & (df["url_textcnn_prob"] < 0.3)
    ).astype(float)

    df["html_vs_rf_conflict"] = (
        (df["html_prob"] > 0.7) & (df["url_rf_prob"] < 0.3)
    ).astype(float)

    return df

META_COLS = [
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
    "all_agree_high",
    "all_agree_low",
    "html_vs_url_conflict",
    "html_vs_rf_conflict",
]

# ============================================================
# MLP META LEARNER
# ============================================================
class MetaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )

class MetaMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_probs_mlp(model, loader):
    model.eval()
    probs, trues = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        logits = model(X)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(probs)

# ============================================================
# MAIN
# ============================================================
def main():
    all_lr_preds = []
    all_rf_preds = []
    all_mlp_preds = []

    lr_fold_metrics = []
    rf_fold_metrics = []
    mlp_fold_metrics = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 01 STACKING FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_df = build_meta_features(train_url, train_html)
        test_df = build_meta_features(test_url, test_html)

        X_train = train_df[META_COLS].values.astype(np.float32)
        X_test = test_df[META_COLS].values.astype(np.float32)
        y_train = train_df["y_true"].astype(int).values
        y_test = test_df["y_true"].astype(int).values

        # shared scaler
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # ---------------- LR ----------------
        lr = LogisticRegression(
            max_iter=2000,
            random_state=SEED
        )
        lr.fit(X_train_sc, y_train)
        lr_prob = lr.predict_proba(X_test_sc)[:, 1]
        lr_metrics = compute_metrics(y_test, lr_prob, thr=0.5)
        lr_metrics["Fold"] = fold
        lr_fold_metrics.append(lr_metrics)

        lr_pred_df = test_df[["row_index", "url", "y_true"]].copy()
        lr_pred_df["stack_prob"] = lr_prob
        lr_pred_df["stack_pred"] = (lr_prob >= 0.5).astype(int)
        lr_pred_df["fold"] = fold
        lr_pred_df["model"] = "logreg"
        all_lr_preds.append(lr_pred_df)

        # ---------------- RF ----------------
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            max_features="sqrt",
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=SEED,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_metrics = compute_metrics(y_test, rf_prob, thr=0.5)
        rf_metrics["Fold"] = fold
        rf_fold_metrics.append(rf_metrics)

        rf_pred_df = test_df[["row_index", "url", "y_true"]].copy()
        rf_pred_df["stack_prob"] = rf_prob
        rf_pred_df["stack_pred"] = (rf_prob >= 0.5).astype(int)
        rf_pred_df["fold"] = fold
        rf_pred_df["model"] = "random_forest"
        all_rf_preds.append(rf_pred_df)

        # ---------------- MLP ----------------
        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(
            idx, test_size=0.1, stratify=y_train, random_state=SEED
        )

        ds_train = MetaDataset(X_train_sc[tr_idx], y_train[tr_idx])
        ds_val = MetaDataset(X_train_sc[val_idx], y_train[val_idx])
        ds_test = MetaDataset(X_test_sc, y_test)

        dl_train = DataLoader(ds_train, batch_size=MLP_BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=MLP_BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=MLP_BATCH_SIZE, shuffle=False)

        mlp = MetaMLP(in_dim=X_train_sc.shape[1]).to(DEVICE)

        n_neg = int((y_train[tr_idx] == 0).sum())
        n_pos = int((y_train[tr_idx] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

        optimizer = torch.optim.AdamW(mlp.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_prauc = -1.0
        wait = 0

        for epoch in range(1, MLP_EPOCHS + 1):
            loss = train_epoch(mlp, dl_train, optimizer, criterion)
            y_val_true, y_val_prob = predict_probs_mlp(mlp, dl_val)
            prauc = average_precision_score(y_val_true, y_val_prob)
            print(f"[Fold {fold}] MLP epoch {epoch:02d} | loss={loss:.4f} | val_pr_auc={prauc:.4f}")

            if prauc > best_val_prauc:
                best_val_prauc = prauc
                best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= MLP_PATIENCE:
                    print(f"[Fold {fold}] MLP early stop at epoch {epoch}")
                    break

        mlp.load_state_dict(best_state)
        y_test_true, mlp_prob = predict_probs_mlp(mlp, dl_test)
        mlp_metrics = compute_metrics(y_test_true, mlp_prob, thr=0.5)
        mlp_metrics["Fold"] = fold
        mlp_fold_metrics.append(mlp_metrics)

        mlp_pred_df = test_df[["row_index", "url", "y_true"]].copy()
        mlp_pred_df["stack_prob"] = mlp_prob
        mlp_pred_df["stack_pred"] = (mlp_prob >= 0.5).astype(int)
        mlp_pred_df["fold"] = fold
        mlp_pred_df["model"] = "mlp"
        all_mlp_preds.append(mlp_pred_df)

        print(
            f"[Fold {fold}] "
            f"LR_MCC={lr_metrics['MCC']:.4f} | "
            f"RF_MCC={rf_metrics['MCC']:.4f} | "
            f"MLP_MCC={mlp_metrics['MCC']:.4f}"
        )

    # save fold predictions
    lr_oof = pd.concat(all_lr_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    rf_oof = pd.concat(all_rf_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    mlp_oof = pd.concat(all_mlp_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)

    lr_oof.to_csv(OUT_DIR / "phase01_logreg_oof_predictions.csv", index=False)
    rf_oof.to_csv(OUT_DIR / "phase01_rf_oof_predictions.csv", index=False)
    mlp_oof.to_csv(OUT_DIR / "phase01_mlp_oof_predictions.csv", index=False)

    # save fold metrics
    lr_metrics_df = pd.DataFrame(lr_fold_metrics)
    rf_metrics_df = pd.DataFrame(rf_fold_metrics)
    mlp_metrics_df = pd.DataFrame(mlp_fold_metrics)

    lr_metrics_df.to_csv(OUT_DIR / "phase01_logreg_fold_metrics.csv", index=False)
    rf_metrics_df.to_csv(OUT_DIR / "phase01_rf_fold_metrics.csv", index=False)
    mlp_metrics_df.to_csv(OUT_DIR / "phase01_mlp_fold_metrics.csv", index=False)

    # summary
    lr_summary = lr_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    rf_summary = rf_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    mlp_summary = mlp_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T

    with pd.ExcelWriter(OUT_DIR / "phase01_stacking_summary.xlsx") as writer:
        lr_summary.to_excel(writer, sheet_name="logreg_summary")
        rf_summary.to_excel(writer, sheet_name="rf_summary")
        mlp_summary.to_excel(writer, sheet_name="mlp_summary")

    print("\n" + "=" * 70)
    print("PHASE 01 STACKING SUMMARY — LOGREG")
    print("=" * 70)
    print(lr_summary)

    print("\n" + "=" * 70)
    print("PHASE 01 STACKING SUMMARY — RANDOM FOREST")
    print("=" * 70)
    print(rf_summary)

    print("\n" + "=" * 70)
    print("PHASE 01 STACKING SUMMARY — MLP")
    print("=" * 70)
    print(mlp_summary)

    # threshold sweeps on OOF predictions
    for name, df in [
        ("logreg", lr_oof),
        ("random_forest", rf_oof),
        ("mlp", mlp_oof),
    ]:
        y_true = df["y_true"].astype(int).values
        y_prob = df["stack_prob"].astype(float).values

        sweep = build_threshold_sweep(y_true, y_prob)
        sweep.to_csv(OUT_DIR / f"phase01_{name}_threshold_sweep.csv", index=False)

        best_mcc = sweep.sort_values(["MCC", "F1", "Recall"], ascending=[False, False, False]).iloc[0]
        best_f1 = sweep.sort_values(["F1", "MCC", "Recall"], ascending=[False, False, False]).iloc[0]
        filtered = sweep[sweep["Precision"] >= PRECISION_FLOOR].copy()
        best_low_fn = None if len(filtered) == 0 else filtered.sort_values(
            ["FN", "MCC", "F1"], ascending=[True, False, False]
        ).iloc[0]

        out = {
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }
        with open(OUT_DIR / f"phase01_{name}_threshold_summary.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase01_logreg_oof_predictions.csv")
    print(OUT_DIR / "phase01_rf_oof_predictions.csv")
    print(OUT_DIR / "phase01_mlp_oof_predictions.csv")
    print(OUT_DIR / "phase01_stacking_summary.xlsx")

if __name__ == "__main__":
    main()