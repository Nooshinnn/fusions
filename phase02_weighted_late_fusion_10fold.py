import json
import itertools
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR = Path("phase02_weighted_late_fusion_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)
PRECISION_FLOOR = 0.90

# weight grid
GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

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
# WEIGHT SEARCH
# ============================================================
def weight_candidates():
    cands = []
    for w_html in GRID:
        for w_url in GRID:
            for w_rf in GRID:
                s = w_html + w_url + w_rf
                if abs(s - 1.0) < 1e-9:
                    cands.append((float(w_html), float(w_url), float(w_rf)))
    return cands

def weighted_prob(df, weights):
    w_html, w_url, w_rf = weights
    return (
        w_html * df["html_prob"].values +
        w_url * df["url_textcnn_prob"].values +
        w_rf * df["url_rf_prob"].values
    )

def find_best_weights(train_df, metric="MCC"):
    y = train_df["y_true"].astype(int).values
    best = None
    best_score = -1e18

    for weights in weight_candidates():
        p = weighted_prob(train_df, weights)
        m = compute_metrics(y, p, thr=0.5)
        score = m[metric]

        # tie-break by F1 then Recall
        tie_key = (score, m["F1"], m["Recall"])
        if best is None or tie_key > best_score:
            best = (weights, m)
            best_score = tie_key

    return best

# ============================================================
# MAIN
# ============================================================
def main():
    all_preds = []
    fold_metrics = []
    chosen_weights = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 02 WEIGHTED LATE FUSION FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_df = pd.DataFrame({
            "row_index": train_html["row_index"].values,
            "url": train_html["url"].values,
            "y_true": train_html["y_true"].astype(int).values,
            "html_prob": train_html["html_prob"].astype(float).values,
            "url_textcnn_prob": train_url["url_textcnn_prob"].astype(float).values,
            "url_rf_prob": train_url["url_rf_prob"].astype(float).values,
        })

        test_df = pd.DataFrame({
            "row_index": test_html["row_index"].values,
            "url": test_html["url"].values,
            "y_true": test_html["y_true"].astype(int).values,
            "html_prob": test_html["html_prob"].astype(float).values,
            "url_textcnn_prob": test_url["url_textcnn_prob"].astype(float).values,
            "url_rf_prob": test_url["url_rf_prob"].astype(float).values,
        })

        best_weights, train_metrics = find_best_weights(train_df, metric="MCC")
        chosen_weights.append({
            "Fold": fold,
            "w_html": best_weights[0],
            "w_url_textcnn": best_weights[1],
            "w_url_rf": best_weights[2],
            "train_mcc_at_0.5": train_metrics["MCC"],
            "train_f1_at_0.5": train_metrics["F1"],
        })

        test_prob = weighted_prob(test_df, best_weights)
        test_metrics = compute_metrics(test_df["y_true"].values, test_prob, thr=0.5)
        test_metrics["Fold"] = fold
        test_metrics["w_html"] = best_weights[0]
        test_metrics["w_url_textcnn"] = best_weights[1]
        test_metrics["w_url_rf"] = best_weights[2]
        fold_metrics.append(test_metrics)

        pred_df = test_df[["row_index", "url", "y_true"]].copy()
        pred_df["weighted_prob"] = test_prob
        pred_df["weighted_pred"] = (test_prob >= 0.5).astype(int)
        pred_df["fold"] = fold
        pred_df["w_html"] = best_weights[0]
        pred_df["w_url_textcnn"] = best_weights[1]
        pred_df["w_url_rf"] = best_weights[2]
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"weights={best_weights} | "
            f"Acc={test_metrics['Accuracy']:.4f} | "
            f"F1={test_metrics['F1']:.4f} | "
            f"MCC={test_metrics['MCC']:.4f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase02_weighted_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase02_weighted_fold_metrics.csv", index=False)

    weights_df = pd.DataFrame(chosen_weights)
    weights_df.to_csv(OUT_DIR / "phase02_chosen_weights.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase02_weighted_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 02 WEIGHTED LATE FUSION SUMMARY")
    print("=" * 70)
    print(summary)

    # threshold sweep on OOF predictions
    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["weighted_prob"].astype(float).values

    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase02_weighted_threshold_sweep.csv", index=False)

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

    with open(OUT_DIR / "phase02_weighted_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase02_weighted_oof_predictions.csv")
    print(OUT_DIR / "phase02_weighted_fold_metrics.csv")
    print(OUT_DIR / "phase02_chosen_weights.csv")
    print(OUT_DIR / "phase02_weighted_summary.csv")
    print(OUT_DIR / "phase02_weighted_threshold_sweep.csv")

if __name__ == "__main__":
    main()
