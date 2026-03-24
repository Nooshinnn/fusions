import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)

PRED_PATH = "phase31_fusion_cv_with_domain_intel_outputs/phase31_domain_intel_fusion_oof_predictions.csv"
OUT_DIR = Path("phase22_results")
OUT_DIR.mkdir(exist_ok=True)

THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)
PRECISION_FLOOR = 0.90

def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
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
        "Log Loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7), labels=[0,1]),
        "Threshold": float(thr),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

def main():
    df = pd.read_csv(PRED_PATH)
    y_true = df["y_true"].astype(int).values
    y_prob = df["fusion_prob"].astype(float).values

    rows = [compute_metrics(y_true, y_prob, t) for t in THRESHOLDS]
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(OUT_DIR / "phase22_threshold_sweep.csv", index=False)

    best_mcc = sweep_df.sort_values(["MCC", "F1", "Recall"], ascending=[False, False, False]).iloc[0]
    best_f1 = sweep_df.sort_values(["F1", "MCC", "Recall"], ascending=[False, False, False]).iloc[0]

    filtered = sweep_df[sweep_df["Precision"] >= PRECISION_FLOOR].copy()
    best_low_fn = None if len(filtered) == 0 else filtered.sort_values(
        ["FN", "MCC", "F1"], ascending=[True, False, False]
    ).iloc[0]

    best_thr = float(best_mcc["Threshold"])
    df["fusion_pred_best"] = (df["fusion_prob"] >= best_thr).astype(int)

    fps = df[(df["y_true"] == 0) & (df["fusion_pred_best"] == 1)].copy()
    fns = df[(df["y_true"] == 1) & (df["fusion_pred_best"] == 0)].copy()

    fps.to_csv(OUT_DIR / "phase22_false_positives.csv", index=False)
    fns.to_csv(OUT_DIR / "phase22_false_negatives.csv", index=False)

    print("\nBest threshold by MCC")
    print(best_mcc.to_string())

    print("\nBest threshold by F1")
    print(best_f1.to_string())

    if best_low_fn is not None:
        print(f"\nBest threshold by lowest FN with precision >= {PRECISION_FLOOR:.2f}")
        print(best_low_fn.to_string())

    print("\nFive FN samples:")
    print(fns.head(5).to_string(index=False))

    summary = {
        "best_threshold_by_mcc": best_mcc.to_dict(),
        "best_threshold_by_f1": best_f1.to_dict(),
        "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
    }

    with open(OUT_DIR / "phase22_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase22_threshold_sweep.csv")
    print(OUT_DIR / "phase22_false_positives.csv")
    print(OUT_DIR / "phase22_false_negatives.csv")
    print(OUT_DIR / "phase22_summary.json")

if __name__ == "__main__":
    main()