import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "final_multimodal_dataset.parquet"
PHASE12_OOF_PATH = "phase12_error_focused_outputs/phase12_error_focused_oof_predictions.csv"

OUT_DIR = Path("phase13_routing_calibration_outputs")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10

# routing thresholds
LOW_LOCK = 0.15
HIGH_LOCK = 0.85

# default operating threshold after calibration
DEFAULT_THRESHOLD = 0.50

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
# CALIBRATION + ROUTING
# ============================================================
def fit_platt_scaler(train_prob, y_train):
    clf = LogisticRegression(
        max_iter=2000,
        random_state=SEED
    )
    clf.fit(train_prob.reshape(-1, 1), y_train)
    return clf

def apply_calibration(calibrator, probs):
    return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

def apply_strict_routing(calibrated_prob, low_lock=0.15, high_lock=0.85):
    """
    Lock very confident predictions.
    Middle region passes through calibrated score unchanged.
    """
    final = calibrated_prob.copy()

    low_mask = calibrated_prob <= low_lock
    high_mask = calibrated_prob >= high_lock

    final[low_mask] = 0.0
    final[high_mask] = 1.0

    route = np.full(len(calibrated_prob), "middle", dtype=object)
    route[low_mask] = "locked_benign"
    route[high_mask] = "locked_phish"

    return final, route

# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    y_all = (df["label"].astype(str).str.lower() == "phish").astype(int).values

    phase12 = pd.read_csv(PHASE12_OOF_PATH).sort_values("row_index").reset_index(drop=True)

    if not np.array_equal(phase12["row_index"].values, np.arange(len(df))):
        raise ValueError("Phase 12 OOF file is not aligned to dataset row_index 0..N-1")

    base_prob_all = phase12["phase12_final_prob"].astype(float).values
    y_phase12 = phase12["y_true"].astype(int).values

    if not np.array_equal(y_all, y_phase12):
        raise ValueError("Label mismatch between dataset and Phase 12 OOF")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_preds = []
    fold_metrics = []
    calibrator_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), 1):
        print("\n" + "=" * 70)
        print(f"PHASE 13 ROUTING+CALIBRATION FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        train_prob = base_prob_all[train_idx]
        test_prob = base_prob_all[test_idx]

        calibrator = fit_platt_scaler(train_prob, y_train)
        cal_test_prob = apply_calibration(calibrator, test_prob)

        routed_prob, route_type = apply_strict_routing(
            cal_test_prob,
            low_lock=LOW_LOCK,
            high_lock=HIGH_LOCK
        )

        metrics = compute_metrics(y_test, routed_prob, thr=DEFAULT_THRESHOLD)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

        calibrator_rows.append({
            "Fold": fold,
            "coef": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
            "locked_benign_rate": float((route_type == "locked_benign").mean()),
            "locked_phish_rate": float((route_type == "locked_phish").mean()),
            "middle_rate": float((route_type == "middle").mean()),
        })

        pred_df = pd.DataFrame({
            "row_index": test_idx,
            "y_true": y_test,
            "phase12_prob": test_prob,
            "phase13_calibrated_prob": cal_test_prob,
            "phase13_final_prob": routed_prob,
            "phase13_final_pred": (routed_prob >= DEFAULT_THRESHOLD).astype(int),
            "route_type": route_type,
            "fold": fold
        })
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"LockedBenign={(route_type == 'locked_benign').mean():.3f} | "
            f"LockedPhish={(route_type == 'locked_phish').mean():.3f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase13_routed_calibrated_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase13_routed_calibrated_fold_metrics.csv", index=False)

    calib_df = pd.DataFrame(calibrator_rows)
    calib_df.to_csv(OUT_DIR / "phase13_calibrator_params.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase13_routed_calibrated_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 13 ROUTING+CALIBRATION SUMMARY")
    print("=" * 70)
    print(summary)

    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["phase13_final_prob"].astype(float).values

    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase13_routed_calibrated_threshold_sweep.csv", index=False)

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

    with open(OUT_DIR / "phase13_routed_calibrated_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase13_routed_calibrated_oof_predictions.csv")
    print(OUT_DIR / "phase13_routed_calibrated_fold_metrics.csv")
    print(OUT_DIR / "phase13_routed_calibrated_summary.csv")
    print(OUT_DIR / "phase13_routed_calibrated_threshold_sweep.csv")
    print(OUT_DIR / "phase13_calibrator_params.csv")

if __name__ == "__main__":
    main()
