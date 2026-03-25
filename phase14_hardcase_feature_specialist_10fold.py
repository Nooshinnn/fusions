import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)
from lightgbm import LGBMClassifier

ROOT_DIR = Path("phase20_branch_cv_outputs")
PHASE13_OOF_PATH = Path("phase13_routing_calibration_outputs/phase13_routed_calibrated_oof_predictions.csv")
PHASE14_FEAT_PATH = Path("phase14_outputs/phase14_comprehensive_offline_features.npy")

OUT_DIR = Path("phase14_feature_specialist_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
BASE_THRESHOLD = 0.41
FINAL_THRESHOLD = 0.41
PRECISION_FLOOR = 0.90
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)
DISAGREEMENT_THRESHOLD = 0.35

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
    return df

def summarize_latents(X: np.ndarray, prefix: str) -> pd.DataFrame:
    eps = 1e-8
    mag = np.abs(X)
    mag = mag / (mag.sum(axis=1, keepdims=True) + eps)

    out = pd.DataFrame({
        f"{prefix}_mean": X.mean(axis=1),
        f"{prefix}_std": X.std(axis=1),
        f"{prefix}_min": X.min(axis=1),
        f"{prefix}_max": X.max(axis=1),
        f"{prefix}_norm": np.linalg.norm(X, axis=1),
        f"{prefix}_q25": np.quantile(X, 0.25, axis=1),
        f"{prefix}_q50": np.quantile(X, 0.50, axis=1),
        f"{prefix}_q75": np.quantile(X, 0.75, axis=1),
        f"{prefix}_entropy_like": -(mag * np.log(mag + eps)).sum(axis=1),
    })
    return out

def build_hard_mask(df: pd.DataFrame):
    base_pred = (df["phase13_final_prob"].values >= BASE_THRESHOLD).astype(int)
    y = df["y_true"].values.astype(int)
    is_error = (base_pred != y)
    is_uncertain = (df["phase13_final_prob"].values >= 0.20) & (df["phase13_final_prob"].values <= 0.80)
    high_disagreement = (
        (df["range_prob"].values >= DISAGREEMENT_THRESHOLD) |
        (df["d_url_html"].values >= DISAGREEMENT_THRESHOLD) |
        (df["d_rf_html"].values >= DISAGREEMENT_THRESHOLD)
    )
    locked = np.isin(df["route_type"].values, ["locked_benign", "locked_phish"])
    # hard means: error, uncertain, disagreement, or not confidently locked
    return (is_error | is_uncertain | high_disagreement | (~locked)).astype(int)

def main():
    phase13 = pd.read_csv(PHASE13_OOF_PATH).sort_values("row_index").reset_index(drop=True)
    phase14_feats = np.load(PHASE14_FEAT_PATH).astype(np.float32)

    all_preds = []
    fold_metrics = []
    fi_rows = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 14 FEATURE SPECIALIST FOLD {fold}/{N_SPLITS}")
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

        train_p13 = phase13[phase13["row_index"].isin(train_df["row_index"])].sort_values("row_index").reset_index(drop=True)
        test_p13 = phase13[phase13["row_index"].isin(test_df["row_index"])].sort_values("row_index").reset_index(drop=True)

        if not np.array_equal(train_df["row_index"].values, train_p13["row_index"].values):
            raise ValueError(f"Train row mismatch on fold {fold}")
        if not np.array_equal(test_df["row_index"].values, test_p13["row_index"].values):
            raise ValueError(f"Test row mismatch on fold {fold}")

        train_df["phase13_final_prob"] = train_p13["phase13_final_prob"].astype(float).values
        train_df["phase13_calibrated_prob"] = train_p13["phase13_calibrated_prob"].astype(float).values
        train_df["route_type"] = train_p13["route_type"].astype(str).values

        test_df["phase13_final_prob"] = test_p13["phase13_final_prob"].astype(float).values
        test_df["phase13_calibrated_prob"] = test_p13["phase13_calibrated_prob"].astype(float).values
        test_df["route_type"] = test_p13["route_type"].astype(str).values

        train_url_sum = summarize_latents(train_url_feat, "url_lat")
        test_url_sum = summarize_latents(test_url_feat, "url_lat")
        train_html_sum = summarize_latents(train_html_feat, "html_lat")
        test_html_sum = summarize_latents(test_html_feat, "html_lat")

        train_feat = phase14_feats[train_df["row_index"].values]
        test_feat = phase14_feats[test_df["row_index"].values]

        X_train_full = pd.concat([
            train_df.drop(columns=["row_index", "url", "y_true", "route_type"]),
            train_url_sum,
            train_html_sum,
            pd.DataFrame(train_feat, index=train_df.index)
        ], axis=1)

        X_test_full = pd.concat([
            test_df.drop(columns=["row_index", "url", "y_true", "route_type"]),
            test_url_sum,
            test_html_sum,
            pd.DataFrame(test_feat, index=test_df.index)
        ], axis=1)

        y_train_full = train_df["y_true"].astype(int).values
        y_test = test_df["y_true"].astype(int).values

        train_hard = build_hard_mask(train_df)
        test_hard = build_hard_mask(test_df)

        hard_idx = np.where(train_hard == 1)[0]
        X_train = X_train_full.iloc[hard_idx].copy()
        y_train = y_train_full[hard_idx]

        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42 + fold,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        hard_prob = model.predict_proba(X_test_full)[:, 1]

        final_prob = test_df["phase13_final_prob"].values.astype(np.float32).copy()
        final_prob[test_hard == 1] = hard_prob[test_hard == 1]

        metrics = compute_metrics(y_test, final_prob, thr=FINAL_THRESHOLD)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

        imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
            "fold": fold
        })
        fi_rows.append(imp)

        pred_df = test_df[["row_index", "url", "y_true"]].copy()
        pred_df["phase13_final_prob"] = test_df["phase13_final_prob"].values
        pred_df["phase14_hard_prob"] = hard_prob
        pred_df["phase14_hard_mask"] = test_hard.astype(int)
        pred_df["phase14_final_prob"] = final_prob
        pred_df["phase14_final_pred"] = (final_prob >= FINAL_THRESHOLD).astype(int)
        pred_df["fold"] = fold
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"HardMaskRate={test_hard.mean():.3f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase14_feature_specialist_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase14_feature_specialist_fold_metrics.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase14_feature_specialist_summary.csv")

    fi = pd.concat(fi_rows, ignore_index=True)
    fi.to_csv(OUT_DIR / "phase14_feature_specialist_feature_importance_by_fold.csv", index=False)
    fi_mean = fi.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    fi_mean.to_csv(OUT_DIR / "phase14_feature_specialist_feature_importance_mean.csv", index=False)

    print("\n" + "=" * 70)
    print("PHASE 14 FEATURE SPECIALIST SUMMARY")
    print("=" * 70)
    print(summary)

    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["phase14_final_prob"].astype(float).values
    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase14_feature_specialist_threshold_sweep.csv", index=False)

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

    with open(OUT_DIR / "phase14_feature_specialist_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase14_feature_specialist_oof_predictions.csv")
    print(OUT_DIR / "phase14_feature_specialist_fold_metrics.csv")
    print(OUT_DIR / "phase14_feature_specialist_summary.csv")
    print(OUT_DIR / "phase14_feature_specialist_threshold_sweep.csv")
    print(OUT_DIR / "phase14_feature_specialist_feature_importance_mean.csv")

if __name__ == "__main__":
    main()
