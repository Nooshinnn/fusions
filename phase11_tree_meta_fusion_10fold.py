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

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("phase20_branch_cv_outputs")
PHASE06_OOF_PATH = Path("phase06_multichannel_joint_outputs/phase06_multichannel_oof_predictions.csv")
PHASE09_OOF_PATH = Path("phase09_specialist_outputs/phase09_specialist_oof_predictions.csv")

OUT_DIR = Path("phase11_tree_meta_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
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
# BASE TAB FEATURES
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

# ============================================================
# LATENT SUMMARY FEATURES
# ============================================================
def summarize_latents(X: np.ndarray, prefix: str) -> pd.DataFrame:
    eps = 1e-8
    df = pd.DataFrame({
        f"{prefix}_mean": X.mean(axis=1),
        f"{prefix}_std": X.std(axis=1),
        f"{prefix}_min": X.min(axis=1),
        f"{prefix}_max": X.max(axis=1),
        f"{prefix}_norm": np.linalg.norm(X, axis=1),
        f"{prefix}_q25": np.quantile(X, 0.25, axis=1),
        f"{prefix}_q50": np.quantile(X, 0.50, axis=1),
        f"{prefix}_q75": np.quantile(X, 0.75, axis=1),
        f"{prefix}_mean_abs": np.abs(X).mean(axis=1),
        f"{prefix}_max_abs": np.abs(X).max(axis=1),
        f"{prefix}_sparsity_small": (np.abs(X) < 0.10).mean(axis=1),
        f"{prefix}_pos_ratio": (X > 0).mean(axis=1),
    })

    # top-k summaries
    sort_X = np.sort(X, axis=1)
    df[f"{prefix}_top1"] = sort_X[:, -1]
    df[f"{prefix}_top2"] = sort_X[:, -2]
    df[f"{prefix}_top3"] = sort_X[:, -3]
    df[f"{prefix}_bot1"] = sort_X[:, 0]
    df[f"{prefix}_bot2"] = sort_X[:, 1]
    df[f"{prefix}_bot3"] = sort_X[:, 2]

    # entropy-like normalized magnitude summary
    mag = np.abs(X)
    mag = mag / (mag.sum(axis=1, keepdims=True) + eps)
    df[f"{prefix}_entropy_like"] = -(mag * np.log(mag + eps)).sum(axis=1)

    return df

# ============================================================
# MAIN
# ============================================================
def main():
    phase06_oof = pd.read_csv(PHASE06_OOF_PATH).sort_values("row_index").reset_index(drop=True)
    phase09_oof = pd.read_csv(PHASE09_OOF_PATH).sort_values("row_index").reset_index(drop=True)

    all_preds = []
    fold_metrics = []
    feature_importance_rows = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 11 TREE META FUSION FOLD {fold}/{N_SPLITS}")
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

        # attach Phase 6 / Phase 9 OOF probabilities by row_index
        train_p6 = phase06_oof[phase06_oof["row_index"].isin(train_df["row_index"])].sort_values("row_index").reset_index(drop=True)
        test_p6 = phase06_oof[phase06_oof["row_index"].isin(test_df["row_index"])].sort_values("row_index").reset_index(drop=True)

        train_p9 = phase09_oof[phase09_oof["row_index"].isin(train_df["row_index"])].sort_values("row_index").reset_index(drop=True)
        test_p9 = phase09_oof[phase09_oof["row_index"].isin(test_df["row_index"])].sort_values("row_index").reset_index(drop=True)

        if not np.array_equal(train_df["row_index"].values, train_p6["row_index"].values):
            raise ValueError(f"Phase 6 train row mismatch on fold {fold}")
        if not np.array_equal(test_df["row_index"].values, test_p6["row_index"].values):
            raise ValueError(f"Phase 6 test row mismatch on fold {fold}")
        if not np.array_equal(train_df["row_index"].values, train_p9["row_index"].values):
            raise ValueError(f"Phase 9 train row mismatch on fold {fold}")
        if not np.array_equal(test_df["row_index"].values, test_p9["row_index"].values):
            raise ValueError(f"Phase 9 test row mismatch on fold {fold}")

        train_df["phase06_prob"] = train_p6["multichannel_prob"].astype(float).values
        test_df["phase06_prob"] = test_p6["multichannel_prob"].astype(float).values

        train_df["phase09_main_prob"] = train_p9["main_prob"].astype(float).values
        train_df["phase09_specialist_prob"] = train_p9["specialist_prob"].astype(float).values
        train_df["phase09_used_specialist"] = train_p9["used_specialist"].astype(float).values
        train_df["phase09_final_prob"] = train_p9["final_prob"].astype(float).values

        test_df["phase09_main_prob"] = test_p9["main_prob"].astype(float).values
        test_df["phase09_specialist_prob"] = test_p9["specialist_prob"].astype(float).values
        test_df["phase09_used_specialist"] = test_p9["used_specialist"].astype(float).values
        test_df["phase09_final_prob"] = test_p9["final_prob"].astype(float).values

        # latent summaries
        train_url_sum = summarize_latents(train_url_feat, "url_lat")
        test_url_sum = summarize_latents(test_url_feat, "url_lat")
        train_html_sum = summarize_latents(train_html_feat, "html_lat")
        test_html_sum = summarize_latents(test_html_feat, "html_lat")

        # cross latent summaries
        train_cross = pd.DataFrame({
            "lat_cosine": np.sum(train_url_feat * train_html_feat[:, :train_url_feat.shape[1]], axis=1)
                          if train_url_feat.shape[1] == train_html_feat.shape[1]
                          else np.nan,
            "lat_norm_diff": np.abs(np.linalg.norm(train_url_feat, axis=1) - np.linalg.norm(train_html_feat, axis=1)),
        })
        test_cross = pd.DataFrame({
            "lat_cosine": np.sum(test_url_feat * test_html_feat[:, :test_url_feat.shape[1]], axis=1)
                          if test_url_feat.shape[1] == test_html_feat.shape[1]
                          else np.nan,
            "lat_norm_diff": np.abs(np.linalg.norm(test_url_feat, axis=1) - np.linalg.norm(test_html_feat, axis=1)),
        })

        # if dims differ, fill cosine with simpler proxy
        if train_cross["lat_cosine"].isna().all():
            train_cross["lat_cosine"] = (
                train_url_sum["url_lat_mean"] * train_html_sum["html_lat_mean"] +
                train_url_sum["url_lat_std"] * train_html_sum["html_lat_std"]
            )
            test_cross["lat_cosine"] = (
                test_url_sum["url_lat_mean"] * test_html_sum["html_lat_mean"] +
                test_url_sum["url_lat_std"] * test_html_sum["html_lat_std"]
            )

        feature_blocks_train = [
            train_df.drop(columns=["row_index", "url", "y_true"]),
            train_url_sum,
            train_html_sum,
            train_cross,
        ]
        feature_blocks_test = [
            test_df.drop(columns=["row_index", "url", "y_true"]),
            test_url_sum,
            test_html_sum,
            test_cross,
        ]

        X_train = pd.concat(feature_blocks_train, axis=1)
        X_test = pd.concat(feature_blocks_test, axis=1)

        y_train = train_df["y_true"].astype(int).values
        y_test = test_df["y_true"].astype(int).values

        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=8,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42 + fold,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, test_prob, thr=0.5)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

        imp_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
            "fold": fold
        })
        feature_importance_rows.append(imp_df)

        pred_df = test_df[["row_index", "url", "y_true"]].copy()
        pred_df["tree_meta_prob"] = test_prob
        pred_df["tree_meta_pred"] = (test_prob >= 0.5).astype(int)
        pred_df["fold"] = fold
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase11_tree_meta_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase11_tree_meta_fold_metrics.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase11_tree_meta_summary.csv")

    feat_imp = pd.concat(feature_importance_rows, ignore_index=True)
    feat_imp.to_csv(OUT_DIR / "phase11_tree_meta_feature_importance_by_fold.csv", index=False)

    feat_imp_mean = feat_imp.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    feat_imp_mean.to_csv(OUT_DIR / "phase11_tree_meta_feature_importance_mean.csv", index=False)

    print("\n" + "=" * 70)
    print("PHASE 11 TREE META FUSION SUMMARY")
    print("=" * 70)
    print(summary)

    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["tree_meta_prob"].astype(float).values

    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase11_tree_meta_threshold_sweep.csv", index=False)

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

    with open(OUT_DIR / "phase11_tree_meta_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase11_tree_meta_oof_predictions.csv")
    print(OUT_DIR / "phase11_tree_meta_fold_metrics.csv")
    print(OUT_DIR / "phase11_tree_meta_summary.csv")
    print(OUT_DIR / "phase11_tree_meta_threshold_sweep.csv")
    print(OUT_DIR / "phase11_tree_meta_feature_importance_mean.csv")

if __name__ == "__main__":
    main()
