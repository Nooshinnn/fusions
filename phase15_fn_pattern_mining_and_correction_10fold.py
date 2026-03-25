import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lightgbm import LGBMClassifier
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
ROOT_DIR = Path("phase20_branch_cv_outputs")
PHASE12_OOF_PATH = Path("phase12_error_focused_outputs/phase12_error_focused_oof_predictions.csv")

OUT_DIR = Path("phase15_fn_correction_outputs")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10

BASE_THRESHOLD = 0.49   # best balanced threshold from phase 12
CORRECTION_THRESHOLD = 0.35
SAFE_BENIGN_LOCK = 0.08
NEAR_MISS_MARGIN = 0.12

PRECISION_FLOOR = 0.90
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)

TOP_FN_URL_TOKENS = 80
TOP_FN_HTML_TOKENS = 120

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
# HELPERS
# ============================================================
def safe_text(x):
    return "" if pd.isna(x) else str(x)

def tokenize(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", safe_text(s).lower()) if len(t) >= 2]

def extract_html_text(html: str):
    html = safe_text(html)
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = safe_text(soup.title.string) if soup.title and soup.title.string else ""
        visible = safe_text(soup.get_text(" ", strip=True))[:12000]
        form_text = []
        for tag in soup.find_all(["form", "input", "button", "label"]):
            vals = []
            for k in ["type", "name", "id", "placeholder", "value", "action", "method"]:
                v = tag.get(k)
                if v:
                    vals.append(safe_text(v))
            txt = tag.get_text(" ", strip=True)
            if txt:
                vals.append(safe_text(txt))
            if vals:
                form_text.append(" ".join(vals))
        return " ".join([title, visible, " ".join(form_text)])
    except Exception:
        return ""

def url_parts(url: str):
    p = urlparse(safe_text(url))
    host = (p.hostname or "").lower()
    path = (p.path or "").lower()
    query = (p.query or "").lower()
    return host, path, query

def summarize_latents(X: np.ndarray, prefix: str) -> pd.DataFrame:
    eps = 1e-8
    mag = np.abs(X)
    mag = mag / (mag.sum(axis=1, keepdims=True) + eps)
    return pd.DataFrame({
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

def build_base_tab_features(url_df: pd.DataFrame, html_df: pd.DataFrame) -> pd.DataFrame:
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
    return df

# ============================================================
# FN TOKEN MINING
# ============================================================
def build_fn_lexicons(train_df: pd.DataFrame, full_df: pd.DataFrame, base_prob: np.ndarray):
    y = train_df["y_true"].values.astype(int)
    pred = (base_prob >= BASE_THRESHOLD).astype(int)

    fn_mask = (y == 1) & (pred == 0)
    hard_pos_mask = (y == 1) & (base_prob < BASE_THRESHOLD + NEAR_MISS_MARGIN)
    hard_neg_mask = (y == 0) & (base_prob > BASE_THRESHOLD - NEAR_MISS_MARGIN)

    url_fn = Counter()
    url_other = Counter()
    html_fn = Counter()
    html_other = Counter()

    train_rows = train_df["row_index"].values

    for ridx, is_fn, is_hp, is_hn in zip(train_rows, fn_mask, hard_pos_mask, hard_neg_mask):
        url = safe_text(full_df.iloc[ridx]["url"])
        html = safe_text(full_df.iloc[ridx]["html"])
        host, path, query = url_parts(url)

        url_text = " ".join([host, path, query, url.lower()])
        html_text = extract_html_text(html)

        u_toks = tokenize(url_text)
        h_toks = tokenize(html_text)

        if is_fn or is_hp:
            url_fn.update(u_toks)
            html_fn.update(h_toks)
        if is_hn or (not is_fn and not is_hp):
            url_other.update(u_toks)
            html_other.update(h_toks)

    def top_log_odds(pos_counter, neg_counter, top_k):
        vocab = set(pos_counter) | set(neg_counter)
        total_pos = sum(pos_counter.values()) + 1
        total_neg = sum(neg_counter.values()) + 1
        scored = []
        for tok in vocab:
            c_pos = pos_counter.get(tok, 0)
            c_neg = neg_counter.get(tok, 0)
            if c_pos + c_neg < 3:
                continue
            score = math.log(((c_pos + 1) / total_pos) / ((c_neg + 1) / total_neg))
            scored.append((tok, score, c_pos + c_neg))
        scored = sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        return {tok: score for tok, score, _ in scored}

    return (
        top_log_odds(url_fn, url_other, TOP_FN_URL_TOKENS),
        top_log_odds(html_fn, html_other, TOP_FN_HTML_TOKENS),
    )

def token_score_features(url: str, html: str, url_lex, html_lex):
    host, path, query = url_parts(url)
    url_text = " ".join([host, path, query, safe_text(url).lower()])
    html_text = extract_html_text(html)

    u_scores = [url_lex[t] for t in tokenize(url_text) if t in url_lex]
    h_scores = [html_lex[t] for t in tokenize(html_text) if t in html_lex]

    def summarize(scores):
        if len(scores) == 0:
            return [0.0] * 6
        arr = np.array(scores, dtype=np.float32)
        return [
            float(arr.sum()),
            float(arr.mean()),
            float(arr.max()),
            float(arr.min()),
            float((arr > 0).sum()),
            float(np.abs(arr).sum()),
        ]

    return np.array(summarize(u_scores) + summarize(h_scores), dtype=np.float32)

TOKEN_FEAT_NAMES = [
    "fn_urltok_sum", "fn_urltok_mean", "fn_urltok_max", "fn_urltok_min", "fn_urltok_pos_count", "fn_urltok_abs_sum",
    "fn_htmltok_sum", "fn_htmltok_mean", "fn_htmltok_max", "fn_htmltok_min", "fn_htmltok_pos_count", "fn_htmltok_abs_sum",
]

# ============================================================
# CORRECTION FEATURE BLOCK
# ============================================================
def build_correction_features(tab_df, url_feat, html_feat, token_feats, base_prob):
    url_sum = summarize_latents(url_feat, "url_lat")
    html_sum = summarize_latents(html_feat, "html_lat")

    X = pd.concat([
        tab_df[[
            "html_prob", "url_textcnn_prob", "url_rf_prob",
            "d_url_html", "d_rf_html", "d_url_rf",
            "mean_prob", "max_prob", "min_prob", "std_prob", "range_prob",
            "html_conf", "url_conf", "rf_conf"
        ]].reset_index(drop=True),
        pd.DataFrame({"base_prob": base_prob}),
        url_sum.reset_index(drop=True),
        html_sum.reset_index(drop=True),
        pd.DataFrame(token_feats, columns=TOKEN_FEAT_NAMES),
    ], axis=1)
    return X

# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    y_all = (df["label"].astype(str).str.lower() == "phish").astype(int).values

    phase12 = pd.read_csv(PHASE12_OOF_PATH).sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(phase12["row_index"].values, np.arange(len(df))):
        raise ValueError("Phase 12 OOF file misaligned")

    base_prob_all = phase12["phase12_final_prob"].astype(float).values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_preds = []
    fold_metrics = []
    fi_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"PHASE 15 FN CORRECTION FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_url_feat = np.load(fold_dir / "train_url_features.npy").astype(np.float32)
        test_url_feat = np.load(fold_dir / "test_url_features.npy").astype(np.float32)
        train_html_feat = np.load(fold_dir / "train_html_features.npy").astype(np.float32)
        test_html_feat = np.load(fold_dir / "test_html_features.npy").astype(np.float32)

        train_tab = build_base_tab_features(train_url, train_html)
        test_tab = build_base_tab_features(test_url, test_html)

        train_base_prob = base_prob_all[train_idx]
        test_base_prob = base_prob_all[test_idx]

        # build lexicons from training fold only
        url_lex, html_lex = build_fn_lexicons(train_tab, df, train_base_prob)

        train_tok = np.vstack([
            token_score_features(df.iloc[r]["url"], df.iloc[r]["html"], url_lex, html_lex)
            for r in train_idx
        ])
        test_tok = np.vstack([
            token_score_features(df.iloc[r]["url"], df.iloc[r]["html"], url_lex, html_lex)
            for r in test_idx
        ])

        X_train = build_correction_features(train_tab, train_url_feat, train_html_feat, train_tok, train_base_prob)
        X_test = build_correction_features(test_tab, test_url_feat, test_html_feat, test_tok, test_base_prob)

        y_train = train_tab["y_true"].values.astype(int)
        y_test = test_tab["y_true"].values.astype(int)

        # define correction target:
        # positive = hard phishing that base system tends to miss
        base_pred_train = (train_base_prob >= BASE_THRESHOLD).astype(int)
        is_fn = (y_train == 1) & (base_pred_train == 0)
        is_near_miss_pos = (y_train == 1) & (train_base_prob < BASE_THRESHOLD + NEAR_MISS_MARGIN)
        correction_target = (is_fn | is_near_miss_pos).astype(int)

        # train on only relevant subset:
        # all correction positives + matched hard negatives
        hard_neg = (y_train == 0) & (train_base_prob > BASE_THRESHOLD - NEAR_MISS_MARGIN)
        train_mask = correction_target.astype(bool) | hard_neg.astype(bool)

        X_corr = X_train.loc[train_mask].reset_index(drop=True)
        y_corr = correction_target[train_mask]

        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=0.35,
            random_state=SEED + fold,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_corr, y_corr)
        corr_prob = model.predict_proba(X_test)[:, 1]

        # correction routing:
        # keep very safe benign locked
        final_prob = test_base_prob.copy()
        safe_benign = test_base_prob <= SAFE_BENIGN_LOCK

        # apply correction when:
        # base is not very safe benign AND correction model sees missed-phish pattern
        apply_mask = (~safe_benign) & (corr_prob >= CORRECTION_THRESHOLD)

        # only increase suspicion, never lower it
        final_prob[apply_mask] = np.maximum(final_prob[apply_mask], corr_prob[apply_mask])

        metrics = compute_metrics(y_test, final_prob, thr=BASE_THRESHOLD)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

        fi = pd.DataFrame({
            "feature": X_corr.columns,
            "importance": model.feature_importances_,
            "fold": fold
        })
        fi_rows.append(fi)

        pred_df = pd.DataFrame({
            "row_index": test_idx,
            "y_true": y_test,
            "phase12_prob": test_base_prob,
            "phase15_corr_prob": corr_prob,
            "phase15_apply_mask": apply_mask.astype(int),
            "phase15_final_prob": final_prob,
            "phase15_final_pred": (final_prob >= BASE_THRESHOLD).astype(int),
            "fold": fold
        })
        all_preds.append(pred_df)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"Applied={apply_mask.mean():.3f}"
        )

    oof_df = pd.concat(all_preds, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase15_fn_correction_oof_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / "phase15_fn_correction_fold_metrics.csv", index=False)

    summary = fold_metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase15_fn_correction_summary.csv")

    fi = pd.concat(fi_rows, ignore_index=True)
    fi.to_csv(OUT_DIR / "phase15_fn_correction_feature_importance_by_fold.csv", index=False)
    fi_mean = fi.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    fi_mean.to_csv(OUT_DIR / "phase15_fn_correction_feature_importance_mean.csv", index=False)

    print("\n" + "=" * 70)
    print("PHASE 15 FN CORRECTION SUMMARY")
    print("=" * 70)
    print(summary)

    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["phase15_final_prob"].astype(float).values

    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase15_fn_correction_threshold_sweep.csv", index=False)

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

    with open(OUT_DIR / "phase15_fn_correction_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "phase15_fn_correction_oof_predictions.csv")
    print(OUT_DIR / "phase15_fn_correction_fold_metrics.csv")
    print(OUT_DIR / "phase15_fn_correction_summary.csv")
    print(OUT_DIR / "phase15_fn_correction_threshold_sweep.csv")
    print(OUT_DIR / "phase15_fn_correction_feature_importance_mean.csv")

if __name__ == "__main__":
    main()
