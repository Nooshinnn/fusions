import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)

DATA_PATH = "final_multimodal_dataset.parquet"
PHASE15_PATH = "phase15_fn_correction_outputs/phase15_fn_correction_oof_predictions.csv"

OUT_DIR = Path("phase16_precision_recovery_outputs")
OUT_DIR.mkdir(exist_ok=True)

BASE_THRESHOLD = 0.55   # best phase15 threshold
PRECISION_FLOOR = 0.90
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)

TRUSTED_HOST_HINTS = [
    "nbcnews.com", "merriam-webster.com", "wikipedia.org", "bbc.com", "nytimes.com",
    "theguardian.com", "reuters.com", "forbes.com", "investopedia.com", "gov.uk",
    "nhs.uk", "edu", "ac.uk"
]

BENIGN_PLATFORM_HINTS = [
    "forms.office.com", "docs.google.com", "sites.google.com", "notion.site",
    "medium.com", "substack.com", "wordpress.com"
]

ARTICLE_WORDS = [
    "news", "article", "health", "sports", "travel", "dictionary", "guide",
    "blog", "post", "story", "resources", "product", "meeting"
]

ACCOUNT_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "password", "wallet", "billing", "recover", "confirm"
]

SAFE_EXTENSIONS = [".pdf", ".txt", ".jpg", ".jpeg", ".png", ".webp"]

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
    return pd.DataFrame([compute_metrics(y_true, y_prob, thr=t) for t in THRESHOLDS])

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def count_hits(text: str, vocab):
    t = text.lower()
    return sum(1 for v in vocab if v in t)

def benign_evidence(url: str, html: str):
    url = safe_text(url)
    html = safe_text(html)
    p = urlparse(url)
    host = (p.hostname or "").lower()
    path = (p.path or "").lower()
    query = (p.query or "").lower()
    full = (url + " " + path + " " + query).lower()

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = BeautifulSoup("", "html.parser")

    visible = safe_text(soup.get_text(" ", strip=True))[:5000].lower()
    title = safe_text(soup.title.string).lower() if soup.title and soup.title.string else ""

    password_inputs = len(soup.find_all("input", {"type": lambda x: x and str(x).lower() == "password"}))
    forms = len(soup.find_all("form"))
    iframes = len(soup.find_all("iframe"))

    score = 0
    reasons = []

    if any(h in host for h in TRUSTED_HOST_HINTS):
        score += 3
        reasons.append("trusted_host")

    if any(h in host for h in BENIGN_PLATFORM_HINTS):
        score += 2
        reasons.append("benign_platform")

    # article/reference style
    article_hits = count_hits(path + " " + title + " " + visible[:1000], ARTICLE_WORDS)
    if article_hits >= 2:
        score += 2
        reasons.append("article_style")

    # no obvious login/account cues
    account_hits = count_hits(full + " " + visible[:1000], ACCOUNT_WORDS)
    if account_hits == 0:
        score += 2
        reasons.append("no_account_words")

    # no credential form
    if password_inputs == 0 and forms == 0:
        score += 2
        reasons.append("no_forms_no_password")

    if password_inputs == 0 and forms <= 1:
        score += 1
        reasons.append("no_password")

    # long informational path
    path_parts = [x for x in re.split(r"[^a-z0-9]+", path) if x]
    if len(path_parts) >= 4 and account_hits == 0:
        score += 1
        reasons.append("long_article_path")

    # safe file/resource style
    if any(path.endswith(ext) for ext in SAFE_EXTENSIONS):
        score += 1
        reasons.append("safe_extension")

    # iframes often suspicious; absence slightly helpful
    if iframes == 0:
        score += 0.5
        reasons.append("no_iframe")

    return score, "|".join(reasons)

def apply_precision_recovery(df_pred: pd.DataFrame, df_data: pd.DataFrame):
    merged = df_pred.merge(
        df_data.reset_index().rename(columns={"index": "row_index"})[["row_index", "url", "html", "label"]],
        on="row_index",
        how="left"
    )

    final_prob = merged["phase15_final_prob"].astype(float).values.copy()
    protector_applied = np.zeros(len(merged), dtype=int)
    protector_score = np.zeros(len(merged), dtype=float)
    protector_reason = []

    for i, row in merged.iterrows():
        prob = float(row["phase15_final_prob"])
        url = safe_text(row["url"])
        html = safe_text(row["html"])

        bscore, breason = benign_evidence(url, html)
        protector_score[i] = bscore
        protector_reason.append(breason)

        # only act on likely phishing cases
        if prob >= 0.45:
            # strong benign evidence
            if bscore >= 6:
                final_prob[i] = min(final_prob[i], 0.20)
                protector_applied[i] = 1
            elif bscore >= 4:
                final_prob[i] = min(final_prob[i], max(0.30, final_prob[i] - 0.20))
                protector_applied[i] = 1

    merged["phase16_final_prob"] = final_prob
    merged["phase16_protector_applied"] = protector_applied
    merged["phase16_protector_score"] = protector_score
    merged["phase16_protector_reason"] = protector_reason
    return merged

def main():
    data = pd.read_parquet(DATA_PATH)
    pred = pd.read_csv(PHASE15_PATH).sort_values("row_index").reset_index(drop=True)

    out = apply_precision_recovery(pred, data)

    y_true = (out["label"].astype(str).str.lower() == "phish").astype(int).values
    y_prob = out["phase16_final_prob"].astype(float).values

    summary_default = compute_metrics(y_true, y_prob, thr=BASE_THRESHOLD)

    print("\n" + "=" * 70)
    print("PHASE 16 PRECISION RECOVERY SUMMARY")
    print("=" * 70)
    for k, v in summary_default.items():
        if isinstance(v, float):
            print(f"{k:<18} {v:.6f}")
        else:
            print(f"{k:<18} {v}")

    sweep = build_threshold_sweep(y_true, y_prob)
    sweep.to_csv(OUT_DIR / "phase16_precision_recovery_threshold_sweep.csv", index=False)

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

    out["phase16_final_pred"] = (out["phase16_final_prob"] >= float(best_mcc["Threshold"])).astype(int)

    out.to_csv(OUT_DIR / "phase16_precision_recovery_oof_predictions.csv", index=False)

    with open(OUT_DIR / "phase16_precision_recovery_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "default_threshold_metrics": summary_default,
            "best_threshold_by_mcc": best_mcc.to_dict(),
            "best_threshold_by_f1": best_f1.to_dict(),
            "best_threshold_by_lowest_fn_precision_floor": None if best_low_fn is None else best_low_fn.to_dict(),
        }, f, indent=2)

    applied_rate = out["phase16_protector_applied"].mean()
    print(f"\nProtector applied rate: {applied_rate:.4f}")

    print("\nSaved:")
    print(OUT_DIR / "phase16_precision_recovery_oof_predictions.csv")
    print(OUT_DIR / "phase16_precision_recovery_threshold_sweep.csv")
    print(OUT_DIR / "phase16_precision_recovery_threshold_summary.json")

if __name__ == "__main__":
    main()
