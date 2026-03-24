import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

DATA_PATH = "phreshphish_balanced.parquet"

OUT_DIR = Path("phase26_outputs")
OUT_DIR.mkdir(exist_ok=True)

TEXT_FEATURES_PATH = OUT_DIR / "html_text_tfidf.npz"
TEXT_VECTORIZER_PATH = OUT_DIR / "html_text_vectorizer.joblib"
BRAND_FEATURES_PATH = OUT_DIR / "brand_mismatch_features.npy"
BRAND_FEATURE_NAMES_PATH = OUT_DIR / "brand_mismatch_feature_names.json"

BRAND_ALIASES = {
    "facebook": ["facebook", "fb", "meta"],
    "instagram": ["instagram", "insta", "ig"],
    "whatsapp": ["whatsapp", "wa"],
    "microsoft": ["microsoft", "office", "outlook", "live", "ms", "onedrive", "sharepoint", "forms"],
    "apple": ["apple", "icloud", "itunes"],
    "paypal": ["paypal"],
    "amazon": ["amazon", "prime"],
    "netflix": ["netflix"],
    "telegram": ["telegram"],
    "canva": ["canva"],
    "coinbase": ["coinbase"],
    "binance": ["binance"],
    "robinhood": ["robinhood"],
    "chase": ["chase"],
    "sbb": ["sbb"],
    "bank": ["bank", "banking"],
}

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_spaces(text)

def extract_text_fields(html: str):
    html = safe_text(html)
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return "", "", "", ""

    title = ""
    if soup.title and soup.title.string:
        title = safe_text(soup.title.string)

    metas = []
    for m in soup.find_all("meta"):
        content = m.get("content")
        if content:
            metas.append(safe_text(content))
    meta_text = " ".join(metas)

    visible = safe_text(soup.get_text(" ", strip=True))
    visible = visible[:20000]

    form_texts = []
    for tag in soup.find_all(["input", "button", "label"]):
        vals = []
        for k in ["placeholder", "value", "name", "id", "aria-label", "type"]:
            v = tag.get(k)
            if v:
                vals.append(safe_text(v))
        txt = tag.get_text(" ", strip=True)
        if txt:
            vals.append(safe_text(txt))
        if vals:
            form_texts.append(" ".join(vals))
    form_text = " ".join(form_texts)

    return title, meta_text, visible, form_text

def host_tokens(host: str):
    toks = re.split(r"[^a-z0-9]+", host.lower())
    return {t for t in toks if t}

def detect_brands(text: str):
    found = []
    t = text.lower()
    for brand, aliases in BRAND_ALIASES.items():
        if any(a in t for a in aliases):
            found.append(brand)
    return found

def brand_feature_row(url: str, title: str, meta_text: str, visible: str, form_text: str):
    p = urlparse(safe_text(url))
    host = safe_text(p.netloc).lower()
    path = safe_text(p.path).lower()
    query = safe_text(p.query).lower()
    host_tok = host_tokens(host)

    title_l = title.lower()
    meta_l = meta_text.lower()
    vis_l = visible.lower()
    form_l = form_text.lower()

    combined = " ".join([title_l, meta_l, vis_l, form_l])

    brands_found = detect_brands(combined)
    n_brands = len(brands_found)

    host_match_count = 0
    path_match_count = 0
    form_match_count = 0

    social_brand = 0
    finance_brand = 0
    microsoft_brand = 0

    for b in brands_found:
        aliases = BRAND_ALIASES[b]
        if any(a in host for a in aliases):
            host_match_count += 1
        if any(a in path or a in query for a in aliases):
            path_match_count += 1
        if any(a in form_l for a in aliases):
            form_match_count += 1

        if b in ["facebook", "instagram", "whatsapp", "telegram", "canva"]:
            social_brand = 1
        if b in ["paypal", "coinbase", "binance", "robinhood", "chase", "bank", "sbb"]:
            finance_brand = 1
        if b in ["microsoft", "apple"]:
            microsoft_brand = 1

    brand_found = int(n_brands > 0)
    host_brand_mismatch = int(brand_found and host_match_count == 0)
    path_brand_mismatch = int(brand_found and path_match_count == 0)
    multi_brand_page = int(n_brands >= 2)
    brand_in_title = int(len(detect_brands(title_l)) > 0)
    brand_in_meta = int(len(detect_brands(meta_l)) > 0)
    brand_in_form = int(len(detect_brands(form_l)) > 0)

    login_tokens = ["login", "sign in", "signin", "password", "verify", "confirm", "account", "security", "recover"]
    login_hits = sum(1 for t in login_tokens if t in combined)

    brand_login_mismatch = int(brand_found and login_hits >= 2 and host_match_count == 0)
    suspicious_legit_service_mismatch = int(
        any(x in host for x in ["forms.office", "web.facebook", "sharepoint", "onedrive", "canva"]) and
        host_match_count == 0 and brand_found
    )

    return [
        brand_found,
        n_brands,
        host_match_count,
        path_match_count,
        form_match_count,
        host_brand_mismatch,
        path_brand_mismatch,
        multi_brand_page,
        brand_in_title,
        brand_in_meta,
        brand_in_form,
        login_hits,
        brand_login_mismatch,
        suspicious_legit_service_mismatch,
        social_brand,
        finance_brand,
        microsoft_brand,
    ]

BRAND_FEATURE_NAMES = [
    "brand_found",
    "n_brands_found",
    "host_brand_match_count",
    "path_brand_match_count",
    "form_brand_match_count",
    "host_brand_mismatch",
    "path_brand_mismatch",
    "multi_brand_page",
    "brand_in_title",
    "brand_in_meta",
    "brand_in_form",
    "login_hits_text",
    "brand_login_mismatch",
    "suspicious_legit_service_mismatch",
    "social_brand_found",
    "finance_brand_found",
    "microsoft_or_apple_brand_found",
]

def main():
    df = pd.read_parquet(DATA_PATH)

    text_docs = []
    brand_rows = []

    for _, row in df.iterrows():
        title, meta_text, visible, form_text = extract_text_fields(row["html"])

        doc = " [TITLE] ".join([
            clean_text(title),
            clean_text(meta_text),
            clean_text(form_text),
            clean_text(visible[:8000])
        ])
        text_docs.append(doc)

        brand_rows.append(
            brand_feature_row(
                safe_text(row["url"]),
                title,
                meta_text,
                visible,
                form_text
            )
        )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True
    )
    X_text = vectorizer.fit_transform(text_docs)
    sparse.save_npz(TEXT_FEATURES_PATH, X_text)
    joblib.dump(vectorizer, TEXT_VECTORIZER_PATH)

    brand_X = np.array(brand_rows, dtype=np.float32)
    np.save(BRAND_FEATURES_PATH, brand_X)

    with open(BRAND_FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(BRAND_FEATURE_NAMES, f, indent=2)

    print("Saved:", TEXT_FEATURES_PATH, X_text.shape)
    print("Saved:", TEXT_VECTORIZER_PATH)
    print("Saved:", BRAND_FEATURES_PATH, brand_X.shape)
    print("Saved:", BRAND_FEATURE_NAMES_PATH)

if __name__ == "__main__":
    main()