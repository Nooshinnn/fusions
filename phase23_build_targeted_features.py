import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

DATA_PATH = "phreshphish_balanced.parquet"

OUT_DIR = Path("phase23_outputs")
OUT_DIR.mkdir(exist_ok=True)

TARGETED_NPY = OUT_DIR / "targeted_features.npy"
TARGETED_JSON = OUT_DIR / "targeted_feature_names.json"

HOSTED_PLATFORMS = [
    "pages.dev", "web.app", "firebaseapp.com", "r2.dev", "ipfs.io",
    "vercel.app", "webflow.io", "github.io", "blogspot.", "highlevel",
    "weebly.com", "wixsite.com", "webadorsite.com", "000webhostapp.com"
]

SHORTLINK_HOSTS = [
    "qrco.de", "bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "rb.gy"
]

STORAGE_HOSTS = [
    "r2.dev", "ipfs.io", "amazonaws.com", "blob.core.windows.net",
    "storage.googleapis.com"
]

SUSPICIOUS_PATH_TOKENS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "support", "check", "update", "confirm", "password",
    "wallet", "recover", "copyright", "admin", "billing", "ionos"
]

BRAND_TOKENS = [
    "facebook", "instagram", "whatsapp", "microsoft", "office", "outlook",
    "apple", "icloud", "paypal", "amazon", "netflix", "telegram",
    "canva", "meta", "sbb", "bank", "visa", "mastercard", "coinbase",
    "binance", "robinhood", "chase", "microsoft forms", "forms"
]

LOGIN_TOKENS = [
    "login", "sign in", "signin", "password", "verify", "verification",
    "confirm", "account", "recover", "security check", "two-factor"
]

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def host_tokens(host):
    parts = re.split(r"[^a-z0-9]+", host.lower())
    return {p for p in parts if p}

def text_tokens(text):
    parts = re.split(r"[^a-z0-9]+", text.lower())
    return {p for p in parts if p}

def count_matches(text, vocab):
    t = text.lower()
    return sum(1 for v in vocab if v in t)

def any_host_contains(host, patterns):
    h = host.lower()
    return int(any(p in h for p in patterns))

def extract_html_signals(html):
    html = safe_text(html)
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = None

    lower = html.lower()

    n_forms = 0 if soup is None else len(soup.find_all("form"))
    n_inputs = 0 if soup is None else len(soup.find_all("input"))
    n_pw = 0 if soup is None else len(soup.find_all("input", {"type": "password"}))

    title = ""
    if soup is not None and soup.title and soup.title.string:
        title = safe_text(soup.title.string)

    metas = []
    if soup is not None:
        for m in soup.find_all("meta"):
            content = m.get("content")
            if content:
                metas.append(safe_text(content))
    meta_text = " ".join(metas)

    visible_text = ""
    if soup is not None:
        visible_text = safe_text(soup.get_text(" ", strip=True))[:10000]

    brand_hits = count_matches(" ".join([title, meta_text, visible_text, lower]), BRAND_TOKENS)
    login_hits = count_matches(" ".join([title, meta_text, visible_text, lower]), LOGIN_TOKENS)

    form_actions = []
    if soup is not None:
        for f in soup.find_all("form"):
            a = safe_text(f.get("action"))
            if a:
                form_actions.append(a.lower())

    return {
        "n_forms": n_forms,
        "n_inputs": n_inputs,
        "n_pw": n_pw,
        "brand_hits_html": brand_hits,
        "login_hits_html": login_hits,
        "has_submit_text": int("submit" in lower or "continue" in lower or "next" in lower),
        "has_form_action": int(len(form_actions) > 0),
        "form_action_count": len(form_actions),
        "form_action_external_like": int(any(a.startswith("http") for a in form_actions)),
        "text_blob": " ".join([title, meta_text, visible_text]).lower()[:15000],
    }

def extract_row_features(url, html):
    url = safe_text(url)
    html = safe_text(html)

    p = urlparse(url)
    host = safe_text(p.netloc).lower()
    path = safe_text(p.path).lower()
    query = safe_text(p.query).lower()
    full_lower = url.lower()

    host_tok = host_tokens(host)

    html_sig = extract_html_signals(html)
    text_blob = html_sig["text_blob"]

    brand_tokens_found = [b for b in BRAND_TOKENS if b in text_blob]
    host_brand_overlap = sum(1 for b in brand_tokens_found if any(tok in host for tok in b.split()))

    suspicious_path_hits = sum(1 for t in SUSPICIOUS_PATH_TOKENS if t in path or t in query or t in full_lower)

    redirect_like = int(any(k in query for k in ["redirect", "redir", "continue", "url=", "next=", "return=", "dest="]))
    many_query_params = int(len(parse_qs(p.query)) >= 4)

    is_hosted = any_host_contains(host, HOSTED_PLATFORMS)
    is_short = any_host_contains(host, SHORTLINK_HOSTS)
    is_storage = any_host_contains(host, STORAGE_HOSTS)

    login_on_neutral = int(
        html_sig["n_pw"] > 0 and
        html_sig["brand_hits_html"] > 0 and
        is_hosted == 1
    )

    brand_host_mismatch = int(
        html_sig["brand_hits_html"] > 0 and
        host_brand_overlap == 0
    )

    high_login_low_brand_overlap = int(
        html_sig["login_hits_html"] >= 2 and
        html_sig["brand_hits_html"] >= 1 and
        host_brand_overlap == 0
    )

    generic_host_with_login = int(
        (is_hosted or is_storage or is_short) and
        (html_sig["n_pw"] > 0 or html_sig["login_hits_html"] >= 2)
    )

    social_brand_mismatch = int(
        any(b in text_blob for b in ["facebook", "instagram", "whatsapp", "telegram", "meta"]) and
        not any(x in host for x in ["facebook", "instagram", "whatsapp", "telegram", "meta"])
    )

    finance_brand_mismatch = int(
        any(b in text_blob for b in ["bank", "paypal", "visa", "mastercard", "chase", "coinbase", "binance", "robinhood"]) and
        not any(x in host for x in ["bank", "paypal", "visa", "mastercard", "chase", "coinbase", "binance", "robinhood"])
    )

    forms_brand_mismatch = int(
        any(b in text_blob for b in ["forms", "microsoft forms", "office", "outlook", "microsoft"]) and
        not any(x in host for x in ["microsoft", "office", "outlook", "forms"])
    )

    return [
        is_hosted,
        is_short,
        is_storage,
        html_sig["n_forms"],
        html_sig["n_inputs"],
        html_sig["n_pw"],
        html_sig["brand_hits_html"],
        html_sig["login_hits_html"],
        html_sig["has_submit_text"],
        html_sig["has_form_action"],
        html_sig["form_action_count"],
        html_sig["form_action_external_like"],
        suspicious_path_hits,
        redirect_like,
        many_query_params,
        host_brand_overlap,
        brand_host_mismatch,
        high_login_low_brand_overlap,
        generic_host_with_login,
        login_on_neutral,
        social_brand_mismatch,
        finance_brand_mismatch,
        forms_brand_mismatch,
    ]

FEATURE_NAMES = [
    "tgt_is_hosted_platform",
    "tgt_is_shortlink_host",
    "tgt_is_storage_host",
    "tgt_n_forms",
    "tgt_n_inputs",
    "tgt_n_password_inputs",
    "tgt_brand_hits_html",
    "tgt_login_hits_html",
    "tgt_has_submit_text",
    "tgt_has_form_action",
    "tgt_form_action_count",
    "tgt_form_action_external_like",
    "tgt_suspicious_path_hits",
    "tgt_redirect_like_query",
    "tgt_many_query_params",
    "tgt_host_brand_overlap",
    "tgt_brand_host_mismatch",
    "tgt_high_login_low_brand_overlap",
    "tgt_generic_host_with_login",
    "tgt_login_on_neutral_host",
    "tgt_social_brand_mismatch",
    "tgt_finance_brand_mismatch",
    "tgt_forms_brand_mismatch",
]

def main():
    df = pd.read_parquet(DATA_PATH)
    rows = []

    for _, r in df.iterrows():
        rows.append(extract_row_features(r["url"], r["html"]))

    X = np.array(rows, dtype=np.float32)
    np.save(TARGETED_NPY, X)

    with open(TARGETED_JSON, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print("Saved:", TARGETED_NPY, X.shape)
    print("Saved:", TARGETED_JSON)
    print("Feature count:", len(FEATURE_NAMES))

if __name__ == "__main__":
    main()