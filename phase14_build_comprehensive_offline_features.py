import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qsl

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

DATA_PATH = "final_multimodal_dataset.parquet"

OUT_DIR = Path("phase14_outputs")
OUT_DIR.mkdir(exist_ok=True)

OUT_NPY = OUT_DIR / "phase14_comprehensive_offline_features.npy"
OUT_JSON = OUT_DIR / "phase14_comprehensive_offline_feature_names.json"
OUT_CSV = OUT_DIR / "phase14_comprehensive_offline_feature_table.csv"

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "work", "shop", "support", "live", "info",
    "buzz", "monster", "cam", "fit", "uno", "mom", "rest", "autos",
    "cfd", "gq", "tk", "ml", "ga", "cf", "sbs"
}

HOSTING_HINTS = [
    "pages.dev", "web.app", "firebaseapp.com", "r2.dev", "ipfs.io",
    "vercel.app", "webflow.io", "github.io", "blogspot", "wixsite",
    "weebly", "000webhost", "teemill", "highlevel", "azurewebsites",
    "netlify", "cloudfront", "workers.dev"
]

SHORTENER_HINTS = [
    "bit.ly", "tinyurl.com", "qrco.de", "rb.gy", "t.co", "goo.gl", "cutt.ly"
]

SUSPICIOUS_PATH_TOKENS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "password", "wallet", "recover",
    "billing", "admin", "support", "check", "session", "auth",
    "unlock", "validate", "security", "suspend", "limited"
]

URGENCY_WORDS = [
    "urgent", "immediately", "verify", "suspended", "limited",
    "action required", "security alert", "confirm", "update now"
]

ACCOUNT_WORDS = [
    "login", "sign in", "password", "account", "security",
    "billing", "wallet", "payment", "support", "recover"
]

BRAND_ALIASES = {
    "facebook": ["facebook", "fb", "meta"],
    "instagram": ["instagram", "insta", "ig"],
    "whatsapp": ["whatsapp", "wa"],
    "microsoft": ["microsoft", "office", "outlook", "live", "onedrive", "sharepoint", "forms"],
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
    "dhl": ["dhl"],
    "fedex": ["fedex"],
    "dropbox": ["dropbox"],
}

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def is_ipv4(host: str) -> int:
    parts = host.split(".")
    if len(parts) != 4:
        return 0
    try:
        nums = [int(p) for p in parts]
        return int(all(0 <= n <= 255 for n in nums))
    except Exception:
        return 0

def count_repeated_char_runs(s: str) -> int:
    cnt = 0
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            if run == 3:
                cnt += 1
        else:
            run = 1
    return cnt

def tokenize_text(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

def count_token_hits(text: str, vocab):
    text = text.lower()
    return sum(1 for v in vocab if v in text)

def detect_brands(text: str):
    text = text.lower()
    found = []
    for brand, aliases in BRAND_ALIASES.items():
        if any(a in text for a in aliases):
            found.append(brand)
    return found

def brand_location_flags(host: str, subdomain: str, path_query: str, brands):
    in_host = 0
    in_subdomain = 0
    in_path = 0
    for b in brands:
        aliases = BRAND_ALIASES[b]
        if any(a in host for a in aliases):
            in_host = 1
        if any(a in subdomain for a in aliases):
            in_subdomain = 1
        if any(a in path_query for a in aliases):
            in_path = 1
    return in_host, in_subdomain, in_path

def get_text_stats(text: str):
    text = safe_text(text)
    if not text:
        return 0, 0.0, 0.0
    letters = sum(c.isalpha() for c in text)
    uppers = sum(c.isupper() for c in text)
    digits = sum(c.isdigit() for c in text)
    upper_ratio = uppers / max(letters, 1)
    digit_ratio = digits / max(len(text), 1)
    return len(text), upper_ratio, digit_ratio

def max_dom_depth(tag, current=0):
    if not hasattr(tag, "contents"):
        return current
    child_depths = [max_dom_depth(c, current + 1) for c in tag.contents if getattr(c, "name", None) is not None]
    return max([current] + child_depths)

FEATURE_NAMES = [
    # URL lexical
    "p14_url_len", "p14_host_len", "p14_path_len", "p14_query_len",
    "p14_num_subdomains", "p14_num_dots", "p14_num_hyphens", "p14_num_digits_host",
    "p14_digit_ratio_host", "p14_entropy_host", "p14_entropy_path",
    "p14_has_ip_host", "p14_has_punycode", "p14_tld_is_suspicious", "p14_tld_len",
    "p14_has_hosting_hint", "p14_has_shortener_hint",
    "p14_num_path_tokens", "p14_num_query_params", "p14_query_key_count",
    "p14_suspicious_path_hits", "p14_has_php", "p14_has_html_ext",
    "p14_has_login_word", "p14_has_secure_word", "p14_has_verify_word",
    "p14_has_account_word", "p14_has_update_word",
    "p14_brand_hits_host", "p14_brand_hits_path", "p14_brand_hits_full_url",
    "p14_repeated_runs_host", "p14_repeated_runs_path",
    "p14_https_scheme", "p14_has_at_symbol", "p14_has_encoded_chars",

    # HTML structure
    "p14_has_title", "p14_title_len", "p14_visible_text_len", "p14_visible_text_upper_ratio", "p14_visible_text_digit_ratio",
    "p14_num_forms", "p14_num_inputs", "p14_num_password_inputs", "p14_num_hidden_inputs",
    "p14_num_email_inputs", "p14_num_tel_inputs", "p14_num_submit_buttons",
    "p14_num_buttons", "p14_num_iframes", "p14_num_scripts", "p14_num_inline_scripts",
    "p14_num_external_scripts", "p14_num_links", "p14_num_external_links",
    "p14_num_empty_links", "p14_num_js_links", "p14_num_mailto_links",
    "p14_num_tel_links", "p14_num_images", "p14_num_meta", "p14_num_base_tags",
    "p14_num_canonical", "p14_has_meta_refresh", "p14_has_favicon",
    "p14_external_favicon", "p14_dom_depth_max",

    # Form analysis
    "p14_num_forms_post", "p14_num_forms_get", "p14_num_forms_empty_action",
    "p14_num_forms_external_action", "p14_num_forms_js_action",

    # JavaScript patterns
    "p14_js_eval_hits", "p14_js_atob_hits", "p14_js_fromchar_hits",
    "p14_js_docwrite_hits", "p14_js_winloc_hits", "p14_js_toploc_hits",
    "p14_js_fetch_hits", "p14_js_xhr_hits", "p14_js_open_hits",

    # Text / semantic cues
    "p14_urgency_hits", "p14_account_word_hits", "p14_detected_brand_count",
    "p14_multi_brand_page", "p14_brand_in_host", "p14_brand_in_subdomain",
    "p14_brand_in_path", "p14_brand_mismatch",

    # Cross consistency
    "p14_form_brand_mismatch", "p14_password_brand_mismatch",
    "p14_suspicious_domain_and_form", "p14_suspicious_domain_and_password",
    "p14_hosting_hint_and_branded_page", "p14_external_form_action_and_login",
    "p14_external_links_ratio", "p14_external_scripts_ratio", "p14_forms_per_input_ratio"
]

def extract_row(url: str, html: str):
    url = safe_text(url)
    html = safe_text(html)

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    subdomain = ".".join(host.split(".")[:-2]) if len(host.split(".")) > 2 else ""
    suffix = host.split(".")[-1] if "." in host else ""

    path_tokens = tokenize_text(path)
    query_params = parse_qsl(query, keep_blank_values=True)
    query_keys = [k for k, _ in query_params]

    suspicious_domain = int(
        suffix in SUSPICIOUS_TLDS or
        any(h in host for h in HOSTING_HINTS) or
        is_ipv4(host) == 1 or
        "xn--" in host
    )

    # parse html
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = BeautifulSoup("", "html.parser")

    title = safe_text(soup.title.string) if soup.title and soup.title.string else ""
    visible_text = safe_text(soup.get_text(" ", strip=True))[:20000]

    text_len, text_upper_ratio, text_digit_ratio = get_text_stats(visible_text)

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    buttons = soup.find_all("button")
    iframes = soup.find_all("iframe")
    scripts = soup.find_all("script")
    links = soup.find_all("a")
    images = soup.find_all("img")
    metas = soup.find_all("meta")
    base_tags = soup.find_all("base")
    canonical_tags = soup.find_all("link", rel=lambda x: x and "canonical" in str(x).lower())

    password_inputs = soup.find_all("input", {"type": lambda x: x and str(x).lower() == "password"})
    hidden_inputs = soup.find_all("input", {"type": lambda x: x and str(x).lower() == "hidden"})
    email_inputs = soup.find_all("input", {"type": lambda x: x and str(x).lower() == "email"})
    tel_inputs = soup.find_all("input", {"type": lambda x: x and str(x).lower() == "tel"})
    submit_buttons = [
        t for t in soup.find_all(["input", "button"])
        if (t.get("type", "") or "").lower() in {"submit", "button"}
    ]

    inline_scripts = [s for s in scripts if not s.get("src")]
    external_scripts = [s for s in scripts if s.get("src")]

    external_links = 0
    empty_links = 0
    js_links = 0
    mailto_links = 0
    tel_links = 0
    for a in links:
        href = safe_text(a.get("href")).strip().lower()
        if href.startswith("http://") or href.startswith("https://"):
            external_links += 1
        if href in {"", "#"}:
            empty_links += 1
        if href.startswith("javascript:"):
            js_links += 1
        if href.startswith("mailto:"):
            mailto_links += 1
        if href.startswith("tel:"):
            tel_links += 1

    forms_post = 0
    forms_get = 0
    forms_empty_action = 0
    forms_external_action = 0
    forms_js_action = 0
    for f in forms:
        method = safe_text(f.get("method")).lower()
        action = safe_text(f.get("action")).strip().lower()
        if method == "post":
            forms_post += 1
        if method == "get":
            forms_get += 1
        if action == "":
            forms_empty_action += 1
        if action.startswith("http://") or action.startswith("https://"):
            forms_external_action += 1
        if action.startswith("javascript:"):
            forms_js_action += 1

    has_meta_refresh = int(any("refresh" == safe_text(m.get("http-equiv")).lower() for m in metas))

    favicon_tags = soup.find_all("link", rel=lambda x: x and "icon" in str(x).lower())
    has_favicon = int(len(favicon_tags) > 0)
    external_favicon = 0
    for f in favicon_tags:
        href = safe_text(f.get("href")).lower()
        if href.startswith("http://") or href.startswith("https://"):
            external_favicon = 1
            break

    html_lower = html.lower()

    js_eval_hits = html_lower.count("eval(")
    js_atob_hits = html_lower.count("atob(")
    js_fromchar_hits = html_lower.count("fromcharcode(")
    js_docwrite_hits = html_lower.count("document.write")
    js_winloc_hits = html_lower.count("window.location")
    js_toploc_hits = html_lower.count("top.location")
    js_fetch_hits = html_lower.count("fetch(")
    js_xhr_hits = html_lower.count("xmlhttprequest")
    js_open_hits = html_lower.count("window.open")

    combined_text = " ".join([title, visible_text, html_lower[:10000]])
    detected_brands = detect_brands(combined_text)
    brand_count = len(detected_brands)
    brand_in_host, brand_in_subdomain, brand_in_path = brand_location_flags(
        host, subdomain, path + " " + query, detected_brands
    )
    brand_mismatch = int(brand_count > 0 and brand_in_host == 0)
    form_brand_mismatch = int(len(forms) > 0 and brand_count > 0 and brand_in_host == 0)
    password_brand_mismatch = int(len(password_inputs) > 0 and brand_count > 0 and brand_in_host == 0)

    urgency_hits = count_token_hits(combined_text, URGENCY_WORDS)
    account_hits = count_token_hits(combined_text, ACCOUNT_WORDS)

    external_links_ratio = external_links / max(len(links), 1)
    external_scripts_ratio = len(external_scripts) / max(len(scripts), 1)
    forms_per_input_ratio = len(forms) / max(len(inputs), 1)

    login_words_in_url = int(any(t in url.lower() for t in ["login", "signin", "sign-in", "account", "verify", "secure"]))

    feats = [
        # URL lexical
        len(url), len(host), len(path), len(query),
        max(len(host.split(".")) - 2, 0) if "." in host else 0,
        url.count("."), url.count("-"), sum(c.isdigit() for c in host),
        sum(c.isdigit() for c in host) / max(len(host), 1),
        shannon_entropy(host), shannon_entropy(path),
        is_ipv4(host), int("xn--" in host), int(suffix in SUSPICIOUS_TLDS), len(suffix),
        int(any(h in host for h in HOSTING_HINTS)),
        int(any(h in host for h in SHORTENER_HINTS)),
        len(path_tokens), len(query_params), len(query_keys),
        count_token_hits(path + " " + query + " " + url.lower(), SUSPICIOUS_PATH_TOKENS),
        int(".php" in path), int(path.endswith(".html") or path.endswith(".htm")),
        int("login" in url.lower() or "signin" in url.lower() or "sign-in" in url.lower()),
        int("secure" in url.lower()), int("verify" in url.lower() or "verification" in url.lower()),
        int("account" in url.lower()), int("update" in url.lower()),
        sum(1 for b in BRAND_ALIASES if b in host),
        sum(1 for b in BRAND_ALIASES if b in (path + " " + query)),
        sum(1 for b in BRAND_ALIASES if b in url.lower()),
        count_repeated_char_runs(host), count_repeated_char_runs(path),
        int(parsed.scheme.lower() == "https"), int("@" in url), int("%" in url),

        # HTML structure
        int(bool(title)), len(title), text_len, text_upper_ratio, text_digit_ratio,
        len(forms), len(inputs), len(password_inputs), len(hidden_inputs),
        len(email_inputs), len(tel_inputs), len(submit_buttons),
        len(buttons), len(iframes), len(scripts), len(inline_scripts),
        len(external_scripts), len(links), external_links,
        empty_links, js_links, mailto_links,
        tel_links, len(images), len(metas), len(base_tags),
        len(canonical_tags), has_meta_refresh, has_favicon,
        external_favicon, max_dom_depth(soup, 0),

        # Form analysis
        forms_post, forms_get, forms_empty_action,
        forms_external_action, forms_js_action,

        # JavaScript patterns
        js_eval_hits, js_atob_hits, js_fromchar_hits,
        js_docwrite_hits, js_winloc_hits, js_toploc_hits,
        js_fetch_hits, js_xhr_hits, js_open_hits,

        # Text / semantic cues
        urgency_hits, account_hits, brand_count,
        int(brand_count >= 2), brand_in_host, brand_in_subdomain,
        brand_in_path, brand_mismatch,

        # Cross consistency
        form_brand_mismatch, password_brand_mismatch,
        int(suspicious_domain and len(forms) > 0),
        int(suspicious_domain and len(password_inputs) > 0),
        int(any(h in host for h in HOSTING_HINTS) and brand_count > 0),
        int(forms_external_action > 0 and login_words_in_url == 1),
        external_links_ratio, external_scripts_ratio, forms_per_input_ratio,
    ]

    return feats

def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    X = np.array([extract_row(u, h) for u, h in zip(df["url"], df["html"])], dtype=np.float32)

    np.save(OUT_NPY, X)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    out_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    out_df.insert(0, "url", df["url"].astype(str).values)
    out_df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_NPY, X.shape)
    print("Saved:", OUT_JSON)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
