import ssl
import socket
import json
import math
import time
import whois
import tldextract
import numpy as np
import pandas as pd
import dns.resolver

from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

DATA_PATH = "phreshphish_balanced.parquet"

OUT_DIR = Path("phase29_outputs")
OUT_DIR.mkdir(exist_ok=True)

OUT_NPY = OUT_DIR / "domain_intel_features.npy"
OUT_JSON = OUT_DIR / "domain_intel_feature_names.json"
OUT_CSV = OUT_DIR / "domain_intel_feature_table.csv"

DNS_TIMEOUT = 3.0
SSL_TIMEOUT = 3.0
SOCKET_TIMEOUT = 3.0

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "country", "gq", "tk", "ml", "ga", "cf", "work",
    "shop", "support", "live", "info", "buzz", "rest", "monster", "cam",
    "fit", "uno", "mom", "cfd", "autos", "sbs"
}

KNOWN_HOSTING_HINTS = [
    "vercel", "firebase", "pages.dev", "web.app", "r2.dev", "ipfs",
    "github.io", "webflow", "wixsite", "weebly", "000webhost", "netlify",
    "azurewebsites", "cloudfront", "blogspot", "teemill", "highlevel"
]

FEATURE_NAMES = [
    "di_has_whois",
    "di_has_creation_date",
    "di_has_expiration_date",
    "di_has_updated_date",
    "di_domain_age_days",
    "di_days_to_expiry",
    "di_days_since_update",
    "di_is_young_domain_lt_30d",
    "di_is_young_domain_lt_180d",
    "di_expires_soon_lt_30d",
    "di_has_dns_a",
    "di_has_dns_mx",
    "di_has_dns_ns",
    "di_has_dns_txt",
    "di_a_count",
    "di_mx_count",
    "di_ns_count",
    "di_txt_count",
    "di_dns_any_present",
    "di_ssl_ok",
    "di_ssl_days_to_expiry",
    "di_ssl_expiring_soon_lt_30d",
    "di_host_is_ip",
    "di_host_has_punycode",
    "di_tld_is_suspicious",
    "di_subdomain_depth",
    "di_subdomain_depth_ge_3",
    "di_host_has_hosting_hint",
    "di_has_port_in_url",
    "di_url_uses_https",
    "di_all_external_missing",
]

def safe_float(x, default=-1.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def now_utc():
    return datetime.now(timezone.utc)

def normalize_dt(x):
    if x is None:
        return None
    if isinstance(x, list):
        x = x[0] if len(x) > 0 else None
    if x is None:
        return None
    if not isinstance(x, datetime):
        return None
    if x.tzinfo is None:
        x = x.replace(tzinfo=timezone.utc)
    return x.astimezone(timezone.utc)

def parse_host(url: str):
    p = urlparse(str(url))
    return p, (p.hostname or "").lower()

def is_ip_host(host: str):
    try:
        socket.inet_aton(host)
        return 1
    except Exception:
        return 0

def get_whois_features(host: str):
    feats = {
        "has_whois": 0,
        "has_creation_date": 0,
        "has_expiration_date": 0,
        "has_updated_date": 0,
        "domain_age_days": -1.0,
        "days_to_expiry": -1.0,
        "days_since_update": -1.0,
        "is_young_domain_lt_30d": 0,
        "is_young_domain_lt_180d": 0,
        "expires_soon_lt_30d": 0,
    }

    try:
        w = whois.whois(host)
    except Exception:
        return feats

    feats["has_whois"] = 1

    now = now_utc()
    creation = normalize_dt(getattr(w, "creation_date", None))
    expiry = normalize_dt(getattr(w, "expiration_date", None))
    updated = normalize_dt(getattr(w, "updated_date", None))

    if creation is not None:
        age = max((now - creation).days, 0)
        feats["has_creation_date"] = 1
        feats["domain_age_days"] = float(age)
        feats["is_young_domain_lt_30d"] = int(age < 30)
        feats["is_young_domain_lt_180d"] = int(age < 180)

    if expiry is not None:
        dte = (expiry - now).days
        feats["has_expiration_date"] = 1
        feats["days_to_expiry"] = float(dte)
        feats["expires_soon_lt_30d"] = int(dte < 30)

    if updated is not None:
        dsu = max((now - updated).days, 0)
        feats["has_updated_date"] = 1
        feats["days_since_update"] = float(dsu)

    return feats

def dns_count(host: str, rtype: str):
    resolver = dns.resolver.Resolver()
    resolver.timeout = DNS_TIMEOUT
    resolver.lifetime = DNS_TIMEOUT
    try:
        ans = resolver.resolve(host, rtype)
        return len(ans), 1
    except Exception:
        return 0, 0

def get_dns_features(host: str):
    a_count, has_a = dns_count(host, "A")
    mx_count, has_mx = dns_count(host, "MX")
    ns_count, has_ns = dns_count(host, "NS")
    txt_count, has_txt = dns_count(host, "TXT")

    return {
        "has_dns_a": has_a,
        "has_dns_mx": has_mx,
        "has_dns_ns": has_ns,
        "has_dns_txt": has_txt,
        "a_count": float(a_count),
        "mx_count": float(mx_count),
        "ns_count": float(ns_count),
        "txt_count": float(txt_count),
        "dns_any_present": int(any([has_a, has_mx, has_ns, has_txt])),
    }

def get_ssl_features(host: str, port=443):
    feats = {
        "ssl_ok": 0,
        "ssl_days_to_expiry": -1.0,
        "ssl_expiring_soon_lt_30d": 0,
    }

    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
        if "notAfter" in cert:
            expiry = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
            now = now_utc()
            dte = (expiry - now).days
            feats["ssl_ok"] = 1
            feats["ssl_days_to_expiry"] = float(dte)
            feats["ssl_expiring_soon_lt_30d"] = int(dte < 30)
    except Exception:
        pass

    return feats

def row_features(url: str):
    p, host = parse_host(url)
    ext = tldextract.extract(host)

    whois_f = get_whois_features(host) if host else {}
    dns_f = get_dns_features(host) if host else {}
    ssl_f = get_ssl_features(host, p.port or 443) if host and (p.scheme == "https" or p.port == 443 or p.port is None) else {
        "ssl_ok": 0, "ssl_days_to_expiry": -1.0, "ssl_expiring_soon_lt_30d": 0
    }

    sub_depth = 0
    if ext.subdomain:
        sub_depth = len([x for x in ext.subdomain.split(".") if x])

    host_has_hosting_hint = int(any(h in host for h in KNOWN_HOSTING_HINTS))
    tld = ext.suffix.lower().split(".")[-1] if ext.suffix else ""
    tld_suspicious = int(tld in SUSPICIOUS_TLDS)

    all_external_missing = int(
        whois_f.get("has_whois", 0) == 0 and
        dns_f.get("dns_any_present", 0) == 0 and
        ssl_f.get("ssl_ok", 0) == 0
    )

    vals = [
        whois_f.get("has_whois", 0),
        whois_f.get("has_creation_date", 0),
        whois_f.get("has_expiration_date", 0),
        whois_f.get("has_updated_date", 0),
        whois_f.get("domain_age_days", -1.0),
        whois_f.get("days_to_expiry", -1.0),
        whois_f.get("days_since_update", -1.0),
        whois_f.get("is_young_domain_lt_30d", 0),
        whois_f.get("is_young_domain_lt_180d", 0),
        whois_f.get("expires_soon_lt_30d", 0),
        dns_f.get("has_dns_a", 0),
        dns_f.get("has_dns_mx", 0),
        dns_f.get("has_dns_ns", 0),
        dns_f.get("has_dns_txt", 0),
        dns_f.get("a_count", 0.0),
        dns_f.get("mx_count", 0.0),
        dns_f.get("ns_count", 0.0),
        dns_f.get("txt_count", 0.0),
        dns_f.get("dns_any_present", 0),
        ssl_f.get("ssl_ok", 0),
        ssl_f.get("ssl_days_to_expiry", -1.0),
        ssl_f.get("ssl_expiring_soon_lt_30d", 0),
        is_ip_host(host),
        int("xn--" in host),
        tld_suspicious,
        float(sub_depth),
        int(sub_depth >= 3),
        host_has_hosting_hint,
        int(p.port is not None),
        int(p.scheme.lower() == "https"),
        all_external_missing,
    ]
    return vals

def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    rows = []

    for i, url in enumerate(df["url"].astype(str).tolist(), start=1):
        rows.append(row_features(url))
        if i % 100 == 0:
            print(f"Processed {i}/{len(df)}")

    X = np.array(rows, dtype=np.float32)
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