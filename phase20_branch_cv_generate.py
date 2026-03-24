import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "final_multimodal_dataset.parquet"
RAW_HTML_PATH = "phreshphish_balanced.parquet"

DOM_CACHE_DIR = "cache_domurlbert_new/cache_domurlbert_new"
DOM_META = "meta.json"
DOM_EMB = "embeddings.npy"

CHAR_X_PATH = "html_char_X.npy"
WORD_X_PATH = "html_word_X.npy"
WORD_VOCAB_PATH = "html_word_vocab.json"
LABELS_PATH = "html_labels.npy"
HTML_FEAT_PATH = "html_struct_features_v2.npy"

OUT_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# URL
URL_BATCH = 128
URL_EPOCHS = 12
URL_LR = 1e-3
URL_WD = 1e-4
URL_NUM_FILTERS = 128
URL_KERNELS = (3, 5, 7)
URL_DROPOUT = 0.5

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 30,
    "max_features": "sqrt",
    "criterion": "gini",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "random_state": 42,
    "n_jobs": -1,
}

# HTML
CHAR_EMBED_DIM = 32
WORD_EMBED_DIM = 64
HTML_NUM_FILTERS = 64
HTML_DROPOUT = 0.5
HTML_BATCH = 64
HTML_EPOCHS = 12
HTML_LR = 1e-3
HTML_WD = 1e-4

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LOADERS
# ============================================================
def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dom_cache(cache_dir, n_expected):
    meta = load_meta(os.path.join(cache_dir, DOM_META))
    n = int(meta["n_samples"])
    L = int(meta["max_tokens"])
    H = int(meta["hidden_size"])
    dtype = np.float16 if meta["dtype"] == "float16" else np.float32
    if n != n_expected:
        raise ValueError(f"DOM cache mismatch: {n} vs {n_expected}")
    emb = np.memmap(os.path.join(cache_dir, DOM_EMB), dtype=dtype, mode="r", shape=(n, L, H))
    return meta, emb

def load_main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    y = (df["label"].astype(str).str.lower() == "phish").astype(int).values
    url_feature_cols = [c for c in df.columns if c.startswith("url_")]
    X_rf = df[url_feature_cols].fillna(0).values.astype(np.float32)
    return df, y, X_rf

def load_html_arrays():
    char_X = np.load(CHAR_X_PATH)
    word_X = np.load(WORD_X_PATH)
    labels = np.load(LABELS_PATH).astype(int)
    feat_X = np.load(HTML_FEAT_PATH).astype(np.float32)
    with open(WORD_VOCAB_PATH, "r", encoding="utf-8") as f:
        token2idx = json.load(f)
    word_vocab_size = max(token2idx.values()) + 1
    char_vocab_size = int(char_X.max()) + 1
    return char_X, word_X, feat_X, labels, char_vocab_size, word_vocab_size

# ============================================================
# DATASETS
# ============================================================
class URLDataset(Dataset):
    def __init__(self, emb_mm, labels, indices):
        self.emb_mm = emb_mm
        self.labels = labels.astype(np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = torch.from_numpy(np.array(self.emb_mm[idx], copy=True)).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        rid = torch.tensor(idx, dtype=torch.long)
        return x, y, rid

class HTMLDataset(Dataset):
    def __init__(self, char_X, word_X, feat_X, labels, indices):
        self.char_X = char_X
        self.word_X = word_X
        self.feat_X = feat_X
        self.labels = labels.astype(np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return (
            torch.tensor(self.char_X[idx], dtype=torch.long),
            torch.tensor(self.word_X[idx], dtype=torch.long),
            torch.tensor(self.feat_X[idx], dtype=torch.float32),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )

# ============================================================
# URL MODEL
# ============================================================
class URLTextCNN(nn.Module):
    def __init__(self, hidden_size, num_filters=128, kernels=(3,5,7), dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = num_filters * len(kernels)
        self.fc = nn.Linear(self.feature_dim, 2)

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            feats.append(torch.max(h, dim=2).values)
        z = torch.cat(feats, dim=1)
        z = self.dropout(z)
        logits = self.fc(z)
        if return_features:
            return logits, z
        return logits

# ============================================================
# HTML MODEL
# ============================================================
class CNNBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, nf=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        def conv(k):
            return nn.Sequential(
                nn.Conv1d(embed_dim, nf, kernel_size=k, padding=k // 2),
                nn.ReLU()
            )
        self.conv3 = conv(3)
        self.conv5 = conv(5)
        self.conv7 = conv(7)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        b3 = torch.max(self.conv3(x), dim=2).values
        b5 = torch.max(self.conv5(x), dim=2).values
        b7 = torch.max(self.conv7(x), dim=2).values
        return torch.cat([b3, b5, b7], dim=1)

class HTMLFusionCNN(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, feat_dim, nf=64):
        super().__init__()
        self.char_branch = CNNBranch(char_vocab_size, CHAR_EMBED_DIM, nf=nf)
        self.word_branch = CNNBranch(word_vocab_size, WORD_EMBED_DIM, nf=nf)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.feature_dim = nf * 3 * 2 + 32
        self.classifier = nn.Sequential(
            nn.Dropout(HTML_DROPOUT),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 1)
        )

    def forward(self, xc, xw, xf, return_features=False):
        c = self.char_branch(xc)
        w = self.word_branch(xw)
        f = self.feat_branch(xf)
        z = torch.cat([c, w, f], dim=1)
        logits = self.classifier(z).squeeze(1)
        if return_features:
            return logits, z
        return logits

# ============================================================
# TRAIN/PREDICT HELPERS
# ============================================================
def train_url_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_url(model, loader):
    model.eval()
    probs, preds, trues, feats, row_ids = [], [], [], [], []
    for X, y, rid in loader:
        X = X.to(DEVICE)
        logits, z = model(X, return_features=True)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        probs.extend(prob.tolist())
        preds.extend(pred.tolist())
        trues.extend(y.numpy().tolist())
        feats.append(z.cpu().numpy())
        row_ids.extend(rid.numpy().tolist())
    return np.array(trues), np.array(preds), np.array(probs), np.concatenate(feats, axis=0), np.array(row_ids)

def train_html_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for xc, xw, xf, y, _ in loader:
        xc, xw, xf, y = xc.to(DEVICE), xw.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xc, xw, xf)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * xc.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_html(model, loader):
    model.eval()
    probs, preds, trues, feats, row_ids = [], [], [], [], []
    for xc, xw, xf, y, rid in loader:
        xc, xw, xf = xc.to(DEVICE), xw.to(DEVICE), xf.to(DEVICE)
        logits, z = model(xc, xw, xf, return_features=True)
        prob = torch.sigmoid(logits).cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        probs.extend(prob.tolist())
        preds.extend(pred.tolist())
        trues.extend(y.numpy().tolist())
        feats.append(z.cpu().numpy())
        row_ids.extend(rid.numpy().tolist())
    return np.array(trues), np.array(preds), np.array(probs), np.concatenate(feats, axis=0), np.array(row_ids)

# ============================================================
# MAIN
# ============================================================
def main():
    print("Using device:", DEVICE)

    df, y, X_rf = load_main()
    char_X, word_X, feat_X, html_labels, char_vocab_size, word_vocab_size = load_html_arrays()
    if not np.array_equal(y, html_labels):
        raise ValueError("Label mismatch between main dataset and HTML arrays")

    dom_meta, dom_emb = load_dom_cache(DOM_CACHE_DIR, len(df))
    hidden_size = int(dom_meta["hidden_size"])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print("\n" + "=" * 70)
        print(f"OUTER FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        fold_dir = OUT_DIR / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True, parents=True)

        # ---------------- URL agent ----------------
        url_train_ds = URLDataset(dom_emb, y, train_idx)
        url_test_ds = URLDataset(dom_emb, y, test_idx)

        url_train_loader = DataLoader(url_train_ds, batch_size=URL_BATCH, shuffle=True)
        url_eval_train_loader = DataLoader(url_train_ds, batch_size=URL_BATCH, shuffle=False)
        url_test_loader = DataLoader(url_test_ds, batch_size=URL_BATCH, shuffle=False)

        url_model = URLTextCNN(
            hidden_size=hidden_size,
            num_filters=URL_NUM_FILTERS,
            kernels=URL_KERNELS,
            dropout=URL_DROPOUT
        ).to(DEVICE)

        url_opt = torch.optim.AdamW(url_model.parameters(), lr=URL_LR, weight_decay=URL_WD)
        url_crit = nn.CrossEntropyLoss()

        for epoch in range(1, URL_EPOCHS + 1):
            loss = train_url_epoch(url_model, url_train_loader, url_opt, url_crit)
            print(f"[Fold {fold}] URL epoch {epoch:02d} | loss={loss:.4f}")

        _, _, url_train_prob, url_train_feat, url_train_row = predict_url(url_model, url_eval_train_loader)
        _, _, url_test_prob, url_test_feat, url_test_row = predict_url(url_model, url_test_loader)

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_rf[train_idx], y[train_idx])
        rf_train_prob = rf.predict_proba(X_rf[train_idx])[:, 1]
        rf_test_prob = rf.predict_proba(X_rf[test_idx])[:, 1]

        pd.DataFrame({
            "row_index": url_train_row,
            "url": df.iloc[url_train_row]["url"].astype(str).values,
            "y_true": y[url_train_row],
            "url_textcnn_prob": url_train_prob,
            "url_rf_prob": rf_train_prob
        }).sort_values("row_index").to_csv(fold_dir / "train_url_outputs.csv", index=False)

        pd.DataFrame({
            "row_index": url_test_row,
            "url": df.iloc[url_test_row]["url"].astype(str).values,
            "y_true": y[url_test_row],
            "url_textcnn_prob": url_test_prob,
            "url_rf_prob": rf_test_prob
        }).sort_values("row_index").to_csv(fold_dir / "test_url_outputs.csv", index=False)

        np.save(fold_dir / "train_url_features.npy", url_train_feat.astype(np.float32))
        np.save(fold_dir / "test_url_features.npy", url_test_feat.astype(np.float32))

        # ---------------- HTML agent ----------------
        scaler = StandardScaler()
        feat_train_scaled = scaler.fit_transform(feat_X[train_idx])
        feat_test_scaled = scaler.transform(feat_X[test_idx])

        # rebuild full-size arrays for simple indexing in dataset
        feat_full = np.zeros_like(feat_X, dtype=np.float32)
        feat_full[train_idx] = feat_train_scaled
        feat_full[test_idx] = feat_test_scaled

        html_train_ds = HTMLDataset(char_X, word_X, feat_full, y, train_idx)
        html_test_ds = HTMLDataset(char_X, word_X, feat_full, y, test_idx)

        html_train_loader = DataLoader(html_train_ds, batch_size=HTML_BATCH, shuffle=True)
        html_eval_train_loader = DataLoader(html_train_ds, batch_size=HTML_BATCH, shuffle=False)
        html_test_loader = DataLoader(html_test_ds, batch_size=HTML_BATCH, shuffle=False)

        html_model = HTMLFusionCNN(
            char_vocab_size=char_vocab_size,
            word_vocab_size=word_vocab_size,
            feat_dim=feat_X.shape[1],
            nf=HTML_NUM_FILTERS
        ).to(DEVICE)

        n_neg = int((y[train_idx] == 0).sum())
        n_pos = int((y[train_idx] == 1).sum())
        pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

        html_opt = torch.optim.Adam(html_model.parameters(), lr=HTML_LR, weight_decay=HTML_WD)
        html_crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        for epoch in range(1, HTML_EPOCHS + 1):
            loss = train_html_epoch(html_model, html_train_loader, html_opt, html_crit)
            print(f"[Fold {fold}] HTML epoch {epoch:02d} | loss={loss:.4f}")

        _, _, html_train_prob, html_train_feat, html_train_row = predict_html(html_model, html_eval_train_loader)
        _, _, html_test_prob, html_test_feat, html_test_row = predict_html(html_model, html_test_loader)

        pd.DataFrame({
            "row_index": html_train_row,
            "url": df.iloc[html_train_row]["url"].astype(str).values,
            "y_true": y[html_train_row],
            "html_prob": html_train_prob
        }).sort_values("row_index").to_csv(fold_dir / "train_html_outputs.csv", index=False)

        pd.DataFrame({
            "row_index": html_test_row,
            "url": df.iloc[html_test_row]["url"].astype(str).values,
            "y_true": y[html_test_row],
            "html_prob": html_test_prob
        }).sort_values("row_index").to_csv(fold_dir / "test_html_outputs.csv", index=False)

        np.save(fold_dir / "train_html_features.npy", html_train_feat.astype(np.float32))
        np.save(fold_dir / "test_html_features.npy", html_test_feat.astype(np.float32))

        print(f"[Fold {fold}] saved branch outputs to {fold_dir}")

if __name__ == "__main__":
    main()