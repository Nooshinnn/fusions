import json
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
PARQUET_PATH     = "phreshphish_balanced.parquet"
CHAR_X_PATH      = "html_char_X.npy"
WORD_X_PATH      = "html_word_X.npy"
WORD_VOCAB_PATH  = "html_word_vocab.json"
LABELS_PATH      = "html_labels.npy"
FEAT_X_PATH      = "html_struct_features_v2.npy"

OUT_DIR          = Path("phase13_outputs")
OUT_DIR.mkdir(exist_ok=True)

CHAR_EMBED_DIM   = 32
WORD_EMBED_DIM   = 64
NUM_FILTERS      = 64
DROPOUT          = 0.5

EPOCHS           = 20
BATCH_SIZE       = 64
NUM_WORKERS      = 0
LR               = 1e-3
GRAD_CLIP        = 5.0
RANDOM_SEED      = 42
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================
# LOAD
# ============================================================
def load_cached_data():
    char_X = np.load(CHAR_X_PATH)
    word_X = np.load(WORD_X_PATH)
    labels = np.load(LABELS_PATH).astype(int)
    feat_X = np.load(FEAT_X_PATH).astype(np.float32)

    with open(WORD_VOCAB_PATH, "r", encoding="utf-8") as f:
        token2idx = json.load(f)

    word_vocab_size = max(token2idx.values()) + 1
    char_vocab_size = int(char_X.max()) + 1

    return char_X, word_X, feat_X, labels, char_vocab_size, word_vocab_size

def load_df():
    df = pd.read_parquet(PARQUET_PATH).copy()
    if df["label"].dtype == object:
        df["label_num"] = df["label"].map({"benign": 0, "phish": 1})
    else:
        df["label_num"] = df["label"].astype(int)
    return df

# ============================================================
# DATASET
# ============================================================
class FusionDataset(Dataset):
    def __init__(self, char_X, word_X, feat_X, labels, row_indices):
        self.char_X = char_X
        self.word_X = word_X
        self.feat_X = feat_X
        self.labels = labels
        self.row_indices = row_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        xc = torch.tensor(self.char_X[i], dtype=torch.long)
        xw = torch.tensor(self.word_X[i], dtype=torch.long)
        xf = torch.tensor(self.feat_X[i], dtype=torch.float32)
        y  = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        rid = torch.tensor(int(self.row_indices[i]), dtype=torch.long)
        return xc, xw, xf, y, rid

# ============================================================
# MODEL
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

class FusionCNN(nn.Module):
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
            nn.Dropout(DROPOUT),
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
# TRAIN / PREDICT
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for xc, xw, xf, y, _ in loader:
        xc, xw, xf, y = xc.to(DEVICE), xw.to(DEVICE), xf.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xc, xw, xf)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * xc.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict_with_features(model, loader):
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

    feats = np.concatenate(feats, axis=0)
    return np.array(trues), np.array(preds), np.array(probs), feats, np.array(row_ids)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Matthews CC": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "Log Loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("Using device:", DEVICE)

    char_X, word_X, feat_X, labels, char_vocab_size, word_vocab_size = load_cached_data()
    df = load_df()

    scaler = StandardScaler()
    feat_X_scaled = scaler.fit_transform(feat_X)

    ds = FusionDataset(
        char_X, word_X, feat_X_scaled, labels, np.arange(len(labels))
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_eval = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = FusionCNN(
        char_vocab_size=char_vocab_size,
        word_vocab_size=word_vocab_size,
        feat_dim=feat_X.shape[1],
        nf=NUM_FILTERS
    ).to(DEVICE)

    n_neg = int((labels == 0).sum())
    n_pos = int((labels == 1).sum())
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, dl, optimizer, criterion)
        print(f"Epoch {epoch:02d} | train_loss={loss:.4f}")

    y_true, y_pred, y_prob, feats, row_ids = predict_with_features(model, dl_eval)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\nFINAL HTML FULL-DATA METRICS")
    for k, v in metrics.items():
        if k != "Confusion Matrix":
            print(f"{k:<20} {v:.4f}")

    probs_full = np.zeros(len(df), dtype=np.float32)
    preds_full = np.zeros(len(df), dtype=np.int32)
    feats_full = np.zeros((len(df), feats.shape[1]), dtype=np.float32)

    probs_full[row_ids] = y_prob
    preds_full[row_ids] = y_pred
    feats_full[row_ids] = feats

    np.save(OUT_DIR / "final_html_probs.npy", probs_full)
    np.save(OUT_DIR / "final_html_preds.npy", preds_full)
    np.save(OUT_DIR / "final_html_features.npy", feats_full)

    out_df = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "url": df["url"].astype(str).values,
        "y_true": labels,
        "final_html_prob": probs_full,
        "final_html_pred": preds_full,
    })
    out_df.to_csv(OUT_DIR / "final_html_predictions.csv", index=False)

    with open(OUT_DIR / "final_html_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "final_html_probs.npy")
    print(OUT_DIR / "final_html_preds.npy")
    print(OUT_DIR / "final_html_features.npy")
    print(OUT_DIR / "final_html_predictions.csv")
    print("Feature shape:", feats_full.shape)

if __name__ == "__main__":
    main()