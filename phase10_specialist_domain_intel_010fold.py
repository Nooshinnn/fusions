import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("phase20_branch_cv_outputs")
DOMAIN_DIR = Path("phase08_domain_intel_folds")
PHASE06_OOF_PATH = Path("phase06_multichannel_joint_outputs/phase06_multichannel_oof_predictions.csv")

OUT_DIR = Path("phase10_specialist_domain_outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SPLITS = 10
SEED = 42

BATCH_SIZE = 128
EPOCHS = 25
PATIENCE = 5
LR = 8e-4

MAIN_THRESHOLD = 0.60
UNCERTAIN_LOW = 0.20
UNCERTAIN_HIGH = 0.80

# ============================================================
# TAB FEATURES
# ============================================================
def build_tab_features(url_df, html_df):
    df = pd.DataFrame({
        "row_index": html_df["row_index"],
        "y_true": html_df["y_true"].astype(int),
        "html_prob": html_df["html_prob"],
        "url_textcnn_prob": url_df["url_textcnn_prob"],
        "url_rf_prob": url_df["url_rf_prob"],
    })

    probs = ["html_prob", "url_textcnn_prob", "url_rf_prob"]

    df["mean"] = df[probs].mean(axis=1)
    df["max"] = df[probs].max(axis=1)
    df["min"] = df[probs].min(axis=1)
    df["std"] = df[probs].std(axis=1)

    df["diff_html_url"] = np.abs(df["html_prob"] - df["url_textcnn_prob"])
    df["diff_html_rf"] = np.abs(df["html_prob"] - df["url_rf_prob"])

    return df

TAB_COLS = [
    "html_prob","url_textcnn_prob","url_rf_prob",
    "mean","max","min","std",
    "diff_html_url","diff_html_rf"
]

# ============================================================
# DATASET
# ============================================================
class SpecialistDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

# ============================================================
# MODEL
# ============================================================
class SpecialistMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ============================================================
# TRAIN
# ============================================================
def train_epoch(model, loader, opt):
    model.train()
    total = 0
    for X,y in loader:
        X,y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(X)
        loss = nn.functional.binary_cross_entropy_with_logits(logits,y)
        loss.backward()
        opt.step()
        total += loss.item()*X.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for X,y in loader:
        X = X.to(DEVICE)
        logits = model(X)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p)
        ys.extend(y.numpy())
    return np.array(ys), np.array(probs)

# ============================================================
# MAIN
# ============================================================
def main():

    phase06_oof = pd.read_csv(PHASE06_OOF_PATH)

    all_preds = []
    metrics_all = []

    for fold in range(1, N_SPLITS+1):

        print(f"\n===== PHASE 10 FOLD {fold} =====")

        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        dom_dir = DOMAIN_DIR / f"fold_{fold:02d}"

        train_url = pd.read_csv(fold_dir/"train_url_outputs.csv")
        test_url = pd.read_csv(fold_dir/"test_url_outputs.csv")
        train_html = pd.read_csv(fold_dir/"train_html_outputs.csv")
        test_html = pd.read_csv(fold_dir/"test_html_outputs.csv")

        train_tab = build_tab_features(train_url, train_html)
        test_tab = build_tab_features(test_url, test_html)

        train_url_feat = np.load(fold_dir/"train_url_features.npy")
        test_url_feat = np.load(fold_dir/"test_url_features.npy")
        train_html_feat = np.load(fold_dir/"train_html_features.npy")
        test_html_feat = np.load(fold_dir/"test_html_features.npy")

        train_dom = np.load(dom_dir/"train_domain_intel.npy")
        test_dom = np.load(dom_dir/"test_domain_intel.npy")

        train_main = phase06_oof.iloc[train_tab.index]["multichannel_prob"].values
        test_main = phase06_oof.iloc[test_tab.index]["multichannel_prob"].values

        # FINAL FEATURE STACK
        X_train = np.concatenate([
            train_tab[TAB_COLS].values,
            train_url_feat,
            train_html_feat,
            train_dom,
            train_main.reshape(-1,1)
        ], axis=1)

        X_test = np.concatenate([
            test_tab[TAB_COLS].values,
            test_url_feat,
            test_html_feat,
            test_dom,
            test_main.reshape(-1,1)
        ], axis=1)

        y_train = train_tab["y_true"].values
        y_test = test_tab["y_true"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=y_train)

        train_ds = SpecialistDataset(X_train[tr_idx], y_train[tr_idx])
        val_ds = SpecialistDataset(X_train[val_idx], y_train[val_idx])
        test_ds = SpecialistDataset(X_test, y_test)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = SpecialistMLP(X_train.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        best = None
        best_auc = -1
        wait = 0

        for ep in range(EPOCHS):
            loss = train_epoch(model, train_dl, opt)
            yv, pv = predict(model, val_dl)
            auc = average_precision_score(yv, pv)

            print(f"epoch {ep} loss {loss:.4f} pr_auc {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

        model.load_state_dict(best)

        yt, pt = predict(model, test_dl)

        # routing
        final = test_main.copy()
        mask = (test_main >= UNCERTAIN_LOW) & (test_main <= UNCERTAIN_HIGH)
        final[mask] = pt[mask]

        m = compute_metrics(yt, final, MAIN_THRESHOLD)
        metrics_all.append(m)

        print("FOLD METRICS:", m)

    df = pd.DataFrame(metrics_all)
    print("\n===== PHASE 10 SUMMARY =====")
    print(df.mean())

if __name__ == "__main__":
    main()
