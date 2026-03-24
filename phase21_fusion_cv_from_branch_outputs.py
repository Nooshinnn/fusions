import json
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR = Path("phase21_fusion_cv_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5
LR = 8e-4
WEIGHT_DECAY = 1e-4

# ============================================================
# DATASET / MODEL
# ============================================================
class FusionDataset(Dataset):
    def __init__(self, x_tab, x_url, x_html, y):
        self.x_tab = x_tab.astype(np.float32)
        self.x_url = x_url.astype(np.float32)
        self.x_html = x_html.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x_tab[i], dtype=torch.float32),
            torch.tensor(self.x_url[i], dtype=torch.float32),
            torch.tensor(self.x_html[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )

class FusionV3(nn.Module):
    def __init__(self, tab_dim, url_dim, html_dim):
        super().__init__()

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.url_proj = nn.Sequential(
            nn.Linear(url_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.html_proj = nn.Sequential(
            nn.Linear(html_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.interaction = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.url_to_html = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.html_to_url = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.gate = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_url, x_html):
        t = self.tab_proj(x_tab)
        u = self.url_proj(x_url)
        h = self.html_proj(x_html)

        inter = torch.cat([u, h, torch.abs(u - h), u * h], dim=1)
        inter = self.interaction(inter)

        u_new = u + self.html_to_url(inter)
        h_new = h + self.url_to_html(inter)

        gate = self.gate(torch.cat([u_new, h_new], dim=1))
        fused = gate * u_new + (1.0 - gate) * h_new

        z = torch.cat([t, u_new, h_new, fused], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

# ============================================================
# HELPERS
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
    }

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x_tab, x_url, x_html, y in loader:
        x_tab, x_url, x_html, y = x_tab.to(DEVICE), x_url.to(DEVICE), x_html.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tab, x_url, x_html)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x_tab.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs, trues = [], []
    for x_tab, x_url, x_html, y in loader:
        x_tab, x_url, x_html = x_tab.to(DEVICE), x_url.to(DEVICE), x_html.to(DEVICE)
        logits = model(x_tab, x_url, x_html)
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(probs)

# ============================================================
# MAIN
# ============================================================
def main():
    all_test_rows = []
    all_fold_metrics = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        print("\n" + "=" * 70)
        print(f"FUSION FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_url_feat = np.load(fold_dir / "train_url_features.npy").astype(np.float32)
        test_url_feat = np.load(fold_dir / "test_url_features.npy").astype(np.float32)
        train_html_feat = np.load(fold_dir / "train_html_features.npy").astype(np.float32)
        test_html_feat = np.load(fold_dir / "test_html_features.npy").astype(np.float32)

        train_tab = pd.DataFrame({
            "html_prob": train_html["html_prob"].values,
            "url_textcnn_prob": train_url["url_textcnn_prob"].values,
            "url_rf_prob": train_url["url_rf_prob"].values,
        })
        test_tab = pd.DataFrame({
            "html_prob": test_html["html_prob"].values,
            "url_textcnn_prob": test_url["url_textcnn_prob"].values,
            "url_rf_prob": test_url["url_rf_prob"].values,
        })

        for df_tab in [train_tab, test_tab]:
            df_tab["d_url_html"] = np.abs(df_tab["url_textcnn_prob"] - df_tab["html_prob"])
            df_tab["d_rf_html"] = np.abs(df_tab["url_rf_prob"] - df_tab["html_prob"])
            df_tab["d_url_rf"] = np.abs(df_tab["url_textcnn_prob"] - df_tab["url_rf_prob"])
            df_tab["mean_prob"] = df_tab[["html_prob", "url_textcnn_prob", "url_rf_prob"]].mean(axis=1)
            df_tab["max_prob"] = df_tab[["html_prob", "url_textcnn_prob", "url_rf_prob"]].max(axis=1)
            df_tab["min_prob"] = df_tab[["html_prob", "url_textcnn_prob", "url_rf_prob"]].min(axis=1)
            df_tab["std_prob"] = df_tab[["html_prob", "url_textcnn_prob", "url_rf_prob"]].std(axis=1)
            df_tab["html_conf"] = np.abs(df_tab["html_prob"] - 0.5)
            df_tab["url_conf"] = np.abs(df_tab["url_textcnn_prob"] - 0.5)
            df_tab["rf_conf"] = np.abs(df_tab["url_rf_prob"] - 0.5)
            df_tab["range_prob"] = df_tab["max_prob"] - df_tab["min_prob"]

        Xtab_train = train_tab.values.astype(np.float32)
        Xtab_test = test_tab.values.astype(np.float32)
        y_train = train_html["y_true"].astype(int).values
        y_test = test_html["y_true"].astype(int).values

        scaler_tab = StandardScaler()
        scaler_url = StandardScaler()
        scaler_html = StandardScaler()

        Xtab_train = scaler_tab.fit_transform(Xtab_train)
        Xtab_test = scaler_tab.transform(Xtab_test)
        Xurl_train = scaler_url.fit_transform(train_url_feat)
        Xurl_test = scaler_url.transform(test_url_feat)
        Xhtml_train = scaler_html.fit_transform(train_html_feat)
        Xhtml_test = scaler_html.transform(test_html_feat)

        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=y_train, random_state=SEED)

        ds_train = FusionDataset(Xtab_train[tr_idx], Xurl_train[tr_idx], Xhtml_train[tr_idx], y_train[tr_idx])
        ds_val = FusionDataset(Xtab_train[val_idx], Xurl_train[val_idx], Xhtml_train[val_idx], y_train[val_idx])
        ds_test = FusionDataset(Xtab_test, Xurl_test, Xhtml_test, y_test)

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = FusionV3(
            tab_dim=Xtab_train.shape[1],
            url_dim=Xurl_train.shape[1],
            html_dim=Xhtml_train.shape[1]
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        best_state = None
        best_val_prauc = -1.0
        wait = 0

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, dl_train, optimizer, criterion)
            y_val_true, y_val_prob = predict_probs(model, dl_val)
            prauc = average_precision_score(y_val_true, y_val_prob)
            print(f"[Fold {fold}] epoch {epoch:02d} | loss={loss:.4f} | val_pr_auc={prauc:.4f}")

            if prauc > best_val_prauc:
                best_val_prauc = prauc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"[Fold {fold}] early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        y_test_true, y_test_prob = predict_probs(model, dl_test)
        metrics = compute_metrics(y_test_true, y_test_prob, thr=0.5)
        metrics["Fold"] = fold
        all_fold_metrics.append(metrics)

        fold_pred = pd.DataFrame({
            "row_index": test_html["row_index"].values,
            "url": test_html["url"].values,
            "y_true": y_test_true,
            "fusion_prob": y_test_prob,
            "fusion_pred": (y_test_prob >= 0.5).astype(int),
            "fold": fold
        })
        all_test_rows.append(fold_pred)

    oof_df = pd.concat(all_test_rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase21_fusion_oof_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(OUT_DIR / "phase21_fold_metrics.csv", index=False)

    summary = metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase21_metrics_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 21 SUMMARY")
    print("=" * 70)
    print(summary)

    print("\nSaved:")
    print(OUT_DIR / "phase21_fusion_oof_predictions.csv")
    print(OUT_DIR / "phase21_fold_metrics.csv")
    print(OUT_DIR / "phase21_metrics_summary.csv")

if __name__ == "__main__":
    main()