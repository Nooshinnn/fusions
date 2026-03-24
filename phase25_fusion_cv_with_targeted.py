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

ROOT_DIR = Path("phase20_branch_cv_outputs")
TARGETED_DIR = Path("phase24_branch_cv_with_targeted")
OUT_DIR = Path("phase25_fusion_cv_with_targeted_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5
LR = 8e-4
WEIGHT_DECAY = 1e-4

class FusionDataset(Dataset):
    def __init__(self, x_tab, x_html, x_tgt, y):
        self.x_tab = x_tab.astype(np.float32)
        self.x_html = x_html.astype(np.float32)
        self.x_tgt = x_tgt.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x_tab[i], dtype=torch.float32),
            torch.tensor(self.x_html[i], dtype=torch.float32),
            torch.tensor(self.x_tgt[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )

class TargetedFusion(nn.Module):
    def __init__(self, tab_dim, html_dim, tgt_dim):
        super().__init__()

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.html_proj = nn.Sequential(
            nn.Linear(html_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.tgt_proj = nn.Sequential(
            nn.Linear(tgt_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(32 + 64 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 32 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_html, x_tgt):
        t = self.tab_proj(x_tab)
        h = self.html_proj(x_html)
        gfeat = self.tgt_proj(x_tgt)

        gate = self.gate(torch.cat([t, h, gfeat], dim=1))
        h_gated = gate * h

        z = torch.cat([t, h, gfeat, h_gated], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

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
    for x_tab, x_html, x_tgt, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_html = x_html.to(DEVICE)
        x_tgt = x_tgt.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tab, x_html, x_tgt)
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
    for x_tab, x_html, x_tgt, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_html = x_html.to(DEVICE)
        x_tgt = x_tgt.to(DEVICE)
        logits = model(x_tab, x_html, x_tgt)
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(probs)

def main():
    all_test_rows = []
    all_fold_metrics = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        tgt_dir = TARGETED_DIR / f"fold_{fold:02d}"

        print("\n" + "=" * 70)
        print(f"TARGETED FUSION FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_html_feat = np.load(fold_dir / "train_html_features.npy").astype(np.float32)
        test_html_feat = np.load(fold_dir / "test_html_features.npy").astype(np.float32)

        train_tgt = np.load(tgt_dir / "train_targeted_features.npy").astype(np.float32)
        test_tgt = np.load(tgt_dir / "test_targeted_features.npy").astype(np.float32)

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
            df_tab["html_gt_url"] = (df_tab["html_prob"] > df_tab["url_textcnn_prob"]).astype(float)
            df_tab["url_gt_html"] = (df_tab["url_textcnn_prob"] > df_tab["html_prob"]).astype(float)
            df_tab["rf_gt_html"] = (df_tab["url_rf_prob"] > df_tab["html_prob"]).astype(float)
            df_tab["top_gap"] = df_tab["max_prob"] - df_tab["mean_prob"]

        Xtab_train = train_tab.values.astype(np.float32)
        Xtab_test = test_tab.values.astype(np.float32)
        y_train = train_html["y_true"].astype(int).values
        y_test = test_html["y_true"].astype(int).values

        scaler_tab = StandardScaler()
        scaler_html = StandardScaler()
        scaler_tgt = StandardScaler()

        Xtab_train = scaler_tab.fit_transform(Xtab_train)
        Xtab_test = scaler_tab.transform(Xtab_test)
        Xhtml_train = scaler_html.fit_transform(train_html_feat)
        Xhtml_test = scaler_html.transform(test_html_feat)
        Xtgt_train = scaler_tgt.fit_transform(train_tgt)
        Xtgt_test = scaler_tgt.transform(test_tgt)

        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=y_train, random_state=SEED)

        ds_train = FusionDataset(Xtab_train[tr_idx], Xhtml_train[tr_idx], Xtgt_train[tr_idx], y_train[tr_idx])
        ds_val = FusionDataset(Xtab_train[val_idx], Xhtml_train[val_idx], Xtgt_train[val_idx], y_train[val_idx])
        ds_test = FusionDataset(Xtab_test, Xhtml_test, Xtgt_test, y_test)

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = TargetedFusion(
            tab_dim=Xtab_train.shape[1],
            html_dim=Xhtml_train.shape[1],
            tgt_dim=Xtgt_train.shape[1]
        ).to(DEVICE)

        n_neg = int((y_train[tr_idx] == 0).sum())
        n_pos = int((y_train[tr_idx] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            "y_true": y_test,
            "fusion_prob": y_test_prob,
            "fusion_pred": (y_test_prob >= 0.5).astype(int),
            "fold": fold
        })
        all_test_rows.append(fold_pred)

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"PR-AUC={metrics['PR-AUC']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"FP={metrics['FP']} | FN={metrics['FN']}"
        )

    oof_df = pd.concat(all_test_rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase25_targeted_fusion_oof_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(OUT_DIR / "phase25_targeted_fold_metrics.csv", index=False)

    summary = metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase25_targeted_metrics_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 25 TARGETED SUMMARY")
    print("=" * 70)
    print(summary)

    print("\nSaved:")
    print(OUT_DIR / "phase25_targeted_fusion_oof_predictions.csv")
    print(OUT_DIR / "phase25_targeted_fold_metrics.csv")
    print(OUT_DIR / "phase25_targeted_metrics_summary.csv")

if __name__ == "__main__":
    main()