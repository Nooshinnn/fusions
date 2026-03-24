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
OUT_DIR = Path("phase25_fusion_cv_with_targeted_v2_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 128
EPOCHS = 22
PATIENCE = 5
LR = 7e-4
WEIGHT_DECAY = 1e-4

# -----------------------------
# Dataset
# -----------------------------
class FusionDataset(Dataset):
    def __init__(self, x_tab, x_html, x_tgt, x_override, y):
        self.x_tab = x_tab.astype(np.float32)
        self.x_html = x_html.astype(np.float32)
        self.x_tgt = x_tgt.astype(np.float32)
        self.x_override = x_override.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x_tab[i], dtype=torch.float32),
            torch.tensor(self.x_html[i], dtype=torch.float32),
            torch.tensor(self.x_tgt[i], dtype=torch.float32),
            torch.tensor(self.x_override[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )

# -----------------------------
# Stronger targeted fusion
# -----------------------------
class TargetedFusionV2(nn.Module):
    def __init__(self, tab_dim, html_dim, tgt_dim, override_dim):
        super().__init__()

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU()
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

        self.override_proj = nn.Sequential(
            nn.Linear(override_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # how much should HTML be trusted more than usual?
        self.html_boost_gate = nn.Sequential(
            nn.Linear(32 + 32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # second gate for final HTML dominance under conflict
        self.override_gate = nn.Sequential(
            nn.Linear(32 + 64 + 32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 32 + 16 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x_tab, x_html, x_tgt, x_override):
        t = self.tab_proj(x_tab)            # [B,32]
        h = self.html_proj(x_html)          # [B,64]
        g = self.tgt_proj(x_tgt)            # [B,32]
        o = self.override_proj(x_override)  # [B,16]

        # boost HTML under suspicious conflict conditions
        html_boost = self.html_boost_gate(torch.cat([t, g, o], dim=1))   # [B,1]
        h_boosted = h * (1.0 + html_boost)

        # final override gate
        ov = self.override_gate(torch.cat([t, h_boosted, g, o], dim=1))  # [B,1]
        h_override = h_boosted * (1.0 + ov)

        # interaction terms from HTML + targeted context
        h_ctx = torch.cat([
            h,
            h_boosted,
            h_override,
            h_override * torch.sigmoid(torch.mean(g, dim=1, keepdim=True))
        ], dim=1)

        z = torch.cat([t, h, g, o, h_boosted, h_override], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

# -----------------------------
# Helpers
# -----------------------------
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
    for x_tab, x_html, x_tgt, x_override, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_html = x_html.to(DEVICE)
        x_tgt = x_tgt.to(DEVICE)
        x_override = x_override.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tab, x_html, x_tgt, x_override)
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
    for x_tab, x_html, x_tgt, x_override, y in loader:
        x_tab = x_tab.to(DEVICE)
        x_html = x_html.to(DEVICE)
        x_tgt = x_tgt.to(DEVICE)
        x_override = x_override.to(DEVICE)

        logits = model(x_tab, x_html, x_tgt, x_override)
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(probs)

# -----------------------------
# Main
# -----------------------------
def main():
    all_test_rows = []
    all_fold_metrics = []

    for fold in range(1, N_SPLITS + 1):
        fold_dir = ROOT_DIR / f"fold_{fold:02d}"
        tgt_dir = TARGETED_DIR / f"fold_{fold:02d}"

        print("\n" + "=" * 70)
        print(f"TARGETED FUSION V2 FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_url = pd.read_csv(fold_dir / "train_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_url = pd.read_csv(fold_dir / "test_url_outputs.csv").sort_values("row_index").reset_index(drop=True)
        train_html = pd.read_csv(fold_dir / "train_html_outputs.csv").sort_values("row_index").reset_index(drop=True)
        test_html = pd.read_csv(fold_dir / "test_html_outputs.csv").sort_values("row_index").reset_index(drop=True)

        train_html_feat = np.load(fold_dir / "train_html_features.npy").astype(np.float32)
        test_html_feat = np.load(fold_dir / "test_html_features.npy").astype(np.float32)

        train_tgt = np.load(tgt_dir / "train_targeted_features.npy").astype(np.float32)
        test_tgt = np.load(tgt_dir / "test_targeted_features.npy").astype(np.float32)

        # base tabular features
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

            # conflict / override features
            df_tab["html_strong_phish"] = (df_tab["html_prob"] > 0.80).astype(float)
            df_tab["url_strong_benign"] = (df_tab["url_textcnn_prob"] < 0.30).astype(float)
            df_tab["rf_strong_benign"] = (df_tab["url_rf_prob"] < 0.30).astype(float)

            df_tab["html_url_conflict"] = (
                (df_tab["html_prob"] > 0.70) & (df_tab["url_textcnn_prob"] < 0.30)
            ).astype(float)

            df_tab["html_rf_conflict"] = (
                (df_tab["html_prob"] > 0.70) & (df_tab["url_rf_prob"] < 0.30)
            ).astype(float)

            df_tab["high_disagreement"] = (df_tab["range_prob"] > 0.50).astype(float)

            df_tab["override_html_flag"] = (
                (df_tab["html_prob"] > 0.85) &
                (df_tab["url_textcnn_prob"] < 0.40)
            ).astype(float)

            df_tab["override_html_rf_flag"] = (
                (df_tab["html_prob"] > 0.85) &
                (df_tab["url_rf_prob"] < 0.40)
            ).astype(float)

        tab_cols = list(train_tab.columns)

        # explicit override block from tab + targeted
        # indices in targeted features from phase23:
        # 0 hosted, 1 short, 2 storage, 5 pw, 6 brand_hits, 7 login_hits,
        # 16 brand_host_mismatch, 17 high_login_low_brand_overlap,
        # 18 generic_host_with_login, 19 login_on_neutral_host,
        # 20 social_brand_mismatch, 21 finance_brand_mismatch, 22 forms_brand_mismatch
        tgt_keep_idx = [0, 1, 2, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22]

        train_override = np.column_stack([
            train_tab["html_strong_phish"].values,
            train_tab["url_strong_benign"].values,
            train_tab["rf_strong_benign"].values,
            train_tab["html_url_conflict"].values,
            train_tab["html_rf_conflict"].values,
            train_tab["high_disagreement"].values,
            train_tab["override_html_flag"].values,
            train_tab["override_html_rf_flag"].values,
            train_tgt[:, tgt_keep_idx]
        ]).astype(np.float32)

        test_override = np.column_stack([
            test_tab["html_strong_phish"].values,
            test_tab["url_strong_benign"].values,
            test_tab["rf_strong_benign"].values,
            test_tab["html_url_conflict"].values,
            test_tab["html_rf_conflict"].values,
            test_tab["high_disagreement"].values,
            test_tab["override_html_flag"].values,
            test_tab["override_html_rf_flag"].values,
            test_tgt[:, tgt_keep_idx]
        ]).astype(np.float32)

        Xtab_train = train_tab[tab_cols].values.astype(np.float32)
        Xtab_test = test_tab[tab_cols].values.astype(np.float32)
        y_train = train_html["y_true"].astype(int).values
        y_test = test_html["y_true"].astype(int).values

        scaler_tab = StandardScaler()
        scaler_html = StandardScaler()
        scaler_tgt = StandardScaler()
        scaler_override = StandardScaler()

        Xtab_train = scaler_tab.fit_transform(Xtab_train)
        Xtab_test = scaler_tab.transform(Xtab_test)

        Xhtml_train = scaler_html.fit_transform(train_html_feat)
        Xhtml_test = scaler_html.transform(test_html_feat)

        Xtgt_train = scaler_tgt.fit_transform(train_tgt)
        Xtgt_test = scaler_tgt.transform(test_tgt)

        Xov_train = scaler_override.fit_transform(train_override)
        Xov_test = scaler_override.transform(test_override)

        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(
            idx,
            test_size=0.1,
            stratify=y_train,
            random_state=SEED
        )

        ds_train = FusionDataset(
            Xtab_train[tr_idx], Xhtml_train[tr_idx], Xtgt_train[tr_idx], Xov_train[tr_idx], y_train[tr_idx]
        )
        ds_val = FusionDataset(
            Xtab_train[val_idx], Xhtml_train[val_idx], Xtgt_train[val_idx], Xov_train[val_idx], y_train[val_idx]
        )
        ds_test = FusionDataset(
            Xtab_test, Xhtml_test, Xtgt_test, Xov_test, y_test
        )

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

        model = TargetedFusionV2(
            tab_dim=Xtab_train.shape[1],
            html_dim=Xhtml_train.shape[1],
            tgt_dim=Xtgt_train.shape[1],
            override_dim=Xov_train.shape[1]
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
        all_test_rows.append(pd.DataFrame({
            "row_index": test_html["row_index"].values,
            "url": test_html["url"].values,
            "y_true": y_test,
            "fusion_prob": y_test_prob,
            "fusion_pred": (y_test_prob >= 0.5).astype(int),
            "fold": fold
        }))

        print(
            f"[Fold {fold}] "
            f"Acc={metrics['Accuracy']:.4f} | "
            f"PR-AUC={metrics['PR-AUC']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"FP={metrics['FP']} | FN={metrics['FN']}"
        )

    oof_df = pd.concat(all_test_rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    oof_df.to_csv(OUT_DIR / "phase25_targeted_v2_fusion_oof_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(OUT_DIR / "phase25_targeted_v2_fold_metrics.csv", index=False)

    summary = metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary.to_csv(OUT_DIR / "phase25_targeted_v2_metrics_summary.csv")

    print("\n" + "=" * 70)
    print("PHASE 25 TARGETED V2 SUMMARY")
    print("=" * 70)
    print(summary)

    print("\nSaved:")
    print(OUT_DIR / "phase25_targeted_v2_fusion_oof_predictions.csv")
    print(OUT_DIR / "phase25_targeted_v2_fold_metrics.csv")
    print(OUT_DIR / "phase25_targeted_v2_metrics_summary.csv")

if __name__ == "__main__":
    main()