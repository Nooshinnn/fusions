import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "final_multimodal_dataset.parquet"
DOMAIN_INTEL_PATH = "phase29_outputs/domain_intel_features.npy"

OUT_DIR = Path("phase30_domain_intel_folds")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10

def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    y = (df["label"].astype(str).str.lower() == "phish").astype(int).values
    X = np.load(DOMAIN_INTEL_PATH).astype(np.float32)

    if len(df) != len(X):
        raise ValueError("Domain intel feature length mismatch")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        fold_dir = OUT_DIR / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        np.save(fold_dir / "train_domain_intel.npy", X[train_idx])
        np.save(fold_dir / "test_domain_intel.npy", X[test_idx])

        pd.DataFrame({
            "row_index": train_idx,
            "url": df.iloc[train_idx]["url"].astype(str).values,
            "y_true": y[train_idx]
        }).to_csv(fold_dir / "train_rows.csv", index=False)

        pd.DataFrame({
            "row_index": test_idx,
            "url": df.iloc[test_idx]["url"].astype(str).values,
            "y_true": y[test_idx]
        }).to_csv(fold_dir / "test_rows.csv", index=False)

        print(f"Fold {fold} saved to {fold_dir}")

if __name__ == "__main__":
    main()