import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "final_multimodal_dataset.parquet"
TARGETED_PATH = "phase23_outputs/targeted_features.npy"

PHASE20_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR = Path("phase24_branch_cv_with_targeted")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10

def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    y = (df["label"].astype(str).str.lower() == "phish").astype(int).values
    targeted = np.load(TARGETED_PATH).astype(np.float32)

    if len(df) != len(targeted):
        raise ValueError("Targeted feature length mismatch")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        src_fold = PHASE20_DIR / f"fold_{fold:02d}"
        dst_fold = OUT_DIR / f"fold_{fold:02d}"
        dst_fold.mkdir(parents=True, exist_ok=True)

        # copy metadata references only by saving split-targeted arrays
        np.save(dst_fold / "train_targeted_features.npy", targeted[train_idx])
        np.save(dst_fold / "test_targeted_features.npy", targeted[test_idx])

        # record row indices for sanity
        pd.DataFrame({
            "row_index": train_idx,
            "url": df.iloc[train_idx]["url"].astype(str).values,
            "y_true": y[train_idx],
        }).to_csv(dst_fold / "train_rows.csv", index=False)

        pd.DataFrame({
            "row_index": test_idx,
            "url": df.iloc[test_idx]["url"].astype(str).values,
            "y_true": y[test_idx],
        }).to_csv(dst_fold / "test_rows.csv", index=False)

        print(f"Fold {fold}: saved targeted features to {dst_fold}")

if __name__ == "__main__":
    main()