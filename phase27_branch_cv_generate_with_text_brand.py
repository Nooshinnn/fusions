import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from scipy import sparse

DATA_PATH = "final_multimodal_dataset.parquet"
TEXT_PATH = "phase26_outputs/html_text_tfidf.npz"
BRAND_PATH = "phase26_outputs/brand_mismatch_features.npy"

OUT_DIR = Path("phase27_text_brand_cv")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10

def main():
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    y = (df["label"].astype(str).str.lower() == "phish").astype(int).values

    X_text = sparse.load_npz(TEXT_PATH)
    X_brand = np.load(BRAND_PATH).astype(np.float32)

    if len(df) != X_text.shape[0] or len(df) != len(X_brand):
        raise ValueError("Feature length mismatch")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        fold_dir = OUT_DIR / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        sparse.save_npz(fold_dir / "train_text_tfidf.npz", X_text[train_idx])
        sparse.save_npz(fold_dir / "test_text_tfidf.npz", X_text[test_idx])

        np.save(fold_dir / "train_brand_features.npy", X_brand[train_idx])
        np.save(fold_dir / "test_brand_features.npy", X_brand[test_idx])

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