import numpy as np
import pandas as pd
from pathlib import Path

PHASE6_OOF = "phase06_multichannel_joint_outputs/phase06_multichannel_oof_predictions.csv"
ROOT_DIR = Path("phase20_branch_cv_outputs")
OUT_DIR = Path("phase09_error_specialist_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10

# uncertainty band around the best main model decision
LOW_CONF = 0.25
HIGH_CONF = 0.75

def main():
    oof = pd.read_csv(PHASE6_OOF).sort_values("row_index").reset_index(drop=True)

    # hard cases:
    # 1) main model mistakes
    # 2) near-boundary cases
    oof["is_error"] = (oof["phase06_multichannel_pred"].astype(int) != oof["y_true"].astype(int)).astype(int)
    oof["is_uncertain"] = ((oof["phase06_multichannel_prob"] >= LOW_CONF) & (oof["phase06_multichannel_prob"] <= HIGH_CONF)).astype(int)

    hard_df = oof[(oof["is_error"] == 1) | (oof["is_uncertain"] == 1)].copy()
    hard_df.to_csv(OUT_DIR / "phase09_hard_cases.csv", index=False)

    # Save per-fold hard-case indices for debugging
    per_fold = []
    for fold in range(1, N_SPLITS + 1):
        fold_df = hard_df[hard_df["fold"] == fold].copy()
        fold_df.to_csv(OUT_DIR / f"phase09_fold_{fold:02d}_hard_cases.csv", index=False)
        per_fold.append({
            "fold": fold,
            "num_hard_cases": len(fold_df),
            "num_errors": int(fold_df["is_error"].sum()),
            "num_uncertain": int(fold_df["is_uncertain"].sum()),
        })

    pd.DataFrame(per_fold).to_csv(OUT_DIR / "phase09_hard_case_summary.csv", index=False)

    print("Saved:", OUT_DIR / "phase09_hard_cases.csv")
    print("Saved:", OUT_DIR / "phase09_hard_case_summary.csv")
    print("Total hard cases:", len(hard_df))

if __name__ == "__main__":
    main()
