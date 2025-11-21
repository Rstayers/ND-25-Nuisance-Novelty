# analysis/tables/ninco_summary.py

import pandas as pd

from analysis.plots.utils_plot import ensure_dir, joinp


def export_ninco_subset_table(
    df: pd.DataFrame,
    out_root: str,
    fpr_focus: float = 0.05,
) -> None:
    """
    Export a table summarizing NINCO performance per (backbone, detector, subset):

      - ood_success_rate: mean of system_outcome == OOD_CorrectReject
      - n: number of samples
    """
    ninco = df[df["dataset"].astype(str).str.contains("ninco", case=False, na=False)].copy()
    if ninco.empty:
        print("[WARN] No NINCO rows; cannot build subset table.")
        return

    if "fpr_target" in ninco.columns:
        ninco = ninco[ninco["fpr_target"].sub(fpr_focus).abs() < 1e-8].copy()
        if ninco.empty:
            print(f"[WARN] No NINCO rows at FPR={fpr_focus}; subset table not written.")
            return

    if "system_outcome" not in ninco.columns:
        print("[WARN] No system_outcome column in NINCO dataframe.")
        return

    ninco["ood_success"] = (ninco["system_outcome"] == "OOD_CorrectReject").astype(float)

    group_cols = ["backbone", "detector", "ood_subset"]
    agg = (
        ninco.groupby(group_cols, dropna=False)
             .agg(
                 ood_success_rate=("ood_success", "mean"),
                 n=("ood_success", "size"),
             )
             .reset_index()
    )

    tables_dir = joinp(out_root, "tables")
    ensure_dir(tables_dir)
    out_csv = joinp(tables_dir, f"ninco_subset_summary_FPR{fpr_focus:.2f}.csv")
    agg.to_csv(out_csv, index=False)

    print(f"[OK] NINCO subset summary table saved → {out_csv}")
