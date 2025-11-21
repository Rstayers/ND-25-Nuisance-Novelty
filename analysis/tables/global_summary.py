# analysis/tables/global_summary.py

import pandas as pd

from analysis.plots.utils_plot import ensure_dir, joinp


def export_global_summary(df_metrics: pd.DataFrame, out_root: str) -> None:
    """
    Export a global metrics table using the OpenOOD-style metrics CSV.

    Expects columns like:
      backbone, detector, dataset, subset, n_id, n_ood, <metric columns...>

    We keep only subset == 'ALL' rows (one per dataset/backbone/detector).
    """
    if df_metrics.empty:
        print("[WARN] Metrics dataframe is empty; skipping global summary.")
        return

    if "subset" in df_metrics.columns:
        df = df_metrics[df_metrics["subset"] == "ALL"].copy()
    else:
        df = df_metrics.copy()

    if df.empty:
        print("[WARN] No rows with subset == 'ALL' in metrics dataframe.")
        return

    tables_dir = joinp(out_root, "tables")
    ensure_dir(tables_dir)
    out_csv = joinp(tables_dir, "global_metrics_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"[OK] Global metrics summary saved → {out_csv}")
