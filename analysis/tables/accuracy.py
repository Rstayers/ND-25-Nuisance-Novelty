# analysis/tables/accuracy.py

import pandas as pd

from analysis.plots.utils_plot import ensure_dir, joinp


def export_cns_accuracy_table(
    df: pd.DataFrame,
    out_root: str,
    fpr_focus: float = 0.05,
):
    """
    Export a table of classifier and detector accuracies on CNS-Bench
    across nuisance severity, plus an average row.

    For each (backbone, detector, severity):
        cls_acc = mean(correct_cls)
        det_acc = mean(accept)
        n       = count

    And an extra severity='avg' row per (backbone, detector).
    """
    cns = df[df["dataset"].astype(str).str.contains("cns", case=False, na=False)].copy()
    if cns.empty:
        print("[WARN] No CNS-Bench rows for accuracy table.")
        return

    if "fpr_target" in cns.columns:
        cns = cns[cns["fpr_target"].sub(fpr_focus).abs() < 1e-8].copy()
        if cns.empty:
            print(f"[WARN] No CNS rows at FPR={fpr_focus}; accuracy table not written.")
            return

    group_cols = ["backbone", "detector", "severity"]
    acc = (
        cns.groupby(group_cols, dropna=False)
           .agg(
               cls_acc=("correct_cls", "mean"),
               det_acc=("accept", "mean"),
               n=("correct_cls", "size"),
           )
           .reset_index()
    )

    # Average over severities
    avg = (
        acc.groupby(["backbone", "detector"], dropna=False)
           .agg(
               cls_acc=("cls_acc", "mean"),
               det_acc=("det_acc", "mean"),
               n=("n", "sum"),
           )
           .reset_index()
    )
    avg["severity"] = "avg"

    acc["severity"] = acc["severity"].astype(str)
    full = pd.concat([acc, avg], ignore_index=True)

    def _sev_key(x):
        try:
            return (0, float(x))
        except ValueError:
            return (1, float("inf"))

    full["severity_order"] = full["severity"].apply(_sev_key)
    full = full.sort_values(["backbone", "detector", "severity_order"]).drop(
        columns=["severity_order"]
    )

    summaries_dir = joinp(out_root, "tables")
    ensure_dir(summaries_dir)
    out_csv = joinp(summaries_dir, f"cns_accuracy_table_FPR{fpr_focus:.2f}.csv")
    full.to_csv(out_csv, index=False)

    print(f"[OK] CNS accuracy table saved → {out_csv}")
