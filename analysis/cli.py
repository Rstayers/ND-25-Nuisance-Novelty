# analysis/cli.py

import argparse
import os

import pandas as pd

from analysis.plots import compare_detectors
from i_o import load_with_enrichment
from plots import (
    scale_curves,
    backbone_compare,
    ninco as ninco_plots,
    composition,
    shift_breakdown,  # only for CNS heatmaps
)
from tables import accuracy as accuracy_tables
from tables import global_summary, ninco_summary


def run_paper_story(
    per_sample_csv: str,
    metrics_csv: str,
    out_root: str,
    fpr: float = 0.05,
    heatmap_backbone: str | None = None,
    detector: str = "msp",
) -> None:
    os.makedirs(out_root, exist_ok=True)

    print(f"[LOAD] Per-sample CSV: {per_sample_csv}")
    df = load_with_enrichment(per_sample_csv)

    try:
        print(f"[LOAD] Metrics CSV: {metrics_csv}")
        df_metrics = pd.read_csv(metrics_csv)
    except FileNotFoundError:
        print(f"[WARN] Metrics CSV not found at {metrics_csv}; global metrics table will be skipped.")
        df_metrics = pd.DataFrame()

    # ---------------- CNS-Bench ----------------
    df_cns = df[df["dataset"].astype(str).str.contains("cns", case=False, na=False)].copy()
    out_cns = os.path.join(out_root, "cns")
    os.makedirs(out_cns, exist_ok=True)

    if df_cns.empty:
        print("[WARN] No CNS-Bench rows found.")
    else:
        print(f"[INFO] CNS rows: {len(df_cns)}")

        # Accuracy curves
        scale_curves.plot_classifier_accuracy(df_cns, out_cns, fpr_focus=fpr, dataset_label="cns")
        scale_curves.plot_detector_accuracy(df_cns, out_cns, fpr_focus=fpr, dataset_label="cns")

        # Outcome curves (4 figures)
        backbone_compare.plot_outcome_curves_all_backbones(
            df_cns,
            out_cns,
            dataset_label="cns",
            fpr_focus=fpr,
        )
        # CNS
        composition.plot_composition_by_backbone(df_cns, out_cns, dataset_label="cns", fpr_focus=fpr)
        composition.plot_composition_by_detector(df_cns, out_cns, dataset_label="cns", fpr_focus=fpr)
        compare_detectors.plot_outcome_curves_all_detectors(
            df_cns, out_cns, dataset_label="cns", fpr_focus=fpr
        )

        compare_detectors.plot_outcome_curves_all_detectors(
            df_cns, out_cns, dataset_label="cns", fpr_focus=fpr
        )
        # Optional CNS heatmap for a single backbone (kept because it’s very interpretable)
        if heatmap_backbone is not None:
            shift_breakdown.plot_cns_shift_heatmap(
                df_cns,
                out_cns,
                backbone=heatmap_backbone,
                detector=detector,
                top_n=15,
                fpr_focus=fpr,
            )

        # CNS accuracy table
        accuracy_tables.export_cns_accuracy_table(df_cns, out_cns, fpr_focus=fpr)

    # ---------------- ImageNet-C ----------------
    df_imc = df[df["dataset"].astype(str).str.contains("imagenet_c", case=False, na=False)].copy()
    out_imc = os.path.join(out_root, "imagenet_c")
    os.makedirs(out_imc, exist_ok=True)

    if df_imc.empty:
        print("[WARN] No ImageNet-C rows found.")
    else:
        print(f"[INFO] ImageNet-C rows: {len(df_imc)}")

        scale_curves.plot_classifier_accuracy(
            df_imc, out_imc, fpr_focus=fpr, dataset_label="imagenet_c"
        )
        scale_curves.plot_detector_accuracy(
            df_imc, out_imc, fpr_focus=fpr, dataset_label="imagenet_c"
        )
        composition.plot_composition_by_backbone(df_imc, out_imc, dataset_label="imagenet_c", fpr_focus=fpr)
        composition.plot_composition_by_detector(df_imc, out_imc, dataset_label="imagenet_c", fpr_focus=fpr)
        compare_detectors.plot_outcome_curves_all_detectors(
            df_imc, out_imc, dataset_label="imagenet_c", fpr_focus=fpr
        )

        # ImageNet-C
        compare_detectors.plot_outcome_curves_all_detectors(
            df_imc,
            out_imc,
            dataset_label="imagenet_c",
            fpr_focus=fpr,
        )
    # ---------------- NINCO ----------------
    df_ninco = df[df["dataset"].astype(str).str.contains("ninco", case=False, na=False)].copy()
    out_ninco = os.path.join(out_root, "ninco")
    os.makedirs(out_ninco, exist_ok=True)

    if df_ninco.empty:
        print("[WARN] No NINCO rows found.")
    else:
        print(f"[INFO] NINCO rows: {len(df_ninco)}")

        ninco_plots.plot_ninco_ood_success_by_subset(df_ninco, out_ninco, fpr_focus=fpr)
        ninco_plots.plot_ninco_score_histograms(df_ninco, out_ninco, fpr_focus=fpr)
        ninco_summary.export_ninco_subset_table(df_ninco, out_ninco, fpr_focus=fpr)

    # ---------------- Global metrics table ----------------
    if not df_metrics.empty:
        global_summary.export_global_summary(df_metrics, out_root)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Psycho Benchmark — Paper Story Visualizations")
    ap.add_argument("--per-sample", required=True, help="Per-sample CSV (psycho_per_sample_*.csv)")
    ap.add_argument("--metrics", required=True, help="Metrics CSV (psycho_metrics_*.csv)")
    ap.add_argument("--out", required=True, help="Output root directory for figures/tables")
    ap.add_argument("--fpr", type=float, default=0.05, help="Target FPR for thresholded metrics")
    ap.add_argument(
        "--heatmap-backbone",
        type=str,
        default=None,
        help="Backbone to use for CNS heatmaps (if omitted, no heatmap).",
    )
    ap.add_argument(
        "--detector",
        type=str,
        default="msp",
        help="Detector name to use for CNS heatmaps (default: msp)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    run_paper_story(
        per_sample_csv=args.per_sample,
        metrics_csv=args.metrics,
        out_root=args.out,
        fpr=args.fpr,
        heatmap_backbone=args.heatmap_backbone,
        detector=args.detector,
    )


if __name__ == "__main__":
    main()
