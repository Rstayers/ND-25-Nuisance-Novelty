# Nuisance Novelty Analysis Module

Post-benchmark analysis pipeline for generating paper-ready figures and tables.

## Quick Start

```bash
# Generate all analysis from benchmark results
python -m analysis.run_report \
    --benchmark_csvs analysis/bench_results/imagenet/bench_samples.csv \
    --out paper_output
```

---

## Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy
```

---

## Input Data

The analysis module reads `bench_samples.csv` files produced by `bench/run_bench.py`.

### Required Columns

| Column | Description |
|--------|-------------|
| `backbone` | Model architecture (resnet50, vit_b_16, etc.) |
| `detector` | OOD detector name (msp, odin, knn, etc.) |
| `dataset` | Dataset name (ImageNet-LN, CUB-LN, etc.) |
| `outcome` | One of: Clean_Success, Nuisance_Novelty, Double_Failure, Contained_Misidentification |
| `level` | Severity level (0-5, where 0=clean) |
| `nuisance` | Corruption type (blur, noise, etc.) |

### Optional Columns (for OSA metrics)

| Column | Description |
|--------|-------------|
| `n_unknown` | Number of OOD test samples |
| `rejected_unknowns` | Number of correctly rejected OOD samples |

---

## Running the Analysis

### Basic Usage

```bash
# Single benchmark dataset
python -m analysis.run_report \
    --benchmark_csvs analysis/bench_results/imagenet/bench_samples.csv \
    --out paper_output

# Multiple benchmark datasets
python -m analysis.run_report \
    --benchmark_csvs \
        analysis/bench_results/imagenet/bench_samples.csv \
        analysis/bench_results/cub/bench_samples.csv \
        analysis/bench_results/cars/bench_samples.csv \
    --out paper_output
```

### Cross-Dataset Comparison

```bash
python -m analysis.run_report \
    --cross_csvs \
        analysis/bench_results/imagenet/bench_samples.csv \
        analysis/bench_results/cub/bench_samples.csv \
        analysis/bench_results/cars/bench_samples.csv \
    --reference "ImageNet-LN" \
    --out paper_output
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--benchmark_csvs` | CSV files for per-dataset analysis |
| `--benchmark_filter` | Filter to specific datasets |
| `--cross_csvs` | CSV files for cross-dataset comparison |
| `--cross_filter` | Filter to specific datasets |
| `--reference` | Reference dataset for transferability (default: ImageNet-LN) |
| `--out` | Output directory (default: paper_output) |

---

## Output Structure

```
paper_output/
├── data/
│   ├── benchmark_agg.csv      # Aggregated benchmark data
│   └── cross_agg.csv          # Aggregated cross-dataset data
│
├── figures/
│   ├── per_dataset/
│   │   ├── ImageNet_LN/
│   │   │   ├── csa_ccr_ImageNet_LN/     # CSA vs CCR plots (per detector)
│   │   │   ├── osa_csa_ImageNet_LN/     # OSA vs CSA plots
│   │   │   ├── fingerprint_ImageNet_LN/ # Top nuisances bar charts
│   │   │   ├── nnr_severity/            # NNR vs severity
│   │   │   └── osa_severity/            # OSA vs severity
│   │   └── CUB_LN/
│   │       └── ...
│   │
│   ├── benchmark/
│   │   ├── csa_ccr_benchmark/           # CSA/CCR across benchmarks
│   │   ├── gap_comparison/              # OSA gap comparison
│   │   ├── NNR_heatmaps_L5.pdf          # NNR heatmap at L5
│   │   ├── OSA_heatmaps_L5.pdf          # OSA heatmap at L5
│   │   ├── ADR_heatmaps.pdf             # ADR heatmap
│   │   ├── ADR_violin.pdf               # ADR violin plot
│   │   └── config_benchmark_*.pdf       # Config x benchmark heatmaps
│   │
│   └── cross_dataset/
│       ├── transferability/             # NNR correlation plots
│       ├── transferability_combined_L5.pdf
│       ├── bump_chart/                  # Detector rank changes
│       ├── finegrained_*.pdf            # CUB vs Cars comparison
│       └── nuisance_heatmap/            # Nuisance category analysis
│
└── tables/
    ├── per_dataset/
    │   ├── summary_ImageNet_LN.csv      # Summary table
    │   ├── NNR_matrix_ImageNet_LN.csv   # NNR detector x backbone
    │   └── full_ImageNet_LN.csv         # Full results table
    │
    ├── benchmark/
    │   ├── NNR_L5_benchmark.csv         # NNR at L5 across benchmarks
    │   ├── OSA_L5_benchmark.csv         # OSA at L5 across benchmarks
    │   ├── ADR_benchmark.csv            # ADR across benchmarks
    │   ├── benchmark_NNR.csv            # Mean NNR
    │   ├── benchmark_OSA.csv            # Mean OSA
    │   └── *_by_severity_*.csv          # Metrics by severity
    │
    └── cross_dataset/
        ├── cross_dataset_NNR.csv        # NNR across datasets
        ├── cross_dataset_OSA.csv        # OSA across datasets
        ├── cross_dataset_ADR.csv        # ADR across datasets
        ├── full_results.csv             # All results
        ├── ranks_NNR.csv                # Configuration rankings
        ├── ranks_OSA.csv                # OSA rankings
        └── outcomes.csv                 # Outcome distribution
```

---

## Metric Definitions

```
CSA     = N_correct / N_total                    # Closed-Set Accuracy
CCR     = Clean_Success / N_total                # Correct Classification Rate @ threshold
NNR     = Nuisance_Novelty / N_total             # Nuisance Novelty Rate (prevalence)
OSA_Gap = CSA - CCR = NNR                        # Known-side divergence
URR     = rejected_unknowns / n_unknown          # Unknown Rejection Rate
OSA     = (Clean_Success + rejected_unknowns) / (N_total + n_unknown)
ADR     = slope of OSA_Gap vs severity           # Accuracy Divergence Rate
```

### Outcome Categories

| Outcome | Description | Meaning |
|---------|-------------|---------|
| Clean_Success | Correct AND Accepted | Ideal case |
| Nuisance_Novelty | Correct AND Rejected | **THE PROBLEM** we measure |
| Double_Failure | Wrong AND Rejected | Model failure, but safely rejected |
| Contained_Misidentification | Wrong AND Accepted | Dangerous misclassification |

---

## Module Files

| File | Purpose |
|------|---------|
| `run_report.py` | Main CLI runner |
| `processing.py` | Data loading, aggregation, metric computation |
| `plots.py` | All plotting functions (paper-ready, colorblind-friendly) |
| `tables.py` | LaTeX and CSV table generation |

---

## Key Design Principles

1. **Never aggregate across backbones or detectors** - Every table/plot preserves full (backbone, detector) granularity

2. **Aggregate over nuisance types** - Sum outcome counts across corruption types, then recompute metrics

3. **Can average over severity levels** - For mean metrics, average levels 1-5 for a given (backbone, detector, dataset)

4. **OOD constants propagate** - `n_unknown` and `rejected_unknowns` are constant per (backbone, detector), enabling OSA computation at any granularity

---

## Customization

### Color Schemes

Colors are defined in `plots.py`:

```python
# Backbone colors (Paul Tol's palette)
BACKBONE_COLORS = {
    'convnext_t': '#4477AA',   # Blue
    'densenet121': '#EE6677',  # Rose
    'resnet50': '#228833',     # Green
    'swin_t': '#CCBB44',       # Gold
    'vit_b_16': '#AA3377',     # Purple
}

# Detector colors (colorblind-friendly)
DETECTOR_COLORS = {
    'msp': '#E69F00',      # Orange
    'odin': '#F0E442',     # Yellow
    'knn': '#000000',      # Black
    ...
}
```

### Adding New Plots

1. Add function to `plots.py`
2. Call from `plot_per_dataset_all()`, `plot_benchmark_all()`, or `plot_cross_dataset_all()`

### Adding New Tables

1. Add function to `tables.py`
2. Call from `run_per_dataset()`, `run_benchmark()`, or `run_cross_dataset()` in `run_report.py`

---

## Troubleshooting

### "No data" or empty plots

- Check that your CSV has the required columns
- Verify `level` column has values > 0 (corrupted samples)
- Check `outcome` column has valid category names

### OSA metrics missing

OSA requires `n_unknown` and `rejected_unknowns` columns in the input CSV. These are added by `bench/run_bench.py` automatically. For older CSV files, use:

```python
from analysis.processing import enrich_with_summary
df = enrich_with_summary(sample_df, "path/to/bench_summary.csv")
```

### Memory issues with large CSVs

Process datasets individually:

```bash
python -m analysis.run_report \
    --benchmark_csvs analysis/bench_results/imagenet/bench_samples.csv \
    --out paper_output/imagenet
```

---

## Example Workflow

```bash
# 1. Run benchmark
python -m bench.run_bench --config bench/configs/imagenet.yaml

# 2. Generate analysis
python -m analysis.run_report \
    --benchmark_csvs analysis/bench_results/imagenet/bench_samples.csv \
    --out paper_output

# 3. Check outputs
ls paper_output/figures/benchmark/
ls paper_output/tables/benchmark/
```
