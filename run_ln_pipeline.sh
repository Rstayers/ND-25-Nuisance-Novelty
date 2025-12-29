#!/bin/bash
set -euo pipefail

AE_WEIGHTS="ln_dataset/assets/ae_classifier_aware_weights.pth"
PARCE_CALIB="ln_dataset/assets/parce.pt"
DATA_ROOT="data/images_largescale"
IMGLIST="data/benchmark_imglist/imagenet/val_imagenet.txt"
OUT_DIR="data/images_largescale/imagenet_ln_v6"

SAMPLES=300
SEED=0
ITERS=5

# Extreme-tail binning (matches your calibration logic: bigger early fracs => deeper low-tail edges)
TARGET_FRACS="0.35,0.25,0.20,0.15,0.05"

TS="$(date +%Y%m%d_%H%M%S)"
BINS_JSON="bin_edges_v5.5_${TS}.json"

echo ">>> [1/2] Calibrating bins -> ${BINS_JSON}"
python -m ln_dataset.calibrate_bins --parce_calib "${PARCE_CALIB}" --ae_weights "${AE_WEIGHTS}" --data "${DATA_ROOT}" --imglist "${IMGLIST}" --samples "${SAMPLES}" --seed "${SEED}" --target_fracs "${TARGET_FRACS}" --iters "${ITERS}" --save_json "${BINS_JSON}" | tee "calibrate_v5.5_${TS}.log"

echo ">>> [2/2] Generating LN v5.5 with calibrated bins"
python -m ln_dataset.generate_ln --ae_weights "${AE_WEIGHTS}" --parce_calib "${PARCE_CALIB}" --bin_edges_json "${BINS_JSON}" --data "${DATA_ROOT}" --imglist "${IMGLIST}" --out_dir "${OUT_DIR}" --debug_maps | tee "generate_v5.5_${TS}.log"

echo "✅ Done. Bins: ${BINS_JSON} | Out: ${OUT_DIR}"
