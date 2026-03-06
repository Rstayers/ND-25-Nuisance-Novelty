"""
Unified LN Dataset Generation Pipeline

Runs all steps from a single config file:
  0. Train Autoencoder (if weights don't exist)
  1. Calibrate PaRCE (per-class reconstruction statistics)
  2. Calibrate Bin Edges (severity level thresholds)
  3. Generate LN Dataset
  4. Verify Manifest

Usage:
    python -m ln_dataset.core.run_pipeline --config ln_dataset/configs/imagenet.yaml
"""

import os
import sys
import subprocess
import argparse
from ln_dataset.core.configs import load_config


def run_command(cmd, step_name):
    """Execute a subprocess command with logging."""
    print(f"\n{'='*60}")
    print(f"[{step_name}]")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Unified LN Dataset Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m ln_dataset.core.run_pipeline --config ln_dataset/configs/imagenet.yaml

This will:
    0. Train the autoencoder (if weights don't exist)
    1. Calibrate PaRCE statistics
    2. Calibrate severity bin edges
    3. Generate the LN dataset
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help="Path to dataset config YAML file")

    # CLI overrides for paths
    parser.add_argument('--train_list', type=str, default=None,
                        help="Override training image list path")
    parser.add_argument('--val_list', type=str, default=None,
                        help="Override validation image list path")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Override output directory")
    parser.add_argument('--ae_weights', type=str, default=None,
                        help="Override autoencoder weights path")

    # Training parameters
    parser.add_argument('--ae_epochs', type=int, default=50,
                        help="Number of epochs for AE training (default: 50)")
    parser.add_argument('--ae_batch_size', type=int, default=32,
                        help="Batch size for AE training (default: 32)")

    # Calibration parameters
    parser.add_argument('--parce_samples', type=int, default=20000,
                        help="Number of samples for PaRCE calibration (default: 20000)")
    parser.add_argument('--bins_samples', type=int, default=5000,
                        help="Number of samples for bin edge calibration (default: 5000)")

    # Generation parameters
    parser.add_argument('--debug_max', type=int, default=0,
                        help="Number of debug images to save during generation (default: 0)")

    # Control flags
    parser.add_argument('--skip_train_ae', action='store_true',
                        help="Skip AE training even if weights don't exist")
    parser.add_argument('--force_recalibrate', action='store_true',
                        help="Force recalibration even if files exist")

    args = parser.parse_args()

    # Load config
    print(f"\nLoading config: {args.config}")
    cfg = load_config(args.config)

    # Resolve paths from config with CLI overrides
    train_data = cfg.paths.train_data if cfg.paths.train_data else "."
    train_list = args.train_list or cfg.paths.train_list

    val_data = cfg.paths.val_data if cfg.paths.val_data else "."
    val_list = args.val_list or cfg.paths.val_list

    out_dir = args.out_dir or cfg.paths.out_dir
    ae_weights = args.ae_weights or cfg.paths.ae_weights

    # Validate required paths
    if not train_list or not val_list or not out_dir:
        print("Error: Missing required paths (train_list, val_list, or out_dir).")
        print("Check your config file or provide CLI overrides.")
        sys.exit(1)

    if not ae_weights:
        print("Error: Missing ae_weights path. Specify in config or via --ae_weights.")
        sys.exit(1)

    # Print configuration
    print("\n" + "=" * 60)
    print("PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"  Config:      {args.config}")
    print(f"  Train Data:  {train_data}")
    print(f"  Train List:  {train_list}")
    print(f"  Val Data:    {val_data}")
    print(f"  Val List:    {val_list}")
    print(f"  AE Weights:  {ae_weights}")
    print(f"  Output Dir:  {out_dir}")
    print(f"  AE Epochs:   {args.ae_epochs}")
    print(f"  Debug Max:   {args.debug_max}")
    print("=" * 60)

    # Ensure directories exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ae_weights), exist_ok=True)

    # Intermediate calibration files (stored alongside AE weights)
    ae_dir = os.path.dirname(ae_weights)
    parce_file = os.path.join(ae_dir, "parce_calib.pth")
    bins_file = os.path.join(ae_dir, "bin_edges.json")

    # ==========================================================================
    # STEP 0: TRAIN AUTOENCODER (if weights don't exist)
    # ==========================================================================
    if os.path.exists(ae_weights) and not args.force_recalibrate:
        print(f"\n--- AE weights found at {ae_weights}. Skipping training... ---")
    elif args.skip_train_ae:
        print(f"\n--- Skipping AE training (--skip_train_ae flag) ---")
        if not os.path.exists(ae_weights):
            print(f"WARNING: AE weights not found at {ae_weights}")
            print("Pipeline may fail in subsequent steps.")
    else:
        print(f"\n--- Training Autoencoder ---")
        cmd = [
            sys.executable, "-m", "ln_dataset.core.train_ae",
            "--config", args.config,
            "--data", train_data,
            "--imglist", train_list,
            "--save_path", ae_weights,
            "--epochs", str(args.ae_epochs),
            "--batch_size", str(args.ae_batch_size)
        ]
        run_command(cmd, "Step 0: Train Autoencoder")

    # ==========================================================================
    # STEP 1: CALIBRATE PaRCE (per-class reconstruction statistics)
    # ==========================================================================
    if os.path.exists(parce_file) and not args.force_recalibrate:
        print(f"\n--- PaRCE calibration found at {parce_file}. Skipping... ---")
    else:
        print(f"\n--- Calibrating PaRCE on training set ---")
        cmd = [
            sys.executable, "-m", "ln_dataset.core.calibrate_parce",
            "--config", args.config,
            "--data", train_data,
            "--imglist", train_list,
            "--ae_weights", ae_weights,
            "--save_path", parce_file,
            "--samples", str(args.parce_samples)
        ]
        run_command(cmd, "Step 1: Calibrate PaRCE")

    # ==========================================================================
    # STEP 2: CALIBRATE BIN EDGES (severity level thresholds)
    # ==========================================================================
    if os.path.exists(bins_file) and not args.force_recalibrate:
        print(f"\n--- Bin edges found at {bins_file}. Skipping... ---")
    else:
        print(f"\n--- Calibrating bin edges on training set ---")
        cmd = [
            sys.executable, "-m", "ln_dataset.core.calibrate_bins",
            "--config", args.config,
            "--data", train_data,
            "--imglist", train_list,
            "--ae_weights", ae_weights,
            "--parce_calib", parce_file,
            "--save_json", bins_file,
            "--samples", str(args.bins_samples)
        ]
        run_command(cmd, "Step 2: Calibrate Bin Edges")

    # ==========================================================================
    # STEP 3: GENERATE LN DATASET
    # ==========================================================================
    print(f"\n--- Generating LN Dataset ---")
    cmd = [
        sys.executable, "-m", "ln_dataset.core.generate_ln",
        "--config", args.config,
        "--data", val_data,
        "--imglist", val_list,
        "--ae_weights", ae_weights,
        "--parce_calib", parce_file,
        "--bin_edges_json", bins_file,
        "--out_dir", out_dir,
        "--debug_max", str(args.debug_max)
    ]
    run_command(cmd, "Step 3: Generate LN Dataset")

    # ==========================================================================
    # STEP 4: VERIFY MANIFEST
    # ==========================================================================
    manifest_path = os.path.join(out_dir, "imglist.txt")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        print(f"  Status:     SUCCESS")
        print(f"  Manifest:   {manifest_path}")
        print(f"  Images:     {len(lines)}")
    else:
        print(f"  Status:     WARNING")
        print(f"  Manifest not found at {manifest_path}")
        print(f"  Generation may have failed.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
