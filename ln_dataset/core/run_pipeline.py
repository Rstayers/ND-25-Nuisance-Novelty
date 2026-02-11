import os
import sys
import subprocess
import argparse
from ln_dataset.core.configs import load_config


def run_command(cmd, step_name):
    print(f"\n[Step {step_name}] Running...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    # CLI overrides
    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--ae_weights', type=str, default=None)

    # Debug Flag
    parser.add_argument('--debug_max', type=str, default="0",
                        help="Number of debug images to save during generation (default: 0)")

    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)

    # 2. Resolve Paths (CORRECTED)
    # Use the data paths from the config file, default to "." only if missing
    train_data = cfg.paths.train_data if cfg.paths.train_data else "."
    train_list = args.train_list or cfg.paths.train_list

    val_data = cfg.paths.val_data if cfg.paths.val_data else "."
    val_list = args.val_list or cfg.paths.val_list

    out_dir = args.out_dir or cfg.paths.out_dir
    ae_weights = args.ae_weights or cfg.paths.ae_weights

    if not train_list or not val_list or not out_dir:
        print("Error: Missing required paths (train_list, val_list, or out_dir). Check your config or CLI args.")
        sys.exit(1)

    print("=" * 40)
    print("PIPELINE CONFIGURATION")
    print(f" Train Data: {train_data}")
    print(f" Train List: {train_list}")
    print(f" Val Data:   {val_data}")
    print(f" Val List:   {val_list}")
    print(f" AE Weights: {ae_weights}")
    print(f" Out Dir:    {out_dir}")
    print(f" Debug Max:  {args.debug_max}")
    print("=" * 40)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Intermediate files
    parce_file = os.path.join(os.path.dirname(ae_weights), "parce_calib.pth")
    bins_file = os.path.join(os.path.dirname(ae_weights), "bin_edges.json")

    # ==============================================================================
    # STEP 1: CALIBRATE PaRCE (Using TRAIN Data)
    # ==============================================================================
    if os.path.exists(parce_file):
        print(f"--- PaRCE Calibration found at {parce_file}. Skipping... ---")
    else:
        print(f"--- Calibrating PaRCE on TRAIN set ---")
        cmd = [
            sys.executable, "-m", "ln_dataset.calibrate_parce",
            "--config", args.config,
            "--data", train_data,  # <--- Now uses correct path
            "--imglist", train_list,
            "--ae_weights", ae_weights,
            "--save_path", parce_file,
            "--samples", "20000"
        ]
        run_command(cmd, "1 (Calibrate PaRCE)")

    # ==============================================================================
    # STEP 2: CALIBRATE BINS (Using TRAIN Data)
    # ==============================================================================
    if os.path.exists(bins_file):
        print(f"--- Bin Edges found at {bins_file}. Skipping... ---")
    else:
        print(f"--- Calibrating Bins on TRAIN set ---")
        cmd = [
            sys.executable, "-m", "ln_dataset.calibrate_bins",
            "--config", args.config,
            "--data", train_data,
            "--imglist", train_list,
            "--ae_weights", ae_weights,
            "--parce_calib", parce_file,
            "--save_json", bins_file,
            "--samples", "5000"
        ]
        run_command(cmd, "2 (Calibrate Bins)")

    # ==============================================================================
    # STEP 3: GENERATE DATASET (Using VAL Data)
    # ==============================================================================
    print(f"--- Generating LN Dataset ---")

    cmd = [
        sys.executable, "-m", "ln_dataset.generate_ln",
        "--config", args.config,
        "--data", val_data,
        "--imglist", val_list,
        "--ae_weights", ae_weights,
        "--parce_calib", parce_file,
        "--bin_edges_json", bins_file,
        "--out_dir", out_dir,
        "--debug_max", args.debug_max
    ]
    run_command(cmd, "3 (Generate)")

    # ==============================================================================
    # STEP 4: VERIFY MANIFEST
    # ==============================================================================
    manifest_path = os.path.join(out_dir, "imglist.txt")
    if os.path.exists(manifest_path):
        print(f"\nSUCCESS: Dataset generated.")
        print(f"Manifest saved at: {manifest_path}")
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        print(f"Total images generated: {len(lines)}")
    else:
        print(f"\nWARNING: Manifest not found at {manifest_path}. Generation might have failed.")


if __name__ == "__main__":
    main()