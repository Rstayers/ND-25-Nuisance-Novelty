import pandas as pd

df = pd.read_csv("results/nuisance_runs/nuisance_full_set.csv")

# Check duplicates by image_path only
subset_cols = ["image_path", "detector", "backbone", "corruption", "severity", "fpr_target"]
dupes = df[df.duplicated(subset=subset_cols, keep=False)]
if dupes.empty:
    print("✅ No duplicate test cases found.")
else:
    print(f"⚠️ Found {dupes.shape[0]} duplicate rows.")

    # Group and print with row indices
    for key, group in dupes.groupby(subset_cols):
        print(f"\nDuplicate group: {key}")
        print("Row indices:", list(group.index))
        print(group.to_string())
