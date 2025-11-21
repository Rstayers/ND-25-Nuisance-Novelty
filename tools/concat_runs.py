import pandas as pd

# Read the two CSV files
df1 = pd.read_csv('../results/nuisance_runs/cns_bench/nuisance_cns_bench_e1858529.csv')
df2 = pd.read_csv('../results/nuisance_runs/cns_bench/nuisance_cns_bench_5656fb2f.csv')

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the result to a new CSV file
combined_df.to_csv('results/nuisance_runs/cns_bench/combined.csv', index=False)