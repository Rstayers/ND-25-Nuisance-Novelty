import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi
from .config import DIRS, DATASET_COLORS


def plot(df):
    print("Generating 4. Nuisance Sensitivity Radar...")

    # Filter for Nuisance Novelty target: ONLY CORRECT SAMPLES
    # We want to know: "When the model is RIGHT, how scared is it?"
    subset = df[(df['dataset'] == 'imagenet_ln') & (df['correct'] == 1)].copy()
    if subset.empty: return

    # Calculate Mean Confidence per Nuisance
    # Ideally, compare to Clean confidence (~0.95)
    stats = subset.groupby('nuisance')['conf'].mean().reset_index()

    # Prepare Data for Radar
    categories = stats['nuisance'].tolist()
    values = stats['conf'].tolist()
    N = len(categories)

    # Close the loop
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    ax.plot(angles, values, linewidth=2, linestyle='solid', color=DATASET_COLORS['imagenet_ln'])
    ax.fill(angles, values, color=DATASET_COLORS['imagenet_ln'], alpha=0.4)

    # Ticks
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1.0)

    plt.title("Sensitivity Profile: Mean Confidence on Correct Predictions", y=1.1)

    save_path = os.path.join(DIRS['radar'], "sensitivity_radar.png")
    plt.savefig(save_path)
    plt.close()