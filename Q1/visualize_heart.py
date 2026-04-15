import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
 
def plot_distributions(input_file, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
 
    df = pd.read_csv(input_file)
 
    cols = df.columns.tolist()
    n = len(cols)
    ncols = 4
    nrows = -(-n // ncols)  # ceiling division
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
 
    for i, col in enumerate(cols):
        counts = df[col].value_counts().sort_index()
        axes[i].bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")
        axes[i].set_title(f"Distribution of {col}", fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)
        axes[i].tick_params(axis="x", rotation=45)
 
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
 
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distributions.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


input_path = "base_Heart-data.csv"
output_dir = "heart-plots"
plot_distributions(input_path, output_dir)