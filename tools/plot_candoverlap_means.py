import os
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Categories and Datasets
PERTS = ["Misspelling", "Ordering", "Synonym", "Paraphrase", "Naturality"]
SPLITS = ["DL19", "DL20", "Dev"]

# Set font types for editable text in vector graphics (optional)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Data (Mean and Std extracted from your previous snippet)
cand = {
    "DL19": [
        {"mean": 0.3743, "std": 0.0203},
        {"mean": 0.8012, "std": 0.0173},
        {"mean": 0.4387, "std": 0.0238},
        {"mean": 0.6246, "std": 0.0200},
        {"mean": 0.7382, "std": 0.0000},
    ],
    "DL20": [
        {"mean": 0.3480, "std": 0.0203},
        {"mean": 0.8014, "std": 0.0043},
        {"mean": 0.4985, "std": 0.0592},
        {"mean": 0.5686, "std": 0.0232},
        {"mean": 0.7079, "std": 0.0000},
    ],
    "Dev": [
        {"mean": 0.3118, "std": 0.0025},
        {"mean": 0.7667, "std": 0.0023},
        {"mean": 0.4604, "std": 0.0019},
        {"mean": 0.5726, "std": 0.0027},
        {"mean": 0.6798, "std": 0.0000},
    ],
}

def plot_point_plot(metric_dict, ylabel, title=None, stem="cand_pointplot"):
    """
    Generates a point plot with error bars to visualize trends and mean differences.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # X-axis positions
    base = np.arange(len(PERTS))
    
    # Offsets to separate the lines slightly (jitter)
    offsets = {"DL19": -0.15, "DL20": 0.0, "Dev": 0.15}
    
    # Styling
    colors = {"DL19": "#1f77b4", "DL20": "#ff7f0e", "Dev": "#2ca02c"}
    markers = {"DL19": "o", "DL20": "s", "Dev": "^"}       # Circle, Square, Triangle
    linestyles = {"DL19": "-", "DL20": "--", "Dev": "-."}  # Solid, Dashed, Dash-dot

    # Loop through datasets and plot
    for split in SPLITS:
        data = metric_dict[split]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]
        
        pos = base + offsets[split]
        
        # 1. Plot Error Bars
        ax.errorbar(
            pos, 
            means, 
            yerr=stds, 
            fmt='none',           # No connecting line from errorbar itself
            ecolor=colors[split], # Error bar color
            elinewidth=2,         # Thicker error bars for visibility
            capsize=5,            # Width of the caps on error bars
            zorder=5              # Draw on top
        )
        
        # 2. Plot Points and Connecting Lines (The "Trend")
        ax.plot(
            pos, 
            means, 
            marker=markers[split], 
            linestyle=linestyles[split], 
            linewidth=1.5, 
            color=colors[split], 
            label=split,
            markersize=8,
            zorder=6
        )

    # Formatting
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(base)
    ax.set_xticklabels(PERTS, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    
    # Add grid for readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Clean Legend
    ax.legend(
        loc="upper left", 
        title="Dataset", 
        frameon=True, 
        fontsize=10, 
        title_fontsize=11
    )
    
    if title:
        ax.set_title(title, fontsize=14, pad=15)

    plt.tight_layout()
    
    # Save output
    outfile = os.path.join(OUTDIR, f"{stem}.pdf")
    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved figure to {outfile}")
    plt.show()

# Run the plotting function
plot_point_plot(cand, "CandOverlap@100", title="Mean CandOverlap Across Categories")