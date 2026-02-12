# /path/to/plot_rq2_stability.py

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

PERTS = ["Misspelling", "Ordering", "Synonym", "Paraphrase", "Naturality"]
SPLITS = ["DL19", "DL20", "Dev"]

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8

cand = {
    "DL19": [
        dict(mean=0.3743, std=0.0203, med=0.3773, q1=0.1120, q3=0.6011, whislo=0.0149, whishi=0.7217),
        dict(mean=0.8012, std=0.0173, med=0.8417, q1=0.7218, q3=0.9139, whislo=0.5802, whishi=0.9541),
        dict(mean=0.4387, std=0.0238, med=0.3935, q1=0.1984, q3=0.6877, whislo=0.0399, whishi=0.8857),
        dict(mean=0.6246, std=0.0200, med=0.6925, q1=0.4027, q3=0.8693, whislo=0.1521, whishi=0.9313),
        dict(mean=0.7382, std=0.0000, med=0.7857, q1=0.5444, q3=0.9231, whislo=0.4267, whishi=1.0000),
    ],
    "DL20": [
        dict(mean=0.3480, std=0.0203, med=0.2908, q1=0.1439, q3=0.5264, whislo=0.0628, whishi=0.6897),
        dict(mean=0.8014, std=0.0043, med=0.8366, q1=0.7332, q3=0.8995, whislo=0.6169, whishi=0.9471),
        dict(mean=0.4985, std=0.0592, med=0.5076, q1=0.2653, q3=0.7207, whislo=0.1444, whishi=0.8951),
        dict(mean=0.5686, std=0.0232, med=0.6172, q1=0.3557, q3=0.8443, whislo=0.1255, whishi=0.9252),
        dict(mean=0.7079, std=0.0000, med=0.7316, q1=0.5748, q3=0.8519, whislo=0.3918, whishi=1.0000),
    ],
    "Dev": [
        dict(mean=0.3118, std=0.0025, med=0.2610, q1=0.0893, q3=0.4948, whislo=0.0152, whishi=0.6949),
        dict(mean=0.7667, std=0.0023, med=0.8051, q1=0.6807, q3=0.8868, whislo=0.5314, whishi=0.9417),
        dict(mean=0.4604, std=0.0019, med=0.4409, q1=0.1696, q3=0.7391, whislo=0.0320, whishi=0.9121),
        dict(mean=0.5726, std=0.0027, med=0.6461, q1=0.3089, q3=0.8519, whislo=0.0776, whishi=0.9380),
        dict(mean=0.6798, std=0.0000, med=0.7094, q1=0.5152, q3=0.8692, whislo=0.3333, whishi=1.0000),
    ],
}

tok = {
    "DL19": [
        dict(mean=0.3523, std=0.0170, med=0.3283, q1=0.2250, q3=0.4703, whislo=0.1582, whishi=0.5511),
        dict(mean=0.6825, std=0.0214, med=0.6983, q1=0.5939, q3=0.7780, whislo=0.4945, whishi=0.8453),
        dict(mean=0.4222, std=0.0253, med=0.3671, q1=0.2581, q3=0.5379, whislo=0.1841, whishi=0.7769),
        dict(mean=0.5510, std=0.0103, med=0.5521, q1=0.3524, q3=0.7168, whislo=0.3021, whishi=0.7871),
        dict(mean=0.6358, std=0.0000, med=0.6000, q1=0.4652, q3=0.7621, whislo=0.4205, whishi=1.0000),
    ],
    "DL20": [
        dict(mean=0.3632, std=0.0157, med=0.3484, q1=0.2351, q3=0.4645, whislo=0.1753, whishi=0.5869),
        dict(mean=0.6836, std=0.0089, med=0.6936, q1=0.5777, q3=0.7964, whislo=0.5180, whishi=0.8494),
        dict(mean=0.4853, std=0.0356, med=0.4388, q1=0.3040, q3=0.6477, whislo=0.2094, whishi=0.8305),
        dict(mean=0.5218, std=0.0157, med=0.5562, q1=0.3346, q3=0.7206, whislo=0.2316, whishi=0.7818),
        dict(mean=0.5769, std=0.0000, med=0.5504, q1=0.4311, q3=0.6914, whislo=0.3423, whishi=1.0000),
    ],
    "Dev": [
        dict(mean=0.3356, std=0.0019, med=0.3123, q1=0.2077, q3=0.4388, whislo=0.1429, whishi=0.5625),
        dict(mean=0.6613, std=0.0018, med=0.6807, q1=0.5687, q3=0.7699, whislo=0.4625, whishi=0.8349),
        dict(mean=0.4748, std=0.0015, med=0.4388, q1=0.2755, q3=0.6529, whislo=0.1723, whishi=0.8315),
        dict(mean=0.5363, std=0.0026, med=0.5601, q1=0.3699, q3=0.7182, whislo=0.2136, whishi=0.8182),
        dict(mean=0.5844, std=0.0000, med=0.5625, q1=0.4286, q3=0.7094, whislo=0.3158, whishi=1.0000),
    ],
}


def plot_grouped_bxp(metric_dict, title, ylabel, figsize=(7.0, 2.8), stem=None):
    """
    Grouped boxplots from precomputed quantiles (p10/p25/median/p75/p90),
    with mean overlay + ±std error bars for the mean.
    """
    fig, ax = plt.subplots(figsize=figsize)

    base = np.arange(len(PERTS))
    offsets = np.array([-0.26, 0.0, 0.26])
    width = 0.16

    linestyles = {"DL19": "-", "DL20": "--", "Dev": ":"}
    # Okabe-Ito palette (colorblind-safe, print-friendly)
    colors = {"DL19": "#0072B2", "DL20": "#E69F00", "Dev": "#009E73"}

    for si, split in enumerate(SPLITS):
        stats, means, mean_stds = [], [], []

        for d in metric_dict[split]:
            stats.append(
                {
                    "label": "",
                    "med": d["med"],
                    "q1": d["q1"],
                    "q3": d["q3"],
                    "whislo": d["whislo"],
                    "whishi": d["whishi"],
                }
            )
            means.append(d["mean"])
            mean_stds.append(d["std"])

        pos = base + offsets[si]

        out = ax.bxp(
            stats,
            positions=pos,
            widths=width,
            showfliers=False,
            manage_ticks=False,
            patch_artist=True,  # <= allow filled boxes
        )

        # Style all elements consistently for this split
        for box in out["boxes"]:
            box.set_facecolor(colors[split])
            box.set_alpha(0.26)  # increase visibility while keeping overlap readable
            box.set_edgecolor(colors[split])
            box.set_linestyle(linestyles[split])
            box.set_linewidth(1.25)

        for whisker in out["whiskers"]:
            whisker.set_color(colors[split])
            whisker.set_linestyle(linestyles[split])
            whisker.set_linewidth(1.1)

        for cap in out["caps"]:
            cap.set_color(colors[split])
            cap.set_linestyle(linestyles[split])
            cap.set_linewidth(1.1)

        for median in out["medians"]:
            median.set_color("#1A1A1A")
            median.set_linewidth(1.45)

        ax.errorbar(
            pos,
            means,
            yerr=mean_stds,
            fmt="o",
            color=colors[split],      # <= marker color matches split
            ecolor=colors[split],     # <= errorbar color matches split
            linestyle="None",
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.5,
            elinewidth=1.0,
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(base)
    ax.set_xticklabels(PERTS, rotation=0, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    handles = [
        Line2D([0], [0], color=colors["DL19"], linestyle=linestyles["DL19"], label="DL19"),
        Line2D([0], [0], color=colors["DL20"], linestyle=linestyles["DL20"], label="DL20"),
        Line2D([0], [0], color=colors["Dev"], linestyle=linestyles["Dev"], label="Dev"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="Mean ± std", markersize=4),
    ]
    ax.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=12,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )

    fig.tight_layout()

    if stem is not None:
        fig.savefig(os.path.join(OUTDIR, f"{stem}.pdf"), bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    plot_grouped_bxp(
        cand,
        title="RQ2 Stability: Candidate Overlap",
        ylabel="CandOverlap@100",
        stem="rq2_candoverlap_bxp",
    )
    plot_grouped_bxp(
        tok,
        title="RQ2 Stability: Token Jaccard",
        ylabel="Token Jaccard@100",
        stem="rq2_tokjaccard_bxp",
    )
    plt.close("all")
