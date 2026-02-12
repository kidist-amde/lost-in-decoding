import os
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8

# Stage 2 (PAG end-to-end), faithful to the LaTeX table values.
languages = ["nl", "fr", "de", "zh"]
methods = ["Seq-only", "Naive", "Aligned", "Translate"]

mrr_data = {
    "Seq-only": [0.046, 0.043, 0.045, 0.013],
    "Naive": [0.090, 0.097, 0.102, 0.027],
    "Aligned": [0.107, 0.156, 0.151, 0.030],
    "Translate": [0.230, 0.221, 0.224, 0.160],
}

ndcg_data = {
    "Seq-only": [0.051, 0.049, 0.052, 0.015],
    "Naive": [0.102, 0.113, 0.116, 0.031],
    "Aligned": [0.124, 0.179, 0.173, 0.035],
    "Translate": [0.263, 0.252, 0.256, 0.182],
}

colors = {
    "Seq-only": "#D65F5F",
    "Naive": "#4878CF",
    "Aligned": "#ECA539",
    "Translate": "#6ACC65",
}
hatches = {"Seq-only": "//", "Naive": "", "Aligned": "xx", "Translate": ".."}

x = np.arange(len(languages))
width = 0.18
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)


def plot_bars(axis, data):
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            data[method],
            width * 0.92,
            label=method,
            color=colors[method],
            hatch=hatches[method],
            edgecolor="#222222",
            linewidth=0.4,
            zorder=3,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(languages, fontsize=12)
    axis.set_ylim(0.0, 0.30)
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="y", labelsize=12)


plot_bars(ax[0], mrr_data)
plot_bars(ax[1], ndcg_data)

ax[0].set_ylabel("Score", fontsize=12)
ax[0].set_xlabel("MRR@10", fontsize=12)
ax[1].set_xlabel("NDCG@10", fontsize=12)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    fontsize=12,
    frameon=False,
    columnspacing=1.0,
    handlelength=1.5,
)

fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(os.path.join(OUTDIR, "rq3_crosslingual_bar.pdf"), bbox_inches="tight")
print("Saved: figures/rq3_crosslingual_bar.pdf")
plt.show()
