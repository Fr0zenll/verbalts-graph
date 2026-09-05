#!/usr/bin/env python3
"""Regenerate the two paper figures as vector PDFs (IEEE camera-ready quality).

Fig 1 (graph_heatmaps_latest.pdf): Pearson corr matrix + Pearson/Identity/Random graphs (2x2 heatmaps).
Fig 2 (sample_comparison_latest.pdf): Real vs VerbalTS vs Graph-VerbalTS curves on one held-out window,
     rows = eth1_pi (network throughput), cpu_r (CPU util), load_1 (load), curr_estab (connections).
Outputs land directly in ../paper/figures/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType (IEEE requirement)
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
})

ROOT = "/data/kangjiale/verbalts-graph"
DATA = f"{ROOT}/data/smd_nl"
SAMPLES = f"{ROOT}/VerbalTS-main/save/paper_samples/run0_start0/smd_nl_paper_samples.npz"
OUT = f"{ROOT}/paper/figures"

VAR_NAMES = ["cpu_r", "load_1", "mem_u_e", "disk_r", "disk_u", "disk_w",
             "eth1_pi", "eth1_po", "tcp_use", "curr_estab"]

# ---------------------------------------------------------------- figure 1
corr_raw = np.load(f"{DATA}/graph_corr_raw.npy")        # raw |Pearson| matrix
g_pear = np.load(f"{DATA}/graph_adj.npy")               # top-k + self-loop + GCN norm
g_ident = np.load(f"{DATA}/graph_adj_identity.npy")
g_rand = np.load(f"{DATA}/graph_adj_random.npy")

panels = [
    (r"|Pearson| correlation", corr_raw, "viridis"),
    ("Pearson graph (top-$k$, norm.)", g_pear, "viridis"),
    ("Identity graph", g_ident, "viridis"),
    ("Random graph", g_rand, "viridis"),
]

fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))
for ax, (title, mat, cmap) in zip(axes.flat, panels):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(mat.max(), 1e-6))
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(VAR_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(VAR_NAMES)
    ax.set_title(title, pad=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6, width=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/graph_heatmaps_latest.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote graph_heatmaps_latest.pdf")

# ---------------------------------------------------------------- figure 2
d = np.load(SAMPLES, allow_pickle=True)
real = d["real"]            # (8, 240, 10)
base = d["baseline_pred"]   # VerbalTS
graph = d["pearson_pred"]   # Graph-VerbalTS

# window with visible network anomaly + tcp/co-movement (matches paper narrative)
w = 4
rows = [(6, "Network throughput (eth1_pi)"),
        (0, "CPU utilization (cpu_r)"),
        (1, "Load (load_1)"),
        (9, "Established connections (curr_estab)")]

fig, axes = plt.subplots(4, 3, figsize=(6.8, 4.6), sharex="col")
for r, (vi, label) in enumerate(rows):
    for c, (arr, tag) in enumerate([(real, "Real"), (base, "VerbalTS"), (graph, "Graph-VerbalTS")]):
        ax = axes[r, c]
        ax.plot(arr[w, :, vi], lw=0.8, color="#1f77b4" if c == 0 else ("#d62728" if c == 1 else "#2ca02c"))
        if r == 0:
            ax.set_title(tag, pad=3)
        if c == 0:
            ax.set_ylabel(label, fontsize=6.5)
        if r == 3:
            ax.set_xlabel("Time step", fontsize=7)
        ax.tick_params(length=2, width=0.5)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
fig.tight_layout(h_pad=0.7, w_pad=0.5)
fig.savefig(f"{OUT}/sample_comparison_latest.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote sample_comparison_latest.pdf")
print("window", w, "attrs", d["attrs"][w].tolist())
