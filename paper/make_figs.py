"""Generate paper figures from the released result CSVs. Pure matplotlib, no seaborn, no styles."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------- Fig 1: three regimes bar chart ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
labels = ["Simple\navg", "Task\nArith.", "TIES", "OT\nbary.", "SVD-TIES\n(control)", "OT-TIES\n(ours)",
          "KnOTS", "Core\nSpace", "GeoMerge"]
vals = [0.636, 0.639, 0.634, 0.622, 0.687, 0.708, 0.740, 0.764, 0.771]
colors = ["#bbbbbb"]*3 + ["#d98c3f"] + ["#6a9fd8", "#2e6db4"] + ["#7bbf7b"]*3
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0.64, ls="--", c="gray", lw=0.8)
ax.text(0.1, 0.645, "naive floor", color="gray", fontsize=8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("avg. normalized accuracy")
ax.set_ylim(0.55, 0.80)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
ax.set_title("Three regimes: averaging $\\to$ resolution $\\to$ OT subspace", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regimes.pdf"))
plt.close(fig)

# ---------------- Fig 2: OT vs SVD subspace across lambda ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
lam = [0.3, 0.4, 0.5, 0.6]
svd = [0.658, 0.672, 0.682, 0.687]
ot = [0.665, 0.682, 0.696, 0.706]
ax.plot(lam, svd, "o--", color="#6a9fd8", label="SVD subspace (control)")
ax.plot(lam, ot, "s-", color="#2e6db4", label="OT subspace (ours)")
ax.axhline(0.64, ls=":", c="gray", lw=0.8); ax.text(0.305, 0.643, "floor", color="gray", fontsize=8)
ax.set_xlabel("scaling $\\lambda$"); ax.set_ylabel("avg. normalized accuracy")
ax.set_title("OT subspace beats SVD at every $\\lambda$\n(identical TIES, $\\kappa{=}0.2$)", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ot_vs_svd.pdf"))
plt.close(fig)

# ---------------- Fig 3: rank sweep is flat (negative result) ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ranks = [16, 32, 64, 128]
bary = [0.622, 0.625, 0.627, 0.626]
ax.plot(ranks, bary, "D-", color="#d98c3f", label="OT barycenter (M-bary)")
ax.axhline(0.636, ls="--", c="gray", lw=0.8); ax.text(18, 0.638, "simple-average floor", color="gray", fontsize=8)
ax.axhline(0.708, ls="-.", c="#2e6db4", lw=0.8); ax.text(18, 0.700, "OT-TIES (ours)", color="#2e6db4", fontsize=8)
ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
ax.set_xlabel("barycenter target rank $R$"); ax.set_ylabel("avg. normalized accuracy")
ax.set_ylim(0.55, 0.73)
ax.set_title("OT averaging plateaus at the floor\nregardless of rank (negative result)", fontsize=10)
ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rank_sweep.pdf"))
plt.close(fig)

# ---------------- Fig 4: per-task heatmap (interference signature) ----------------
tasks = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
rows = ["Simple avg", "TA $\\lambda$=0.1", "TIES mean", "OT align", "OT bary."]
M = np.array([
    [0.623,0.609,0.628,0.455,0.427,0.405,0.531,0.424],
    [0.628,0.609,0.633,0.483,0.405,0.392,0.535,0.431],
    [0.627,0.612,0.614,0.495,0.405,0.340,0.563,0.428],
    [0.627,0.615,0.613,0.431,0.341,0.342,0.491,0.431],
    [0.626,0.607,0.617,0.460,0.378,0.353,0.520,0.428],
])
fig, ax = plt.subplots(figsize=(7.2, 3.0))
im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.7)
ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="per-task acc.")
ax.set_title("Interference signature: SVHN/GTSRB/EuroSAT collapse under averaging", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pertask_heatmap.pdf"))
plt.close(fig)

# ---------------- Fig 5: heterogeneous-rank (graceful degradation) ----------------
fig, ax = plt.subplots(figsize=(5.6, 3.4))
labels = ["OT-TIES\nhomo r16", "OT-TIES\nmixed {4,8,16}", "GW\nmixed {4,8,16}"]
vals = [0.708, 0.698, 0.623]
colors = ["#2e6db4", "#5a8fc4", "#9ec4e0"]
ax.bar(range(3), vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0.64, ls="--", c="gray", lw=0.8); ax.text(0.0, 0.645, "naive floor", color="gray", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("avg. normalized accuracy"); ax.set_ylim(0.55, 0.74)
for i, v in enumerate(vals):
    ax.text(i, v+0.004, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("Heterogeneous-rank merging: graceful $-1.0$pt\n(rank-fixed baselines cannot run here)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hetero.pdf"))
plt.close(fig)

print("wrote figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
