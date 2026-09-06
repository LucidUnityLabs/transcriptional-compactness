import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

with open(os.path.join(HERE, "LABEL_VALIDATION", "results.json")) as f:
    LV = json.load(f)
with open(os.path.join(HERE, "BATCH_CORRECTION", "results.json")) as f:
    BC = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
fig.subplots_adjust(bottom=0.24, top=0.88, left=0.06, right=0.985, wspace=0.22)

ax = axes[0]
LV_COHORTS = [
    ("prostate_chen_GSE176031", "Chen prostate (GSE176031)"),
    ("ovarian_olalekan_GSE147082", "Olalekan HGSOC (GSE147082)"),
]
SCHEMES = [
    ("original", "original", "#0072B2", -0.28),
    ("alt1_broad", "alt1 (broad)", "#009E73", 0.0),
    ("alt2_cluster", "alt2 (cluster)", "#E69F00", 0.28),
]
cx = [0.0, 1.7]
for (key, label), c in zip(LV_COHORTS, cx):
    block = LV[key]
    ax.hlines(block["published_delta"], c - 0.42, c + 0.42,
              color="0.35", lw=0.9, ls=(0, (3, 2)))
    ax.text(c + 0.44, block["published_delta"],
            "published delta %.2f" % block["published_delta"],
            fontsize=6.3, color="0.35", va="center")
    for scheme, slabel, color, dx in SCHEMES:
        s = block["schemes"][scheme]
        d = block["decomposition_fixed_comparator"][scheme]
        sig = s["p_mwu"] < 0.05
        ax.plot([c + dx, c + dx], [s["delta"], d["delta"]],
                color=color, lw=0.7, alpha=0.55, zorder=1)
        ax.plot(c + dx, s["delta"], "o", ms=6.2, mfc=color if sig else "none",
                mec=color, mew=1.3, zorder=3)
        ax.plot(c + dx, d["delta"], "^", ms=5.2, mfc="none", mec=color,
                mew=1.1, zorder=3)
        ax.annotate("%+.3f" % s["delta"], (c + dx, s["delta"]),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=6.2, color=color)
    ax.text(c, -0.72, label, ha="center", fontsize=7.6)
ax.axhline(0.0, color="0.2", lw=0.8)
ax.set_xlim(-0.62, 2.55)
ax.set_ylim(-0.78, 0.86)
ax.set_xticks([])
ax.set_ylabel("Cliff's delta (kappa)")
ax.set_title("A  Annotation-label sensitivity (3 labelling schemes)", fontsize=9,
             loc="left", fontweight="bold")
ax.plot([], [], "o", ms=6, mfc="#444444", mec="#444444",
        label="delta vs matched comparator (filled: MWU p < 0.05)")
ax.plot([], [], "^", ms=5, mfc="none", mec="#444444",
        label="delta vs fixed comparator (decomposition)")
ax.plot([], [], color="0.35", lw=0.9, ls=(0, (3, 2)),
        label="published delta")
ax.legend(loc="upper right", fontsize=6.3, handlelength=1.6)

ax = axes[1]
BC_COHORTS = [
    ("Tirosh_melanoma", "Tirosh\nmelanoma"),
    ("Puram_HNSCC", "Puram\nHNSCC"),
    ("Darmanis_GBM", "Darmanis\nGBM"),
]
CONDS = [
    ("original", "Original (no correction)", "#1F77B4"),
    ("harmony", "Harmony (patient as batch)", "#D55E00"),
    ("random", "Random-embedding null (mean of 3)", "#C8C8C8"),
    ("umi", "UMI-only embedding", "#8C8C8C"),
]
width = 0.19
offs = {"original": -1.5, "harmony": -0.5, "random": 0.5, "umi": 1.5}
x = list(range(len(BC_COHORTS)))
for cond, label, color in CONDS:
    ys = []
    for key, _ in BC_COHORTS:
        block = BC["cohorts"][key]
        if cond == "random":
            seeds = [e["cliff_delta"] for e in block["random_null"]]
            ys.append(sum(seeds) / len(seeds))
        elif cond == "umi":
            ys.append(block["cliff_delta_umi_only"])
        else:
            ys.append(block["cliff_delta_%s" % cond])
    rects = ax.bar([i + offs[cond] * width for i in x], ys, width=width,
                   color=color, edgecolor="0.2", linewidth=0.5, label=label)
    ax.bar_label(rects, fmt="%+.3f", fontsize=6.0, padding=1.5)
pur = BC["cohorts"]["Puram_HNSCC"]
ax.annotate(
    "sign flip after Harmony\n(%+.3f -> %+.3f)" % (
        pur["cliff_delta_original"], pur["cliff_delta_harmony"]),
    xy=(1 + offs["harmony"] * width, pur["cliff_delta_harmony"]),
    xytext=(1.28, -0.72), fontsize=6.8, ha="center",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.25", shrinkB=2.0),
)
ax.axhline(0.0, color="0.2", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in BC_COHORTS], fontsize=7.6)
ax.set_ylim(-0.92, 0.98)
ax.set_ylabel("Cliff's delta (kappa)")
ax.set_title("B  Batch-correction sensitivity (Harmony vs none)", fontsize=9,
             loc="left", fontweight="bold")
ax.legend(loc="upper right", fontsize=6.3)

out = os.path.join(HERE, "FIGGEN_fig5.png")
fig.savefig(out)
print("wrote", out)
