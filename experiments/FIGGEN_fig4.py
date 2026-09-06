import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "CLASSIFIER_AUC")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

with open(os.path.join(R, "per_cohort_auc.csv")) as f:
    within = list(csv.DictReader(f))
with open(os.path.join(R, "leave_one_out_auc.csv")) as f:
    loco_rows = list(csv.DictReader(f))
loco = {r["held_out_cohort"]: r for r in loco_rows}
with open(os.path.join(R, "results.json")) as f:
    summary = json.load(f)["summary"]

order = sorted(within, key=lambda r: -float(r["kappa_auc_roc_mean"]))
cohorts = [r["cohort"] for r in order]
li_idx = cohorts.index("Li_CRC")
xticklabels = [c.replace("_", " ") for c in cohorts]
if li_idx is not None:
    xticklabels[li_idx] += "\n(delta = -0.78, reversed)"


def col(rows, name):
    return [float(r[name]) if r.get(name) not in (None, "") else None for r in rows]


FEATURES = [
    ("kappa (Ollivier-Ricci)", "kappa_auc_roc_mean", "kappa_auc_roc_std", "#0072B2", "o"),
    ("multi-feature", "multi_auc_roc_mean", "multi_auc_roc_std", "#CC79A7", "s"),
    ("n_genes", "n_genes_auc_roc_mean", None, "#009E73", "^"),
    ("UMI", "umi_auc_roc_mean", "umi_auc_roc_std", "#E69F00", "D"),
    ("random feature", "random_auc_roc_mean", "random_auc_roc_std", "#999999", "o"),
]
OFFSETS = {
    "kappa_auc_roc_mean": -0.24,
    "multi_auc_roc_mean": -0.12,
    "n_genes_auc_roc_mean": 0.0,
    "umi_auc_roc_mean": 0.12,
    "random_auc_roc_mean": 0.24,
}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
fig.subplots_adjust(bottom=0.3, top=0.88, left=0.06, right=0.985, wspace=0.22)

handles = []
ax = axes[0]
x = list(range(len(order)))
for name, mcol, scol, color, marker in FEATURES:
    xs = [i + OFFSETS[mcol] for i in x]
    ys = col(order, mcol)
    yerr = col(order, scol) if scol else None
    opened = name == "random feature"
    h = ax.errorbar(
        xs, ys, yerr=yerr, fmt=marker, ms=4.4,
        mfc="none" if opened else color, mec=color, mew=1.0,
        ecolor=color, elinewidth=0.8, capsize=0, lw=0, label=name,
    )
    handles.append(h)
ax.axhline(0.5, color="0.35", lw=0.7, ls=":")
ax.axhline(summary["kappa_within_auc_mean"], color="#0072B2", lw=0.8,
           ls=(0, (4, 3)), alpha=0.65)
ax.text(6.45, summary["kappa_within_auc_mean"] + 0.006,
        "kappa mean 0.734", color="#0072B2", fontsize=6.5, ha="right")
ax.annotate(
    "highest AUC but direction reversed\n(T cells higher kappa than epithelial)",
    xy=(li_idx + OFFSETS["kappa_auc_roc_mean"], 0.902),
    xytext=(li_idx + 0.42, 0.965), fontsize=6.8, ha="left", va="top",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.25", shrinkB=2.5),
)
ax.set_ylim(0.38, 1.0)
ax.set_ylabel("AUC-ROC")
ax.set_title("A  Within-cohort (10 seeds, 80/20 stratified)", fontsize=9,
             loc="left", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=7.2)

ax = axes[1]
LOCO_FEATURES = [
    ("kappa (Ollivier-Ricci)", "kappa_auc_roc", "#0072B2", "o"),
    ("multi-feature", "multi_auc_roc", "#CC79A7", "s"),
    ("UMI", "umi_auc_roc", "#E69F00", "D"),
]
loco_order = [loco[c] for c in cohorts]
for name, mcol, color, marker in LOCO_FEATURES:
    xs = [i + OFFSETS.get(mcol + "_mean", {"kappa_auc_roc": -0.22, "multi_auc_roc": 0.0, "umi_auc_roc": 0.22}[mcol]) for i in x]
    ys = col(loco_order, mcol)
    ax.plot(xs, ys, marker, ms=4.6, mfc=color, mec=color, lw=0, label=name)
ax.axhline(0.5, color="0.35", lw=0.7, ls=":")
ax.axhline(summary["kappa_loco_auc_mean"], color="#0072B2", lw=0.8,
           ls=(0, (4, 3)), alpha=0.65)
ax.text(6.45, summary["kappa_loco_auc_mean"] + 0.006,
        "kappa mean 0.619", color="#0072B2", fontsize=6.5, ha="right")
ax.annotate(
    "0.112: reversed direction of the Li cohort\nbreaks cross-cohort transfer",
    xy=(li_idx - 0.22, 0.112),
    xytext=(li_idx + 0.55, 0.155), fontsize=6.8, ha="left", va="bottom",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.25", shrinkB=2.5),
)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("held-out AUC-ROC")
ax.set_title("B  Leave-one-cohort-out (train 6, test 1)", fontsize=9,
             loc="left", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=7.2)

fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.6,
           bbox_to_anchor=(0.5, 0.02), handletextpad=0.3, columnspacing=1.2)

out = os.path.join(HERE, "FIGGEN_fig4.png")
fig.savefig(out)
print("wrote", out)
