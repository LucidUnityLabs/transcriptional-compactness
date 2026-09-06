"""
E2: Forman-Ricci curvature on the kNN graph of cells.

Independent geometric corroboration of the Ollivier-Ricci finding from Path 1
(malignant > non-malignant local curvature). Forman-Ricci is a discrete
curvature that captures degree/edge-weight signal but, per Samal et al. 2018,
misses the triangle (clustering) signal that Ollivier-Ricci picks up.

Formula (Sreejith et al. 2016, edge-based, w_u = w_v = 1 unweighted nodes):
    Ric_F(e) = 2 w_e
              - w_e * sum_{e_u ~ e}    1/sqrt(w_e * w_{e_u})
              - w_e * sum_{e_v ~ e}    1/sqrt(w_e * w_{e_v})
where e = (u, v), edge weights w_e are Euclidean distance in PCA-50 space.
Per-cell mean FR = mean of FR over edges incident to that cell.

Datasets:
  - Tirosh 2016 melanoma (GSE72056) malignant vs T cells
  - Darmanis 2017 GBM (GSE84465) neoplastic vs immune cells

Comparison vs Ollivier-Ricci:
  Tirosh OR    : mal kappa = +0.120 vs T kappa = -0.054 (delta ~ 0.17)
  Darmanis OR  : neo kappa = +0.081 vs imm kappa = +0.010 (delta ~ 0.07)
"""

import re
import sys
import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data"
RNG = np.random.default_rng(20260507)


# ---------------- kNN graph (copied from path1_hyperbolic/run_ricci.py) ------
def build_knn_graph(X, k=15):
    """Symmetric kNN graph as networkx Graph with Euclidean edge weights."""
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, inds = nbr.kneighbors(X)
    G = nx.Graph()
    n = X.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = inds[i, j_idx]
            d = float(dists[i, j_idx])
            if not G.has_edge(i, j) or G[i][j].get("weight", np.inf) > d:
                G.add_edge(i, j, weight=d)
    return G


# ---------------- Forman-Ricci ----------------------------------------------
def forman_ricci_edges(G):
    """
    Forman-Ricci on every edge. With unit node weights and w_e = edge weight:
        Ric_F(e=(u,v)) = 2 w_e
                       - w_e * sum_{e_u ~ e, e_u != e} 1/sqrt(w_e * w_{e_u})
                       - w_e * sum_{e_v ~ e, e_v != e} 1/sqrt(w_e * w_{e_v})
    Returns dict edge -> Ric_F.
    """
    out = {}
    # Pre-grab adjacency once
    nbrs = {n: list(G[n].items()) for n in G.nodes()}  # node -> [(other, dat),..]
    for (u, v, data) in G.edges(data=True):
        w_e = float(data["weight"])
        if w_e <= 0 or not np.isfinite(w_e):
            out[(u, v)] = np.nan
            continue
        # incident edges to u other than (u,v)
        f_u = 0.0
        for other, d in nbrs[u]:
            if other == v:
                continue
            w_eu = float(d["weight"])
            if w_eu <= 0 or not np.isfinite(w_eu):
                continue
            f_u += 1.0 / np.sqrt(w_e * w_eu)
        f_v = 0.0
        for other, d in nbrs[v]:
            if other == u:
                continue
            w_ev = float(d["weight"])
            if w_ev <= 0 or not np.isfinite(w_ev):
                continue
            f_v += 1.0 / np.sqrt(w_e * w_ev)
        ric = 2.0 * w_e - w_e * (f_u + f_v)
        out[(u, v)] = float(ric)
    return out


def per_cell_mean_curvature(G, edge_curv):
    """Mean of incident-edge curvatures per cell."""
    by_cell = {n: [] for n in G.nodes()}
    for (u, v), c in edge_curv.items():
        if np.isnan(c):
            continue
        by_cell[u].append(c)
        by_cell[v].append(c)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by_cell.items()}


# ---------------- Ollivier-Ricci (subsampled, for OR-vs-FR scatter) ----------
def ollivier_ricci_edges(G, alpha=0.5, n_edges=None, rng=None):
    """Ollivier-Ricci on (sampled) edges; same routine as path1_hyperbolic."""
    if rng is None:
        rng = np.random.default_rng(0)
    edges = list(G.edges())
    if n_edges is not None and n_edges < len(edges):
        idx = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in idx]
    out = {}
    for (u, v) in edges:
        nu = list(G.neighbors(u))
        nv = list(G.neighbors(v))
        sup_u = [u] + nu
        sup_v = [v] + nv
        w_u = (np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu))
               if nu else np.array([1.0]))
        w_v = (np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv))
               if nv else np.array([1.0]))
        sub_nodes = list(set(sup_u + sup_v))
        sub = G.subgraph(sub_nodes)
        try:
            dists = {n: nx.single_source_dijkstra_path_length(sub, n)
                     for n in sup_u}
        except Exception:
            out[(u, v)] = np.nan
            continue
        C = np.zeros((len(sup_u), len(sup_v)))
        for i, a in enumerate(sup_u):
            for j, b in enumerate(sup_v):
                C[i, j] = dists[a].get(b, np.inf)
        if not np.all(np.isfinite(C)):
            out[(u, v)] = np.nan
            continue
        W = ot.emd2(w_u, w_v, C)
        d_uv = G[u][v]["weight"]
        out[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan
    return out


# ---------------- effect size -----------------------------------------------
def cliffs_delta(a, b):
    """Cliff's delta = (#a>b - #a<b) / (na*nb). O(n log n) via merge."""
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    # rank-based shortcut
    combined = np.concatenate([a, b])
    order = np.argsort(combined)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(combined))
    ra = ranks[:na]
    # number of b elements less than each a (rank in combined minus rank in a)
    ra_in_a = np.argsort(np.argsort(a))
    less = ra - ra_in_a  # for each a, # of b's strictly below or equal-but-earlier
    # Use brute force for clarity, n <= 600 each so 600*600 is fine.
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (na * nb))


# ---------------- dataset loaders -------------------------------------------
def load_tirosh_subset():
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("  loading Tirosh ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)  # genes x cells

    mask_mal = (malignant == 2)
    mask_T = (malignant == 1) & (nonmal_type == 1)

    var = X.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    X = X[hvg]

    rng = np.random.default_rng(0)
    n_per = 600
    idx_mal = rng.choice(np.where(mask_mal)[0], size=min(n_per, mask_mal.sum()), replace=False)
    idx_T = rng.choice(np.where(mask_T)[0], size=min(n_per, mask_T.sum()), replace=False)
    cells_idx = np.concatenate([idx_mal, idx_T])
    labels = np.array(["Malignant"] * len(idx_mal) + ["T cells"] * len(idx_T))

    Xs = X[:, cells_idx].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50).fit_transform(Xs)
    print(f"  Tirosh subset: {Xpca.shape[0]} cells, "
          f"{(labels=='Malignant').sum()} mal / {(labels=='T cells').sum()} T")
    return Xpca, labels


def load_darmanis_subset():
    EXPR = DATA / "GSE84465_GBM.csv"
    META = DATA / "GSE84465_meta.txt"
    print("  loading Darmanis ...")
    text = META.read_text(errors="replace")
    fields = {}
    for line in text.splitlines():
        if not line.startswith("!Sample_characteristics_ch1"):
            continue
        parts = re.findall(r'"([^"]*)"', line)
        if not parts:
            continue
        prefix = parts[0].split(":", 1)[0].strip()
        values = [p.split(":", 1)[1].strip() if ":" in p else "" for p in parts]
        fields[prefix] = values
    meta = pd.DataFrame(fields)
    meta["cell_id"] = meta["plate id"] + "." + meta["well"]

    df = pd.read_csv(EXPR, sep=r"\s+", header=0, index_col=0, engine="c")
    df.columns = [c.strip('"') for c in df.columns]
    meta_idx = meta.set_index("cell_id").loc[df.columns]
    cell_type = meta_idx["cell type"].values

    Xlog = np.log1p(df.values.astype(np.float32))
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xlog = Xlog[hvg]

    mask_neo = (cell_type == "Neoplastic")
    mask_imm = (cell_type == "Immune cell")
    rng = np.random.default_rng(0)
    n_per = 600
    idx_neo = rng.choice(np.where(mask_neo)[0], size=min(n_per, mask_neo.sum()), replace=False)
    idx_imm = rng.choice(np.where(mask_imm)[0], size=min(n_per, mask_imm.sum()), replace=False)
    cells_idx = np.concatenate([idx_neo, idx_imm])
    labels = np.array(["Neoplastic"] * len(idx_neo) + ["Immune"] * len(idx_imm))

    Xs = Xlog[:, cells_idx].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50).fit_transform(Xs)
    print(f"  Darmanis subset: {Xpca.shape[0]} cells, "
          f"{(labels=='Neoplastic').sum()} neo / {(labels=='Immune').sum()} immune")
    return Xpca, labels


# ---------------- analysis driver -------------------------------------------
def analyze(name, X, labels, k=15, or_n_edges=4000, label_pos=None):
    """
    Compute FR on every edge + OR on a sample, then per-cell means and stats.
    label_pos: which label is the 'positive' / cancer group (kept first in
    output ordering & used for direction sign).
    """
    print(f"\n[{name}]  building kNN (k={k}) graph ...")
    G = build_knn_graph(X, k=k)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"  computing Forman-Ricci on all {G.number_of_edges()} edges ...")
    fr_edges = forman_ricci_edges(G)
    fr_per_cell = per_cell_mean_curvature(G, fr_edges)

    print(f"  computing Ollivier-Ricci on {or_n_edges} sampled edges ...")
    or_edges = ollivier_ricci_edges(G, alpha=0.5, n_edges=or_n_edges, rng=RNG)
    or_per_cell = per_cell_mean_curvature(G, or_edges)

    uniq = list(np.unique(labels))
    if label_pos is not None and label_pos in uniq:
        label_a = label_pos
        label_b = [l for l in uniq if l != label_pos][0]
    else:
        label_a, label_b = uniq[0], uniq[1]

    cells_a = [i for i, l in enumerate(labels) if l == label_a]
    cells_b = [i for i, l in enumerate(labels) if l == label_b]

    fr_a = np.array([fr_per_cell[c] for c in cells_a if not np.isnan(fr_per_cell[c])])
    fr_b = np.array([fr_per_cell[c] for c in cells_b if not np.isnan(fr_per_cell[c])])
    or_a = np.array([or_per_cell[c] for c in cells_a if not np.isnan(or_per_cell[c])])
    or_b = np.array([or_per_cell[c] for c in cells_b if not np.isnan(or_per_cell[c])])

    t_stat, p_t = ttest_ind(fr_a, fr_b, equal_var=False)
    u_stat, p_u = mannwhitneyu(fr_a, fr_b)
    delta_fr = cliffs_delta(fr_a, fr_b)
    delta_or = cliffs_delta(or_a, or_b)

    print(f"  FR  {label_a:11s}  per-cell mean = {fr_a.mean():+.4f}  "
          f"+- {fr_a.std():.4f}  (n={len(fr_a)})")
    print(f"  FR  {label_b:11s}  per-cell mean = {fr_b.mean():+.4f}  "
          f"+- {fr_b.std():.4f}  (n={len(fr_b)})")
    print(f"  FR  Welch t={t_stat:.3f}  p={p_t:.3g}    "
          f"MWU U={u_stat:.1f}  p={p_u:.3g}    Cliff's delta={delta_fr:+.3f}")
    print(f"  OR  {label_a:11s} mean = {or_a.mean():+.4f}    "
          f"{label_b:11s} mean = {or_b.mean():+.4f}    "
          f"Cliff's delta={delta_or:+.3f}")

    return dict(
        name=name, label_a=label_a, label_b=label_b,
        fr_a=fr_a, fr_b=fr_b, or_a=or_a, or_b=or_b,
        fr_per_cell=fr_per_cell, or_per_cell=or_per_cell,
        labels=labels, p_t=float(p_t), p_u=float(p_u),
        delta_fr=delta_fr, delta_or=delta_or,
        cells_a=cells_a, cells_b=cells_b,
    )


def main():
    print("=" * 70)
    print("E2  Forman-Ricci curvature  vs  Ollivier-Ricci on scRNA-seq")
    print("=" * 70)

    results = []
    Xt, lt = load_tirosh_subset()
    results.append(analyze("Tirosh melanoma", Xt, lt, label_pos="Malignant"))

    Xd, ld = load_darmanis_subset()
    results.append(analyze("Darmanis GBM", Xd, ld, label_pos="Neoplastic"))

    # ---------- CSV summary ----------
    rows = []
    or_published = {
        "Tirosh melanoma": {"Malignant": 0.120, "T cells": -0.054, "delta": 0.17},
        "Darmanis GBM":    {"Neoplastic": 0.081, "Immune": 0.010, "delta": 0.07},
    }
    for r in results:
        for label, fr_vals, or_vals in [
            (r["label_a"], r["fr_a"], r["or_a"]),
            (r["label_b"], r["fr_b"], r["or_b"]),
        ]:
            rows.append({
                "dataset": r["name"],
                "group": label,
                "fr_mean": float(fr_vals.mean()),
                "fr_std": float(fr_vals.std()),
                "or_mean": float(or_vals.mean()),
                "or_std": float(or_vals.std()),
                "n_cells": int(len(fr_vals)),
                "fr_p_welch": r["p_t"],
                "fr_p_mwu": r["p_u"],
                "fr_cliffs_delta": r["delta_fr"],
                "or_cliffs_delta": r["delta_or"],
                "or_published_mean": or_published[r["name"]].get(label, np.nan),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "e2_forman_ricci.csv", index=False)
    print(f"\nWrote {OUT/'e2_forman_ricci.csv'}")

    # ---------- summary histograms ----------
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[0, i]
        ax.hist(r["fr_a"], bins=40, alpha=0.55, label=r["label_a"], density=True)
        ax.hist(r["fr_b"], bins=40, alpha=0.55, label=r["label_b"], density=True)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("per-cell mean Forman-Ricci")
        ax.set_ylabel("density")
        ax.set_title(f"{r['name']}\n"
                     f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}  "
                     f"d={r['delta_fr']:+.2f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "e2_summary.png", dpi=120)
    plt.close()
    print(f"Wrote {OUT/'e2_summary.png'}")

    # ---------- OR-vs-FR scatter (Tirosh) ----------
    r0 = results[0]
    fr_vals = []
    or_vals = []
    color_lbl = []
    for c in range(len(r0["labels"])):
        fr = r0["fr_per_cell"].get(c, np.nan)
        orr = r0["or_per_cell"].get(c, np.nan)
        if np.isnan(fr) or np.isnan(orr):
            continue
        fr_vals.append(fr)
        or_vals.append(orr)
        color_lbl.append(r0["labels"][c])
    fr_vals = np.array(fr_vals)
    or_vals = np.array(or_vals)
    color_lbl = np.array(color_lbl)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lab, color in zip([r0["label_a"], r0["label_b"]], ["tab:red", "tab:blue"]):
        m = (color_lbl == lab)
        ax.scatter(or_vals[m], fr_vals[m], s=10, alpha=0.55, c=color, label=lab)
    # Pearson + Spearman
    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(or_vals, fr_vals)
    sr, sp = spearmanr(or_vals, fr_vals)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("per-cell Ollivier-Ricci kappa")
    ax.set_ylabel("per-cell Forman-Ricci")
    ax.set_title(f"{r0['name']}: OR vs FR  "
                 f"Pearson r={pr:.2f}  Spearman r={sr:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "e2_or_vs_fr_correlation.png", dpi=120)
    plt.close()
    print(f"Wrote {OUT/'e2_or_vs_fr_correlation.png'}")

    # ---------- text report ----------
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("E2  Forman-Ricci summary")
    report_lines.append("=" * 70)
    for r in results:
        fr_diff = r["fr_a"].mean() - r["fr_b"].mean()
        or_diff_pub = or_published[r["name"]]["delta"]
        ratio = abs(fr_diff) / or_diff_pub if or_diff_pub else float("nan")
        sign_match = (fr_diff > 0) == (or_diff_pub > 0)
        report_lines.append(
            f"{r['name']}: FR {r['label_a']}={r['fr_a'].mean():+.4f}  "
            f"{r['label_b']}={r['fr_b'].mean():+.4f}  "
            f"FR-delta={fr_diff:+.4f}  OR-delta-pub={or_diff_pub:+.3f}  "
            f"sign-match={sign_match}  |FR/OR|={ratio:.2f}  "
            f"FR-Cliff={r['delta_fr']:+.2f}  OR-Cliff={r['delta_or']:+.2f}  "
            f"Welch-p={r['p_t']:.2g}  MWU-p={r['p_u']:.2g}"
        )
    txt = "\n".join(report_lines)
    print("\n" + txt)
    (OUT / "e2_report.txt").write_text(txt + "\n")
    print(f"Wrote {OUT/'e2_report.txt'}")


if __name__ == "__main__":
    main()
