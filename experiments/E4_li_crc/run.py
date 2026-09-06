"""
E4 — Ollivier-Ricci curvature on a fourth tumor type: colorectal cancer
(Li et al. 2017, GSE81861, "tumor all cells" FPKM file).

Replicates the per-cell OR pipeline of path1_hyperbolic/run_ricci.py with:
  - log1p, HVG-2000, PCA-50, kNN k=15
  - alpha=0.5 lazy walk; sampled edges = min(2500, |E|) given small N
  - per-cell mean kappa across cells from the bootstrap (5 seeds)
  - two contrasts: Epithelial (malignant) vs all others; Epithelial vs T cells

Cell-type labels are encoded directly in the column headers as
   <barcode>__<cellType>__<colorHex>
"""

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
SEED = 20260507


# ---------------- Ollivier-Ricci on a kNN graph ----------------
def build_knn_graph(X, k=15):
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


def ollivier_ricci_edges(G, alpha=0.5, n_edges=None, rng=None):
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
        w_u = (np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu)) if nu
               else np.array([1.0]))
        w_v = (np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv)) if nv
               else np.array([1.0]))
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
        kappa = 1.0 - W / d_uv if d_uv > 0 else np.nan
        out[(u, v)] = float(kappa)
    return out


def per_cell_mean_curvature(G, edge_kappa):
    by_cell = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k):
            continue
        by_cell[u].append(k)
        by_cell[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by_cell.items()}


# ---------------- Cliff's delta ----------------
def cliffs_delta(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return np.nan
    # vectorised count of (a_i > b_j) - (a_i < b_j)
    gt = 0
    lt = 0
    # block-wise to be safe on memory
    bsz = 500
    for i in range(0, n_a, bsz):
        chunk = a[i:i + bsz][:, None]
        gt += np.sum(chunk > b[None, :])
        lt += np.sum(chunk < b[None, :])
    return float(gt - lt) / (n_a * n_b)


# ---------------- dataset loader ----------------
def load_li_crc():
    """Return (X_log_hvg [genes x cells], cell_types) for Li 2017 CRC tumor."""
    csv_path = DATA / "GSE81861_CRC_tumor_FPKM.csv"
    print(f"  loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    cols = df.columns.tolist()
    # parse <barcode>__<celltype>__<color>
    cell_types = []
    for c in cols:
        parts = c.split("__")
        cell_types.append(parts[1] if len(parts) >= 2 else "NA")
    cell_types = np.array(cell_types)
    # drop NA-typed cells
    keep = cell_types != "NA"
    df = df.loc[:, keep]
    cell_types = cell_types[keep]

    X = df.values.astype(np.float32)  # genes x cells
    # FPKM -> log1p before HVG selection
    Xlog = np.log1p(X)
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xlog = Xlog[hvg]
    print(f"  matrix: {Xlog.shape[0]} HVGs x {Xlog.shape[1]} cells")
    cnts = pd.Series(cell_types).value_counts()
    print("  cell-type counts:")
    for k, v in cnts.items():
        print(f"    {k:14s} {v}")
    return Xlog, cell_types


def make_pca(X_genes_x_cells):
    Xs = X_genes_x_cells.T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    n_comp = min(50, Xs.shape[0] - 1, Xs.shape[1])
    return PCA(n_components=n_comp).fit_transform(Xs)


# ---------------- contrast helper ----------------
def run_contrast(name, Xpca_full, cell_types_full, label_a, mask_b_fn,
                 b_label, n_per_cap=250, seeds=(0, 1, 2, 3, 4),
                 k=15, n_edges_target=2500):
    """
    Group a = cells whose cell_type == label_a.
    Group b = cells satisfying mask_b_fn(cell_types_full).
    Bootstrap over `seeds`: each seed subsamples per group up to
    min(n_a, n_b, n_per_cap), runs the full pipeline, collects per-cell kappa.
    Average per-cell kappa across seeds for cells that appear at least once.
    """
    print(f"\n--- contrast: {name} ---")
    mask_a = cell_types_full == label_a
    mask_b = mask_b_fn(cell_types_full)
    n_a_total = int(mask_a.sum())
    n_b_total = int(mask_b.sum())
    print(f"  total: {label_a} n={n_a_total}   {b_label} n={n_b_total}")
    n_per = min(n_a_total, n_b_total, n_per_cap)
    print(f"  per-group cap per bootstrap = {n_per}")

    idx_a_pool = np.where(mask_a)[0]
    idx_b_pool = np.where(mask_b)[0]

    # For each cell in the union pool we'll collect per-cell kappa across seeds.
    accum = {int(i): [] for i in np.concatenate([idx_a_pool, idx_b_pool])}

    for s in seeds:
        rng = np.random.default_rng(SEED + s)
        sub_a = rng.choice(idx_a_pool, size=n_per, replace=False)
        sub_b = rng.choice(idx_b_pool, size=n_per, replace=False)
        sel = np.concatenate([sub_a, sub_b])
        Xs = Xpca_full[sel]
        G = build_knn_graph(Xs, k=k)
        n_edges = min(n_edges_target, G.number_of_edges())
        edge_kappa = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=rng)
        per_cell = per_cell_mean_curvature(G, edge_kappa)
        # map local idx -> global idx
        for local_i, global_i in enumerate(sel):
            v = per_cell.get(local_i, np.nan)
            if not np.isnan(v):
                accum[int(global_i)].append(v)
        n_valid = sum(1 for x in edge_kappa.values() if not np.isnan(x))
        print(f"  seed {s}: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}"
              f"  edges sampled={n_edges}  valid={n_valid}")

    avg = {gi: float(np.mean(vs)) for gi, vs in accum.items() if len(vs) > 0}
    kappa_a = np.array([avg[i] for i in idx_a_pool if i in avg])
    kappa_b = np.array([avg[i] for i in idx_b_pool if i in avg])

    if len(kappa_a) < 2 or len(kappa_b) < 2:
        return dict(name=name, label_a=label_a, label_b=b_label,
                    kappa_a=kappa_a, kappa_b=kappa_b,
                    p_t=np.nan, p_u=np.nan, delta=np.nan)

    t_stat, p_t = ttest_ind(kappa_a, kappa_b, equal_var=False)
    u_stat, p_u = mannwhitneyu(kappa_a, kappa_b)
    delta = cliffs_delta(kappa_a, kappa_b)
    print(f"  {label_a:18s} mean kappa = {kappa_a.mean():.4f}"
          f" +- {kappa_a.std():.4f}   n={len(kappa_a)}")
    print(f"  {b_label:18s} mean kappa = {kappa_b.mean():.4f}"
          f" +- {kappa_b.std():.4f}   n={len(kappa_b)}")
    print(f"  Welch t={t_stat:.3f} p={p_t:.4g}   "
          f"MWU U={u_stat:.1f} p={p_u:.4g}   Cliff's delta={delta:.3f}")
    return dict(name=name, label_a=label_a, label_b=b_label,
                kappa_a=kappa_a, kappa_b=kappa_b,
                p_t=float(p_t), p_u=float(p_u), delta=float(delta))


def main():
    print("=" * 60)
    print("E4 — Li 2017 CRC (GSE81861) Ollivier-Ricci replication")
    print("=" * 60)

    Xlog_hvg, cell_types = load_li_crc()
    Xpca = make_pca(Xlog_hvg)
    print(f"  PCA: {Xpca.shape}")

    results = []
    # contrast 1: Epithelial vs all other (non-NA, non-Epithelial)
    res1 = run_contrast(
        name="Epithelial vs Other",
        Xpca_full=Xpca,
        cell_types_full=cell_types,
        label_a="Epithelial",
        mask_b_fn=lambda ct: (ct != "Epithelial") & (ct != "NA"),
        b_label="Other",
    )
    results.append(res1)

    # contrast 2: Epithelial vs T cells (largest single non-malignant type)
    res2 = run_contrast(
        name="Epithelial vs Tcell",
        Xpca_full=Xpca,
        cell_types_full=cell_types,
        label_a="Epithelial",
        mask_b_fn=lambda ct: ct == "Tcell",
        b_label="Tcell",
    )
    results.append(res2)

    # ---- summary CSV ----
    rows = []
    for r in results:
        for label, vals in [(r["label_a"], r["kappa_a"]),
                            (r["label_b"], r["kappa_b"])]:
            rows.append({"contrast": r["name"], "group": label,
                         "kappa_mean": float(vals.mean()) if len(vals) else np.nan,
                         "kappa_std": float(vals.std()) if len(vals) else np.nan,
                         "n_cells": int(len(vals)),
                         "p_welch": r["p_t"],
                         "p_mwu": r["p_u"],
                         "cliffs_delta": r["delta"]})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT / "e4_li_summary.csv", index=False)
    print(f"\nSaved {OUT / 'e4_li_summary.csv'}")

    # ---- distribution plot ----
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[0, i]
        if len(r["kappa_a"]):
            ax.hist(r["kappa_a"], bins=30, alpha=0.55,
                    label=f'{r["label_a"]} (n={len(r["kappa_a"])})',
                    density=True, color="#cc3344")
        if len(r["kappa_b"]):
            ax.hist(r["kappa_b"], bins=30, alpha=0.55,
                    label=f'{r["label_b"]} (n={len(r["kappa_b"])})',
                    density=True, color="#3366cc")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
        ax.set_ylabel("density")
        ax.set_title(f"{r['name']}\n"
                     f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}  "
                     f"delta={r['delta']:.3f}")
        ax.legend()
    plt.tight_layout()
    fig_path = OUT / "e4_li_distributions.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
