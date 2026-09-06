"""
E6 — Per-cell Ollivier-Ricci curvature along differentiation pseudotime.

Tests whether kappa(cell) is a general "stemness vs terminal differentiation"
geometric signal, on a non-cancer hematopoiesis reference: Paul 2015 mouse
myeloid (MEP -> erythroid / megakaryocytic / monocyte / neutrophil / ...).

Pipeline:
  1. scanpy preprocess (HVG=2000, PCA=50)
  2. neighbors -> diffmap -> dpt with iroot in most stem-like cluster
  3. kNN graph on PCA-50, OR curvature (alpha=0.5) on 4000 sampled edges
  4. Spearman rho(kappa_per_cell, dpt)
  5. Per-cluster mean kappa, sorted by mean dpt
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ot
import networkx as nx
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
RNG = np.random.default_rng(20260507)
SEED = 20260507
np.random.seed(SEED)


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
    return {n: float(np.mean(vs)) if vs else np.nan
            for n, vs in by_cell.items()}


# ---------------- main ----------------
def main():
    print("=" * 60)
    print("E6 — Ollivier-Ricci kappa vs DPT on Paul 2015 myeloid")
    print("=" * 60)

    # 1. Load + preprocess
    print("\n[1] Loading Paul 2015 ...")
    adata = sc.datasets.paul15()
    print(f"  initial: {adata.shape}")
    print(f"  clusters: {list(adata.obs['paul15_clusters'].cat.categories)}")

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, random_state=SEED)
    print(f"  after preprocess: {adata.shape}")

    # 2. Neighbors + diffmap + dpt
    print("\n[2] Computing neighbors / diffmap / DPT ...")
    sc.pp.neighbors(adata, n_neighbors=15, random_state=SEED)
    sc.tl.diffmap(adata)

    # Pick most stem-like cluster as iroot.
    # Paul 2015: '7MEP' = megakaryocyte-erythrocyte progenitor (most upstream
    # within this snapshot of myeloid commitment).
    cats = list(adata.obs["paul15_clusters"].cat.categories)
    root_cluster = None
    for cand in ["7MEP", "8Mk"]:
        if cand in cats:
            root_cluster = cand
            break
    if root_cluster is None:
        root_cluster = cats[0]
    print(f"  iroot cluster: {root_cluster}")
    root_idx = (adata.obs["paul15_clusters"] == root_cluster).values.nonzero()[0]
    adata.uns["iroot"] = int(root_idx[0])
    sc.tl.dpt(adata)

    dpt = adata.obs["dpt_pseudotime"].values
    print(f"  DPT range: [{np.nanmin(dpt):.3f}, {np.nanmax(dpt):.3f}], "
          f"n_finite={np.isfinite(dpt).sum()}/{len(dpt)}")

    # 3. kNN graph + OR
    print("\n[3] Building kNN(k=15) on PCA-50 + OR (4000 sampled edges) ...")
    Xpca = adata.obsm["X_pca"]
    G = build_knn_graph(Xpca, k=15)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    edge_kappa = ollivier_ricci_edges(G, alpha=0.5, n_edges=4000, rng=RNG)
    n_valid = sum(1 for v in edge_kappa.values() if not np.isnan(v))
    print(f"  valid edges: {n_valid}/{len(edge_kappa)}")
    per_cell = per_cell_mean_curvature(G, edge_kappa)

    # 4. Assemble per-cell df
    cells = np.arange(adata.n_obs)
    kappa_arr = np.array([per_cell[c] for c in cells])
    cluster_arr = adata.obs["paul15_clusters"].values.astype(str)
    df = pd.DataFrame({
        "cell_id": adata.obs_names.values,
        "cluster": cluster_arr,
        "dpt": dpt,
        "kappa": kappa_arr,
    })
    df.to_csv(OUT / "e6_paul_per_cell.csv", index=False)
    print(f"  wrote per-cell CSV: {len(df)} rows ({df['kappa'].notna().sum()} with kappa)")

    # 5. Spearman correlation
    mask = np.isfinite(df["kappa"].values) & np.isfinite(df["dpt"].values)
    rho, pval = spearmanr(df.loc[mask, "kappa"], df.loc[mask, "dpt"])
    print(f"\n[4] Spearman rho(kappa, DPT) = {rho:.4f}   p = {pval:.3e}   "
          f"(n={mask.sum()})")

    # 6. Per-cluster mean kappa, sorted by mean dpt
    grp = (df.dropna(subset=["kappa", "dpt"])
             .groupby("cluster")
             .agg(mean_kappa=("kappa", "mean"),
                  std_kappa=("kappa", "std"),
                  mean_dpt=("dpt", "mean"),
                  n_cells=("kappa", "size"))
             .sort_values("mean_dpt"))
    grp.to_csv(OUT / "e6_paul_per_cluster.csv")
    print("\n[5] Per-cluster means (sorted by mean DPT):")
    print(grp.to_string())

    # 7. Plot kappa vs DPT, colored by cluster
    fig, ax = plt.subplots(figsize=(9, 6))
    sub = df.dropna(subset=["kappa", "dpt"])
    cats = sorted(sub["cluster"].unique())
    cmap = plt.get_cmap("tab20", len(cats))
    for i, c in enumerate(cats):
        s = sub[sub["cluster"] == c]
        ax.scatter(s["dpt"], s["kappa"], s=8, alpha=0.6,
                   color=cmap(i), label=c)
    # smoothed trend
    order = np.argsort(sub["dpt"].values)
    x = sub["dpt"].values[order]
    y = sub["kappa"].values[order]
    win = max(50, len(y) // 40)
    if len(y) > win:
        kernel = np.ones(win) / win
        y_smooth = np.convolve(y, kernel, mode="same")
        ax.plot(x, y_smooth, color="k", lw=2, label="rolling mean")
    ax.set_xlabel("Diffusion pseudotime (DPT)")
    ax.set_ylabel("Per-cell mean Ollivier-Ricci kappa")
    ax.set_title(f"Paul 2015 myeloid: kappa vs DPT  "
                 f"(Spearman rho={rho:.3f}, p={pval:.2e}, n={mask.sum()})")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig(OUT / "e6_paul_pseudotime.png", dpi=130)
    plt.close()
    print(f"\n  wrote {OUT/'e6_paul_pseudotime.png'}")

    # Save summary line
    summary = {
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n_cells_used": int(mask.sum()),
        "iroot_cluster": root_cluster,
        "n_clusters": int(df["cluster"].nunique()),
        "kappa_mean_overall": float(np.nanmean(kappa_arr)),
        "kappa_std_overall": float(np.nanstd(kappa_arr)),
    }
    pd.Series(summary).to_csv(OUT / "e6_paul_summary.csv")
    print("\n[done] summary:", summary)


if __name__ == "__main__":
    main()
