"""
E3 — Per-cell Ollivier-Ricci curvature on Puram et al. 2017 HNSCC scRNA-seq
(GSE103322). Third tumor type to test whether the malignant > non-malignant
per-cell kappa direction replicates beyond Tirosh melanoma and Darmanis GBM.

Header structure of GSE103322_HNSCC_all_data.txt (TSV, 5 metadata rows):
  row 1: <empty>          | cell IDs ...
  row 2: processed by Maxima enzyme  | {0,1}
  row 3: Lymph node                  | {0,1}
  row 4: classified  as cancer cell  | {0,1}
  row 5: classified as non-cancer cells | {0,1}
  row 6: non-cancer cell type        | {0, Fibroblast, T cell, B cell,
                                        Macrophage, Endothelial, Mast,
                                        myocyte, Dendritic, -Fibroblast}
  rows 7+: gene symbol  | log2(TPM/10 + 1) values

Pipeline (matches path1_hyperbolic/run_ricci.py exactly for comparability):
  HVG-2000 by variance, PCA-50, kNN(k=15), sampled Ollivier-Ricci on 4000
  edges (alpha=0.5, lazy walk), per-cell mean kappa.

Two contrasts:
  (a) malignant vs all non-malignant
  (b) malignant vs largest specific non-malignant subtype (Fibroblast)
With 600 + 600 cells per contrast, seed=20260507.
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
DATA = OUT.parent.parent / "data" / "GSE103322_HNSCC.txt"
SEED = 20260507
RNG = np.random.default_rng(SEED)


# ---------------- Ollivier-Ricci on a kNN graph ----------------
def build_knn_graph(X, k=15):
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, inds = nbr.kneighbors(X)
    G = nx.Graph()
    n = X.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = int(inds[i, j_idx])
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


def per_cell_mean_curvature(G, edge_kappa):
    by_cell = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k):
            continue
        by_cell[u].append(k)
        by_cell[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by_cell.items()}


def cliffs_delta(a, b):
    """Cliff's delta non-parametric effect size in [-1, 1]."""
    a = np.asarray(a)
    b = np.asarray(b)
    # rank-based formula: delta = (2 * U / (na*nb)) - 1, U is Mann-Whitney
    na, nb = len(a), len(b)
    # explicit pairwise to avoid scipy version variance
    gt = 0
    lt = 0
    # Vectorized comparison (broadcast)
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return (gt - lt) / (na * nb)


# ---------------- Puram loader ----------------
def load_puram():
    print(f"Loading {DATA.name} ...")
    # Header is 6 metadata rows: row 0 cell IDs, row 1 maxima, row 2 lymph node,
    # row 3 cancer flag, row 4 non-cancer flag, row 5 non-cancer subtype string.
    header = pd.read_csv(DATA, sep="\t", nrows=6, header=None, low_memory=False)
    cell_ids = header.iloc[0, 1:].values
    cancer = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    noncancer = pd.to_numeric(header.iloc[4, 1:], errors="coerce").values
    nonmal_type = header.iloc[5, 1:].astype(str).values
    # Normalize the inconsistent "-Fibroblast" tag (small group, treat as fibroblast)
    nonmal_type = np.array([t.lstrip("-").strip() for t in nonmal_type])

    print(f"  total cells: {len(cell_ids)}")
    print(f"  cancer=1: {int(np.sum(cancer == 1))}")
    print(f"  non-cancer=1: {int(np.sum(noncancer == 1))}")
    counts = pd.Series(nonmal_type[noncancer == 1]).value_counts()
    print("  non-cancer subtype counts:")
    for k, v in counts.items():
        print(f"    {k:15s} {v}")

    print("  reading expression matrix ...")
    expr = pd.read_csv(DATA, sep="\t", skiprows=6, header=None, low_memory=False)
    gene_names = expr.iloc[:, 0].values
    X = expr.iloc[:, 1:].values.astype(np.float32)  # genes x cells
    print(f"  matrix: {X.shape} (genes x cells)")
    return cell_ids, cancer, noncancer, nonmal_type, gene_names, X


# ---------------- analysis driver ----------------
def analyze(name, X_pca, labels, label_a, label_b, k=15, n_edges=4000, rng=None):
    print(f"\n[{name}]  building kNN (k={k}) graph ...")
    G = build_knn_graph(X_pca, k=k)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Ollivier-Ricci on {n_edges} sampled edges ...")
    kappa = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=rng)
    n_valid = sum(1 for v in kappa.values() if not np.isnan(v))
    print(f"  valid edges: {n_valid}/{len(kappa)}")
    per_cell = per_cell_mean_curvature(G, kappa)

    cells_a = [i for i, l in enumerate(labels) if l == label_a]
    cells_b = [i for i, l in enumerate(labels) if l == label_b]

    kappa_a = np.array([per_cell[c] for c in cells_a if not np.isnan(per_cell[c])])
    kappa_b = np.array([per_cell[c] for c in cells_b if not np.isnan(per_cell[c])])

    t_stat, p_t = ttest_ind(kappa_a, kappa_b, equal_var=False)
    u_stat, p_u = mannwhitneyu(kappa_a, kappa_b)
    d = cliffs_delta(kappa_a, kappa_b)

    print(f"  {label_a:25s}  kappa = {kappa_a.mean():.4f} +- {kappa_a.std():.4f}  (n={len(kappa_a)})")
    print(f"  {label_b:25s}  kappa = {kappa_b.mean():.4f} +- {kappa_b.std():.4f}  (n={len(kappa_b)})")
    print(f"  Welch t={t_stat:.3f} p={p_t:.4g}  MWU U={u_stat:.1f} p={p_u:.4g}  Cliff d={d:.3f}")

    return dict(name=name, label_a=label_a, label_b=label_b,
                kappa_a=kappa_a, kappa_b=kappa_b,
                mean_a=float(kappa_a.mean()), mean_b=float(kappa_b.mean()),
                std_a=float(kappa_a.std()), std_b=float(kappa_b.std()),
                n_a=int(len(kappa_a)), n_b=int(len(kappa_b)),
                p_t=float(p_t), p_u=float(p_u), cliffs_d=float(d))


def build_subset(X_genes_x_cells, mask_a, mask_b, label_a, label_b,
                 n_per=600, rng=None):
    """Sample 600+600 cells, run HVG (over both groups), PCA-50."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    idx_a = rng.choice(np.where(mask_a)[0],
                       size=min(n_per, int(mask_a.sum())), replace=False)
    idx_b = rng.choice(np.where(mask_b)[0],
                       size=min(n_per, int(mask_b.sum())), replace=False)
    cells_idx = np.concatenate([idx_a, idx_b])
    labels = np.array([label_a] * len(idx_a) + [label_b] * len(idx_b))

    Xs = X_genes_x_cells[:, cells_idx]  # genes x cells
    var = Xs.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xs = Xs[hvg]

    Xc = Xs.T.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50).fit_transform(Xc)
    print(f"  subset: {Xpca.shape[0]} cells "
          f"({(labels == label_a).sum()} {label_a} / "
          f"{(labels == label_b).sum()} {label_b})")
    return Xpca, labels


def main():
    print("=" * 60)
    print("E3 Ollivier-Ricci on Puram 2017 HNSCC (GSE103322)")
    print("=" * 60)
    cell_ids, cancer, noncancer, nonmal_type, gene_names, X = load_puram()

    mask_mal = (cancer == 1)
    mask_non = (noncancer == 1)
    counts = pd.Series(nonmal_type[mask_non]).value_counts()
    largest_sub = counts.index[0]
    print(f"\n  largest non-malignant subtype: {largest_sub} (n={counts.iloc[0]})")

    results = []

    # Contrast (a): malignant vs all non-malignant
    print("\n--- Contrast A: Malignant vs all non-malignant ---")
    Xa, la = build_subset(X, mask_mal, mask_non,
                          "Malignant", "Non-malignant",
                          n_per=600, rng=np.random.default_rng(SEED))
    results.append(analyze("Puram HNSCC: Malignant vs Non-malignant",
                           Xa, la, "Malignant", "Non-malignant",
                           rng=np.random.default_rng(SEED)))

    # Contrast (b): malignant vs largest subtype
    print(f"\n--- Contrast B: Malignant vs {largest_sub} ---")
    mask_sub = mask_non & (nonmal_type == largest_sub)
    Xb, lb = build_subset(X, mask_mal, mask_sub,
                          "Malignant", largest_sub,
                          n_per=600, rng=np.random.default_rng(SEED + 1))
    results.append(analyze(f"Puram HNSCC: Malignant vs {largest_sub}",
                           Xb, lb, "Malignant", largest_sub,
                           rng=np.random.default_rng(SEED + 1)))

    # ---------------- save outputs ----------------
    rows = []
    for r in results:
        for label, vals, mean, std, n in [
            (r["label_a"], r["kappa_a"], r["mean_a"], r["std_a"], r["n_a"]),
            (r["label_b"], r["kappa_b"], r["mean_b"], r["std_b"], r["n_b"]),
        ]:
            rows.append({
                "contrast": f"{r['label_a']} vs {r['label_b']}",
                "group": label,
                "kappa_mean": mean,
                "kappa_std": std,
                "n_cells": n,
                "p_welch": r["p_t"],
                "p_mwu": r["p_u"],
                "cliffs_delta": r["cliffs_d"],
            })
    summary = pd.DataFrame(rows)
    summary_path = OUT / "e3_puram_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # plot
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[0, i]
        ax.hist(r["kappa_a"], bins=40, alpha=0.55, label=r["label_a"],
                density=True)
        ax.hist(r["kappa_b"], bins=40, alpha=0.55, label=r["label_b"],
                density=True)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
        ax.set_ylabel("density")
        delta = r["mean_a"] - r["mean_b"]
        ax.set_title(f"{r['name']}\n"
                     f"delta_mean={delta:+.4f}  Cliff d={r['cliffs_d']:.3f}\n"
                     f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}")
        ax.legend()
    plt.suptitle("Per-cell Ollivier-Ricci curvature, Puram 2017 HNSCC (GSE103322)")
    plt.tight_layout()
    fig_path = OUT / "e3_puram_distributions.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved {fig_path}")

    # one-line direction summary
    print("\nDirection summary (kappa malignant - kappa control):")
    for r in results:
        delta = r["mean_a"] - r["mean_b"]
        sign = "MAL > NON" if delta > 0 else "MAL < NON"
        print(f"  {r['name']:55s}  delta={delta:+.4f}  ({sign})")


if __name__ == "__main__":
    main()
