"""
E5 — Hyperparameter robustness sweep for per-cell Ollivier-Ricci curvature
on Tirosh 2016 melanoma malignant vs T cells.

Grid:
  HVG   in {500, 2000, 5000}
  PCA   in {20, 50, 100}
  kNN k in {5, 15, 30, 50}
  alpha in {0.0, 0.5}
  --> 72 cells total.

For each cell: compute per-cell mean OR on min(3000,|E|) sampled edges,
then Cliff's delta and Mann-Whitney U for malignant vs T-cell distributions.

Caching:
  HVG+PCA matrices cached per (HVG, PCA)  -- 9 combos
  kNN graphs        cached per (HVG, PCA, k) -- 36 combos
  alpha varies on top of the cached graph (32x faster than naive).
"""

import sys
import time
import itertools
import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data"

SEED = 20260507
N_PER_GROUP = 600
# With precomputed APSP (one Dijkstra-from-all per graph), per-edge OR is
# O(k^2) for both the cost matrix lookup and the EMD solve. We can afford
# the user-specified min(3000, |E|) sample at all k.
EDGE_CAP_BY_K = {5: 3000, 15: 3000, 30: 3000, 50: 3000}

HVG_GRID = [500, 2000, 5000]
PCA_GRID = [20, 50, 100]
K_GRID = [5, 15, 30, 50]
ALPHA_GRID = [0.0, 0.5]


# ---------------- data loading ----------------
def load_tirosh_raw():
    """Load Tirosh expression and group masks. Returns (X_genes_x_cells, mask_mal, mask_T)."""
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("loading Tirosh ...", flush=True)
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)
    mask_mal = (malignant == 2)
    mask_T = (malignant == 1) & (nonmal_type == 1)
    print(f"  full expr: {X.shape}, malignant={mask_mal.sum()}, T={mask_T.sum()}", flush=True)
    return X, mask_mal, mask_T


def subsample_cells(X, mask_mal, mask_T, n_per=N_PER_GROUP, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_mal_all = np.where(mask_mal)[0]
    idx_T_all = np.where(mask_T)[0]
    idx_mal = rng.choice(idx_mal_all, size=min(n_per, len(idx_mal_all)), replace=False)
    idx_T = rng.choice(idx_T_all, size=min(n_per, len(idx_T_all)), replace=False)
    cells_idx = np.concatenate([idx_mal, idx_T])
    labels = np.array(["Malignant"] * len(idx_mal) + ["T cells"] * len(idx_T))
    return X[:, cells_idx], labels


# ---------------- HVG + PCA ----------------
def hvg_pca(X_sub, n_hvg, n_pca):
    """X_sub: genes x cells. Returns Xpca (cells x n_pca)."""
    var = X_sub.var(axis=1)
    hvg = np.argsort(var)[::-1][:n_hvg]
    Xs = X_sub[hvg].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    npc = min(n_pca, Xs.shape[0] - 1, Xs.shape[1])
    return PCA(n_components=npc, random_state=SEED).fit_transform(Xs)


# ---------------- kNN graph ----------------
def build_knn_graph(X, k):
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


# ---------------- Ollivier-Ricci ----------------
def or_kappa_for_edge(u, v, neigh, dist_lookup, alpha, edge_w):
    """Compute one Ollivier-Ricci kappa using a precomputed dist_lookup(a,b).
    `neigh[node]` is a list of that node's neighbors in the graph.
    `dist_lookup(a, b)` returns shortest-path distance.
    """
    nu = neigh[u]
    nv = neigh[v]
    sup_u = [u] + nu
    sup_v = [v] + nv
    if alpha == 0.0:
        w_u = np.array([0.0] + [1.0 / len(nu)] * len(nu)) if nu else np.array([1.0])
        w_v = np.array([0.0] + [1.0 / len(nv)] * len(nv)) if nv else np.array([1.0])
    else:
        w_u = np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu)) if nu else np.array([1.0])
        w_v = np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv)) if nv else np.array([1.0])
    nu_len = len(sup_u); nv_len = len(sup_v)
    C = np.empty((nu_len, nv_len))
    for i, a in enumerate(sup_u):
        ra = dist_lookup[a]
        for j, b in enumerate(sup_v):
            d_ab = ra.get(b, np.inf)
            if not np.isfinite(d_ab):
                return np.nan
            C[i, j] = d_ab
    W = ot.emd2(w_u, w_v, C)
    return float(1.0 - W / edge_w) if edge_w > 0 else np.nan


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
    """Cliff's delta in [-1, 1].
    delta > 0  => a tends to be greater than b
    delta < 0  => a tends to be smaller than b
    Computed via Mann-Whitney U: delta = 2U/(n*m) - 1.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    # use scipy MWU (greater alternative) to avoid manual O(nm)
    u, _ = mannwhitneyu(a, b, alternative="greater")
    return 2.0 * u / (n * m) - 1.0


# ---------------- main sweep ----------------
def main():
    t0 = time.time()
    X_full, mask_mal, mask_T = load_tirosh_raw()
    X_sub, labels = subsample_cells(X_full, mask_mal, mask_T, n_per=N_PER_GROUP, seed=SEED)
    print(f"subsampled: {X_sub.shape[1]} cells "
          f"({(labels=='Malignant').sum()} mal / {(labels=='T cells').sum()} T)", flush=True)

    is_mal = (labels == "Malignant")
    is_T = (labels == "T cells")

    # Cache PCA matrices  (HVG, PCA) -> Xpca
    pca_cache = {}
    for n_hvg in HVG_GRID:
        for n_pc in PCA_GRID:
            print(f"  PCA cache: HVG={n_hvg} PCA={n_pc}", flush=True)
            pca_cache[(n_hvg, n_pc)] = hvg_pca(X_sub, n_hvg, n_pc)

    rows = []
    n_total = len(HVG_GRID) * len(PCA_GRID) * len(K_GRID) * len(ALPHA_GRID)
    cell_i = 0

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    for n_hvg, n_pc, k in itertools.product(HVG_GRID, PCA_GRID, K_GRID):
        Xpca = pca_cache[(n_hvg, n_pc)]
        t_g = time.time()
        G = build_knn_graph(Xpca, k=k)
        n_e = G.number_of_edges()
        n_nodes = G.number_of_nodes()
        n_sample = min(EDGE_CAP_BY_K[k], n_e)
        # per-graph rng so the sampled edge set is deterministic
        graph_rng = np.random.default_rng(SEED + 1000 * n_hvg + 13 * n_pc + k)
        edges_all = list(G.edges())
        if n_sample < n_e:
            sel = graph_rng.choice(n_e, size=n_sample, replace=False)
            sample_edges = [edges_all[i] for i in sel]
        else:
            sample_edges = edges_all

        # Precompute adjacency lists and edge weights once
        neigh = {n: list(G.neighbors(n)) for n in G.nodes()}
        edge_weight = {(u, v): G[u][v]["weight"] for (u, v) in G.edges()}
        edge_weight.update({(v, u): w for (u, v), w in list(edge_weight.items())})

        # All-pairs shortest path (sparse Dijkstra). For 1200 nodes this is
        # ~0.5-2s and amortizes across the alpha loop and across all sampled
        # edges (vs O(|sample|*k) Dijkstras the naive way).
        src_idx = []; dst_idx = []; w_arr = []
        for (u, v) in G.edges():
            w = G[u][v]["weight"]
            src_idx.append(u); dst_idx.append(v); w_arr.append(w)
            src_idx.append(v); dst_idx.append(u); w_arr.append(w)
        A = csr_matrix((w_arr, (src_idx, dst_idx)), shape=(n_nodes, n_nodes))
        # Restrict APSP to the source nodes that actually appear as supports
        # in our sampled edges (their u/v + neighbors). For dense kNN this is
        # often most of the graph anyway, so just compute full APSP.
        D = dijkstra(A, directed=False)  # n x n dense
        graph_secs = time.time() - t_g

        for alpha in ALPHA_GRID:
            cell_i += 1
            t1 = time.time()
            kappa = {}
            for (u, v) in sample_edges:
                nu = neigh[u]
                nv = neigh[v]
                sup_u = [u] + nu
                sup_v = [v] + nv
                if alpha == 0.0:
                    if not nu or not nv:
                        kappa[(u, v)] = np.nan
                        continue
                    w_u = np.array([0.0] + [1.0 / len(nu)] * len(nu))
                    w_v = np.array([0.0] + [1.0 / len(nv)] * len(nv))
                else:
                    w_u = np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu)) if nu else np.array([1.0])
                    w_v = np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv)) if nv else np.array([1.0])
                # Cost matrix: D[sup_u][:, sup_v]
                C = D[np.ix_(sup_u, sup_v)]
                if not np.all(np.isfinite(C)):
                    kappa[(u, v)] = np.nan
                    continue
                W = ot.emd2(w_u, w_v, C)
                d_uv = edge_weight[(u, v)]
                kappa[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan

            per_cell = per_cell_mean_curvature(G, kappa)
            kappa_mal = np.array([per_cell[i] for i in range(len(labels))
                                  if is_mal[i] and not np.isnan(per_cell[i])])
            kappa_T = np.array([per_cell[i] for i in range(len(labels))
                                if is_T[i] and not np.isnan(per_cell[i])])

            if len(kappa_mal) == 0 or len(kappa_T) == 0:
                cd = np.nan
                p_u = np.nan
            else:
                cd = cliffs_delta(kappa_mal, kappa_T)
                _, p_u = mannwhitneyu(kappa_mal, kappa_T, alternative="two-sided")

            row = dict(
                HVG=n_hvg, PCA=n_pc, k=k, alpha=alpha,
                n_edges_total=n_e, n_edges_sampled=n_sample,
                n_mal_cells=len(kappa_mal), n_T_cells=len(kappa_T),
                mean_kappa_mal=float(kappa_mal.mean()) if len(kappa_mal) else np.nan,
                mean_kappa_T=float(kappa_T.mean()) if len(kappa_T) else np.nan,
                cliff_delta=float(cd) if cd == cd else np.nan,
                mwu_p=float(p_u) if p_u == p_u else np.nan,
            )
            rows.append(row)
            dt = time.time() - t1
            elapsed = time.time() - t0
            extra = f" (apsp {graph_secs:.1f}s)" if alpha == ALPHA_GRID[0] else ""
            print(f"  [{cell_i:2d}/{n_total}] HVG={n_hvg} PCA={n_pc} k={k:2d} a={alpha} "
                  f"|E|={n_e} k_mal={row['mean_kappa_mal']:.3f} k_T={row['mean_kappa_T']:.3f} "
                  f"delta={row['cliff_delta']:+.3f} p={row['mwu_p']:.2g}  "
                  f"({dt:.1f}s{extra}, total {elapsed:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    csv_path = OUT / "e5_hparam_grid.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}", flush=True)

    # ---------------- heatmap ----------------
    # Tile k x HVG with PCA panels, one figure per alpha
    fig, axes = plt.subplots(len(ALPHA_GRID), len(PCA_GRID),
                             figsize=(4.0 * len(PCA_GRID), 3.4 * len(ALPHA_GRID)),
                             squeeze=False)
    vmax = max(0.05, float(np.nanmax(np.abs(df["cliff_delta"].values))))
    for ai, alpha in enumerate(ALPHA_GRID):
        for pi, n_pc in enumerate(PCA_GRID):
            ax = axes[ai, pi]
            sub = df[(df["alpha"] == alpha) & (df["PCA"] == n_pc)]
            mat = sub.pivot(index="k", columns="HVG", values="cliff_delta")
            mat = mat.reindex(index=K_GRID, columns=HVG_GRID)
            im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto", origin="lower")
            ax.set_xticks(range(len(HVG_GRID)))
            ax.set_xticklabels(HVG_GRID)
            ax.set_yticks(range(len(K_GRID)))
            ax.set_yticklabels(K_GRID)
            ax.set_xlabel("HVG")
            ax.set_ylabel("k (kNN)")
            ax.set_title(f"alpha={alpha}, PCA={n_pc}")
            for ii in range(mat.shape[0]):
                for jj in range(mat.shape[1]):
                    v = mat.values[ii, jj]
                    if np.isnan(v):
                        continue
                    ax.text(jj, ii, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8,
                            color="black" if abs(v) < 0.5 * vmax else "white")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("E5 hparam sweep: Cliff's delta (malignant vs T cells)\n"
                 "positive = malignant > T per-cell mean kappa", y=1.0)
    plt.tight_layout()
    png_path = OUT / "e5_hparam_heatmap.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}", flush=True)

    # ---------------- summary stats ----------------
    deltas = df["cliff_delta"].dropna().values
    frac_pos = (deltas > 0).mean()
    median_d = float(np.median(deltas))
    min_d = float(np.min(deltas))
    max_d = float(np.max(deltas))
    print("\n=== summary ===")
    print(f"  N cells: {len(deltas)}/{len(df)}")
    print(f"  fraction with Cliff's delta > 0 (mal > T): {frac_pos:.3f}")
    print(f"  median Cliff's delta: {median_d:+.3f}")
    print(f"  range: [{min_d:+.3f}, {max_d:+.3f}]")

    # Variance attributable to each axis (fraction of total variance explained
    # by the group means around grand mean -- a simple ANOVA-style ratio)
    grand = deltas.mean()
    total_var = float(np.var(deltas))
    expl = {}
    for axis in ["HVG", "PCA", "k", "alpha"]:
        gm = df.groupby(axis)["cliff_delta"].mean()
        # weighted variance of group means with group sizes
        sizes = df.groupby(axis)["cliff_delta"].count().values
        between = float(np.sum(sizes * (gm.values - grand) ** 2) / sizes.sum())
        expl[axis] = between / total_var if total_var > 0 else 0.0
    print("  variance fraction explained by axis:")
    for axis, v in sorted(expl.items(), key=lambda x: -x[1]):
        print(f"    {axis}: {v:.3f}")
    dominant = max(expl, key=expl.get)
    print(f"  dominant axis: {dominant}")

    # robustness verdict
    if frac_pos > 0.80 and abs(median_d) > 0.30:
        verdict = "stable"
    elif frac_pos < 0.50 or abs(median_d) < 0.10:
        verdict = "fragile"
    else:
        verdict = "partly fragile"
    print(f"  ROBUSTNESS VERDICT: {verdict}")

    # also save summary
    with open(OUT / "e5_summary.txt", "w") as f:
        f.write(f"E5 hyperparameter sweep summary\n")
        f.write(f"N cells (valid / total): {len(deltas)} / {len(df)}\n")
        f.write(f"frac(delta>0): {frac_pos:.3f}\n")
        f.write(f"median delta: {median_d:+.3f}\n")
        f.write(f"range: [{min_d:+.3f}, {max_d:+.3f}]\n")
        f.write("variance fraction explained by axis:\n")
        for axis, v in sorted(expl.items(), key=lambda x: -x[1]):
            f.write(f"  {axis}: {v:.3f}\n")
        f.write(f"dominant axis: {dominant}\n")
        f.write(f"robustness verdict: {verdict}\n")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
