"""
T6 - Negative-control sanity check for HVG-2000 selection.

Replace HVG-2000 (top-variance gene selection) with 2000 randomly-chosen genes
across 20 random seeds, and re-run the per-cell Ollivier-Ricci kappa analysis
on Tirosh 2016 melanoma malignant (600) vs T cells (600).

Hypotheses:
  - If random gene sets produce comparable kappa separation (Cliff's delta),
    the original HVG-2000 effect was an artifact of HVG selection or trivial
    sparsity patterns.
  - If random sets produce ~null effects while HVG-2000 produces a strong
    effect, the signal is genuinely from biological-variance-driven gene
    selection.

Cell sample is fixed (seed 20260507). Only the gene sample varies (seeds
1..20). Reference run uses real HVG-2000.

Pipeline per gene-set:
  HVG/random gene selection (2000 genes)
  -> PCA-50
  -> kNN graph with k=15
  -> Ollivier-Ricci kappa with alpha=0.5 on 3000 sampled edges
  -> per-cell mean kappa
  -> Cliff's delta and Mann-Whitney U on malignant vs T cell distributions
"""

import time
import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data"

CELL_SEED = 20260507
N_PER_GROUP = 600
N_GENES = 2000
N_PCA = 50
K_NN = 15
ALPHA = 0.5
N_EDGES = 3000
N_RANDOM_SEEDS = 20


# ---------------- data loading ----------------
def load_tirosh_raw():
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


def subsample_cells(X, mask_mal, mask_T, n_per=N_PER_GROUP, seed=CELL_SEED):
    rng = np.random.default_rng(seed)
    idx_mal_all = np.where(mask_mal)[0]
    idx_T_all = np.where(mask_T)[0]
    idx_mal = rng.choice(idx_mal_all, size=min(n_per, len(idx_mal_all)), replace=False)
    idx_T = rng.choice(idx_T_all, size=min(n_per, len(idx_T_all)), replace=False)
    cells_idx = np.concatenate([idx_mal, idx_T])
    labels = np.array(["Malignant"] * len(idx_mal) + ["T cells"] * len(idx_T))
    return X[:, cells_idx], labels


# ---------------- gene selection ----------------
def select_hvg(X_sub, n_genes=N_GENES):
    """Top-variance gene indices."""
    var = X_sub.var(axis=1)
    return np.argsort(var)[::-1][:n_genes]


def select_random_genes(X_sub, n_genes=N_GENES, seed=1):
    """Uniformly sample n_genes from the gene universe."""
    rng = np.random.default_rng(seed)
    n_total = X_sub.shape[0]
    return rng.choice(n_total, size=n_genes, replace=False)


def project_pca(X_sub, gene_idx, n_pca=N_PCA, seed=CELL_SEED):
    Xs = X_sub[gene_idx].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    npc = min(n_pca, Xs.shape[0] - 1, Xs.shape[1])
    return PCA(n_components=npc, random_state=seed).fit_transform(Xs)


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
def per_cell_mean_curvature(G, edge_kappa):
    by_cell = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k):
            continue
        by_cell[u].append(k)
        by_cell[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by_cell.items()}


def cliffs_delta(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    u, _ = mannwhitneyu(a, b, alternative="greater")
    return 2.0 * u / (n * m) - 1.0


def run_pipeline(Xpca, labels, edge_seed):
    """Build kNN, sample edges, compute OR kappa, per-cell mean, Cliff's delta."""
    is_mal = (labels == "Malignant")
    is_T = (labels == "T cells")

    G = build_knn_graph(Xpca, k=K_NN)
    n_e = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_sample = min(N_EDGES, n_e)
    rng_edge = np.random.default_rng(edge_seed)
    edges_all = list(G.edges())
    if n_sample < n_e:
        sel = rng_edge.choice(n_e, size=n_sample, replace=False)
        sample_edges = [edges_all[i] for i in sel]
    else:
        sample_edges = edges_all

    neigh = {n: list(G.neighbors(n)) for n in G.nodes()}
    edge_weight = {(u, v): G[u][v]["weight"] for (u, v) in G.edges()}
    edge_weight.update({(v, u): w for (u, v), w in list(edge_weight.items())})

    src_idx = []
    dst_idx = []
    w_arr = []
    for (u, v) in G.edges():
        w = G[u][v]["weight"]
        src_idx.append(u)
        dst_idx.append(v)
        w_arr.append(w)
        src_idx.append(v)
        dst_idx.append(u)
        w_arr.append(w)
    A = csr_matrix((w_arr, (src_idx, dst_idx)), shape=(n_nodes, n_nodes))
    D = dijkstra(A, directed=False)

    kappa = {}
    for (u, v) in sample_edges:
        nu = neigh[u]
        nv = neigh[v]
        sup_u = [u] + nu
        sup_v = [v] + nv
        if not nu or not nv:
            kappa[(u, v)] = np.nan
            continue
        w_u = np.array([ALPHA] + [(1 - ALPHA) / len(nu)] * len(nu))
        w_v = np.array([ALPHA] + [(1 - ALPHA) / len(nv)] * len(nv))
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
        return dict(mean_kappa_mal=np.nan, mean_kappa_T=np.nan,
                    cliff_delta=np.nan, mwu_p=np.nan,
                    n_mal=len(kappa_mal), n_T=len(kappa_T),
                    n_edges_total=n_e, n_edges_sampled=n_sample)

    cd = cliffs_delta(kappa_mal, kappa_T)
    _, p_u = mannwhitneyu(kappa_mal, kappa_T, alternative="two-sided")
    return dict(
        mean_kappa_mal=float(kappa_mal.mean()),
        mean_kappa_T=float(kappa_T.mean()),
        cliff_delta=float(cd),
        mwu_p=float(p_u),
        n_mal=len(kappa_mal),
        n_T=len(kappa_T),
        n_edges_total=n_e,
        n_edges_sampled=n_sample,
    )


# ---------------- main ----------------
def main():
    t0 = time.time()
    X_full, mask_mal, mask_T = load_tirosh_raw()
    X_sub, labels = subsample_cells(X_full, mask_mal, mask_T,
                                    n_per=N_PER_GROUP, seed=CELL_SEED)
    print(f"subsampled: {X_sub.shape[1]} cells "
          f"({(labels=='Malignant').sum()} mal / {(labels=='T cells').sum()} T)",
          flush=True)
    print(f"gene universe size: {X_sub.shape[0]}", flush=True)

    # Reference run: real HVG-2000
    print("\n=== Reference run: real HVG-2000 ===", flush=True)
    t_ref = time.time()
    hvg_idx = select_hvg(X_sub, n_genes=N_GENES)
    Xpca = project_pca(X_sub, hvg_idx, n_pca=N_PCA, seed=CELL_SEED)
    ref_res = run_pipeline(Xpca, labels, edge_seed=CELL_SEED)
    print(f"  HVG-2000: kappa_mal={ref_res['mean_kappa_mal']:+.4f}  "
          f"kappa_T={ref_res['mean_kappa_T']:+.4f}  "
          f"delta={ref_res['cliff_delta']:+.4f}  "
          f"p={ref_res['mwu_p']:.3g}  ({time.time()-t_ref:.1f}s)",
          flush=True)

    # Random-gene runs
    print(f"\n=== Random gene sets ({N_RANDOM_SEEDS} seeds) ===", flush=True)
    rows = []
    rows.append(dict(
        seed="HVG (real)",
        mean_kappa_mal=ref_res["mean_kappa_mal"],
        mean_kappa_T=ref_res["mean_kappa_T"],
        cliff_delta=ref_res["cliff_delta"],
        mwu_p=ref_res["mwu_p"],
        n_mal=ref_res["n_mal"],
        n_T=ref_res["n_T"],
        n_edges_total=ref_res["n_edges_total"],
        n_edges_sampled=ref_res["n_edges_sampled"],
    ))

    random_deltas = []
    for seed in range(1, N_RANDOM_SEEDS + 1):
        t1 = time.time()
        gene_idx = select_random_genes(X_sub, n_genes=N_GENES, seed=seed)
        Xpca = project_pca(X_sub, gene_idx, n_pca=N_PCA, seed=CELL_SEED)
        # use a deterministic edge seed per run that varies with gene seed so the
        # OR edge sample isn't identical across runs but is reproducible
        res = run_pipeline(Xpca, labels, edge_seed=CELL_SEED + seed)
        rows.append(dict(
            seed=seed,
            mean_kappa_mal=res["mean_kappa_mal"],
            mean_kappa_T=res["mean_kappa_T"],
            cliff_delta=res["cliff_delta"],
            mwu_p=res["mwu_p"],
            n_mal=res["n_mal"],
            n_T=res["n_T"],
            n_edges_total=res["n_edges_total"],
            n_edges_sampled=res["n_edges_sampled"],
        ))
        random_deltas.append(res["cliff_delta"])
        print(f"  seed={seed:2d}: kappa_mal={res['mean_kappa_mal']:+.4f}  "
              f"kappa_T={res['mean_kappa_T']:+.4f}  "
              f"delta={res['cliff_delta']:+.4f}  "
              f"p={res['mwu_p']:.3g}  ({time.time()-t1:.1f}s, total {time.time()-t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    csv_path = OUT / "t6_random_hvg_grid.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}", flush=True)

    # ---------------- summary stats ----------------
    rd = np.array(random_deltas, dtype=float)
    rd = rd[~np.isnan(rd)]
    ref_delta = ref_res["cliff_delta"]
    rand_mean = float(np.mean(rd)) if len(rd) else float("nan")
    rand_std = float(np.std(rd, ddof=1)) if len(rd) > 1 else float("nan")
    rand_lo = float(np.percentile(rd, 2.5)) if len(rd) else float("nan")
    rand_hi = float(np.percentile(rd, 97.5)) if len(rd) else float("nan")
    rand_median = float(np.median(rd)) if len(rd) else float("nan")

    # Z-score of the real HVG against the random distribution
    if rand_std > 0 and not np.isnan(rand_std):
        z_ref = (ref_delta - rand_mean) / rand_std
    else:
        z_ref = float("nan")

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rd, bins=15, color="#888888", alpha=0.75, edgecolor="black",
            label=f"random gene sets (n={len(rd)})")
    ax.axvline(ref_delta, color="C3", lw=2.5, ls="-",
               label=f"real HVG-2000 (delta={ref_delta:+.3f})")
    ax.axvline(rand_mean, color="C0", lw=1.5, ls="--",
               label=f"random mean ({rand_mean:+.3f})")
    ax.axvspan(rand_lo, rand_hi, color="C0", alpha=0.10,
               label=f"random 95% interval [{rand_lo:+.3f}, {rand_hi:+.3f}]")
    ax.axvline(0.0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Cliff's delta (Malignant - T cells per-cell mean kappa)")
    ax.set_ylabel("# random gene sets")
    ax.set_title("T6 negative control: HVG-2000 vs random 2000-gene sets\n"
                 "Tirosh 2016 melanoma, OR kappa (k=15, alpha=0.5, 3000 edges)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    png_path = OUT / "t6_random_hvg_summary.png"
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}", flush=True)

    # ---------------- interpret ----------------
    abs_ref = abs(ref_delta)
    abs_rand = abs(rand_mean)
    same_sign = (np.sign(ref_delta) == np.sign(rand_mean)) and rand_mean != 0
    in_random_ci = (rand_lo <= ref_delta <= rand_hi)

    if abs_rand < 0.10 and abs_ref >= 0.30:
        verdict = ("PASS: random Cliff's delta near zero while HVG-2000 is "
                   "strong - the HVG-2000 effect is genuine biological signal "
                   "from variance-driven gene selection.")
    elif in_random_ci and abs_ref < 0.10:
        verdict = ("NULL: both HVG-2000 and random are near zero - no real "
                   "effect to begin with.")
    elif in_random_ci:
        verdict = ("INTRINSIC: HVG-2000 falls within the random 95% interval - "
                   "the effect is intrinsic to ANY gene selection and likely "
                   "reflects cell-population structure that any random "
                   "projection captures, not HVG-specific signal.")
    elif same_sign and abs_ref > abs_rand:
        verdict = ("GRADED: random and HVG-2000 share sign but HVG is larger - "
                   "HVG amplifies a real underlying signal that any gene set "
                   "partly captures.")
    elif not same_sign:
        verdict = ("BIZARRE: HVG-2000 and random gene sets have OPPOSITE signs "
                   "- needs investigation; possible HVG-induced artifact or "
                   "stochastic split.")
    else:
        verdict = "UNCLASSIFIED: see numbers below."

    txt = []
    txt.append("T6 random-gene negative control on Tirosh 2016 melanoma")
    txt.append("=" * 60)
    txt.append(f"cells: 600 malignant + 600 T (cell seed {CELL_SEED})")
    txt.append(f"genes per set: {N_GENES} (universe = {X_sub.shape[0]})")
    txt.append(f"PCA={N_PCA}, kNN k={K_NN}, OR alpha={ALPHA}, "
               f"sampled edges={N_EDGES}")
    txt.append("")
    txt.append(f"REAL HVG-2000 Cliff's delta:        {ref_delta:+.4f}")
    txt.append(f"  mean kappa malignant: {ref_res['mean_kappa_mal']:+.4f}")
    txt.append(f"  mean kappa T cells:   {ref_res['mean_kappa_T']:+.4f}")
    txt.append(f"  Mann-Whitney p:       {ref_res['mwu_p']:.3g}")
    txt.append("")
    txt.append(f"Random gene sets (n={len(rd)}):")
    txt.append(f"  mean Cliff's delta:   {rand_mean:+.4f}")
    txt.append(f"  median:               {rand_median:+.4f}")
    txt.append(f"  std:                  {rand_std:.4f}")
    txt.append(f"  95% interval:         [{rand_lo:+.4f}, {rand_hi:+.4f}]")
    txt.append(f"  HVG-2000 z-score vs random: {z_ref:+.2f}")
    txt.append(f"  HVG-2000 inside random 95% CI: {in_random_ci}")
    txt.append("")
    txt.append(f"VERDICT: {verdict}")
    txt.append("")
    txt.append(f"runtime: {time.time()-t0:.1f}s")

    summary_path = OUT / "t6_summary.txt"
    summary_path.write_text("\n".join(txt) + "\n")
    print(f"Saved {summary_path}", flush=True)
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
