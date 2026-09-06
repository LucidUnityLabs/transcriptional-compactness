"""
N4 - Per-cell Ollivier-Ricci kappa as a within-tumor heterogeneity (ITH)
readout: does kappa stratify CNV-defined subclones within a single tumor?

Pipeline:
  1. Load Tirosh 2016 melanoma malignant cells (n=1257), keep patient IDs.
  2. Infer per-cell CNV with a manual chromosome-arm moving-average proxy
     (consistent with Tirosh's original InferCNV-style approach):
       - Map gene symbols to Ensembl chromosome + start position.
       - For each chromosome arm (1p, 1q, 2p, ..., 22q), order genes by
         genomic position, smooth log-expression with a 100-gene moving
         average, then for each cell take the arm's mean smoothed value.
       - Center each cell's per-arm CNV scores against a reference panel of
         Tirosh non-malignant immune cells (T+B+Macro) from the same patient
         pool.
  3. Per-tumor subclone assignment: Ward linkage on per-arm CNV scores within
     each patient (>=50 malignant cells), cut at k chosen so each cluster has
     >=10 cells (try k=5 down to k=2).
  4. Compute per-cell kappa on a kNN k=15 graph of all malignant cells in
     PCA-50.
  5. Per tumor, test kappa stratification across subclones via Kruskal-Wallis;
     report pairwise Cliff's delta between min and max subclone kappa means.
  6. Optional aggregate: do high-kappa subclones share aggressive CNVs
     (8q amp = MYC, 9p loss = CDKN2A, 17p loss = TP53)?

Outputs:
  n4_subclone_per_cell.csv
  n4_subclone_summary.csv
  n4_subclone_distributions.png
  n4_summary.txt
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist
from scipy.stats import kruskal, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data"
EXPR_FILE = DATA / "GSE72056_melanoma.txt"
GENE_POS_FILE = DATA / "ensembl_gene_pos.csv"

SEED = 20260508
RNG = np.random.default_rng(SEED)

N_HVG = 2000
N_PCA = 50
K_NN = 15
ALPHA = 0.5
N_EDGES = 4000

CNV_WINDOW = 100      # moving-average window in genes
MIN_GENES_PER_ARM = 30
MIN_TUMOR_CELLS = 50  # min malignant cells per patient to attempt subclone calling
MIN_SUBCLONE = 10     # min cells per subclone for testing


# ----------------------------------------------------------------------
# UCSC hg38 centromere midpoints (Mb), used to split each chromosome into
# p (short) and q (long) arms. Coordinates in bp.
# Source: UCSC Genome Browser cytoBand.txt approximate centromere midpoints.
# ----------------------------------------------------------------------
CENTROMERE_BP = {
    "1":  123_400_000,
    "2":   93_900_000,
    "3":   90_900_000,
    "4":   50_000_000,
    "5":   48_800_000,
    "6":   59_800_000,
    "7":   60_100_000,
    "8":   45_200_000,
    "9":   43_000_000,
    "10":  39_800_000,
    "11":  53_400_000,
    "12":  35_500_000,
    "13":  17_700_000,   # acrocentric (no useful p arm in expression data)
    "14":  17_200_000,
    "15":  19_000_000,
    "16":  36_800_000,
    "17":  25_100_000,
    "18":  18_500_000,
    "19":  26_200_000,
    "20":  28_100_000,
    "21":  12_000_000,
    "22":  15_000_000,
    "X":   61_000_000,
    "Y":   10_400_000,
}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_tirosh():
    print(f"Loading {EXPR_FILE.name} ...", flush=True)
    header = pd.read_csv(EXPR_FILE, sep="\t", nrows=4, header=None,
                         low_memory=False)
    cell_ids = header.iloc[0, 1:].values
    tumor_id = pd.to_numeric(header.iloc[1, 1:], errors="coerce").values
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values

    print("  reading expression matrix ...", flush=True)
    expr = pd.read_csv(EXPR_FILE, sep="\t", skiprows=4, header=None,
                       low_memory=False)
    gene_names = expr.iloc[:, 0].astype(str).values
    X = expr.iloc[:, 1:].values.astype(np.float32)  # genes x cells
    print(f"  matrix: {X.shape} (genes x cells)", flush=True)
    print(f"  malignant cells (=2): {(malignant == 2).sum()}", flush=True)
    print(f"  non-malignant (=1):   {(malignant == 1).sum()}", flush=True)
    return cell_ids, tumor_id, malignant, nonmal_type, gene_names, X


def load_gene_positions():
    """Returns dict gene_symbol -> (chrom, start_bp). One representative
    position per gene name (first non-scaffold chromosome encountered)."""
    df = pd.read_csv(GENE_POS_FILE)
    df = df.rename(columns={
        "Gene name": "gene",
        "Chromosome/scaffold name": "chrom",
        "Gene start (bp)": "start",
        "Gene end (bp)": "end",
    })
    keep = [str(i) for i in range(1, 23)] + ["X", "Y"]
    df = df[df["chrom"].isin(keep)].copy()
    df = df.dropna(subset=["gene", "chrom", "start"])
    df = df.drop_duplicates(subset=["gene"], keep="first")
    pos = {row.gene: (row.chrom, int(row.start)) for row in df.itertuples()}
    print(f"  gene positions: {len(pos)} unique gene symbols on chrs 1-22,X,Y",
          flush=True)
    return pos


# ----------------------------------------------------------------------
# Manual CNV proxy via per-chromosome-arm moving average
# ----------------------------------------------------------------------
def assign_arm(chrom: str, start: int) -> str | None:
    cm = CENTROMERE_BP.get(chrom)
    if cm is None:
        return None
    # Acrocentric chromosomes 13/14/15/21/22 effectively have no p-arm with
    # genes in expression data; we still split, but small p arms will be
    # filtered by MIN_GENES_PER_ARM.
    return f"{chrom}p" if start < cm else f"{chrom}q"


def build_arm_index(gene_names: np.ndarray, pos: dict[str, tuple[str, int]]):
    """Return dict arm -> list of (gene_idx, chrom, start), already sorted by
    start position. Only genes with known mapping included."""
    arm_genes: dict[str, list[tuple[int, str, int]]] = {}
    for i, g in enumerate(gene_names):
        info = pos.get(g)
        if info is None:
            continue
        chrom, start = info
        arm = assign_arm(chrom, start)
        if arm is None:
            continue
        arm_genes.setdefault(arm, []).append((i, chrom, start))
    for arm in list(arm_genes.keys()):
        arm_genes[arm].sort(key=lambda t: t[2])
    # Filter arms with too few genes
    arm_genes = {arm: lst for arm, lst in arm_genes.items()
                 if len(lst) >= MIN_GENES_PER_ARM}
    n_total = sum(len(lst) for lst in arm_genes.values())
    print(f"  arms retained: {len(arm_genes)} (>={MIN_GENES_PER_ARM} genes), "
          f"{n_total} mapped genes total", flush=True)
    return arm_genes


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along last axis. Edge handling: use cumulative
    sum and divide by actual window size, so edges are mean of available."""
    if window <= 1 or x.shape[-1] <= 1:
        return x
    n = x.shape[-1]
    w = min(window, n)
    # pad symmetric edges
    pad = w // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    csum = np.cumsum(xp, axis=-1)
    # smoothed[i] = (csum[i+w] - csum[i]) / w  on padded
    smooth = (csum[..., w:] - csum[..., :-w]) / w
    return smooth[..., :n]


def compute_cnv_scores(X: np.ndarray, arm_genes: dict, ref_mask: np.ndarray,
                       lfc_clip: float = 3.0):
    """For each cell and each arm, compute a CNV score:
       score(cell, arm) = mean over arm-genes of [smoothed(cell, arm) -
                                                  smoothed(ref_mean, arm)]
    where smoothed(.) is a CNV_WINDOW-gene moving average of the cell's
    expression vector restricted to the arm, in the arm's genomic order.

    Implementation:
      For each arm: extract X[arm_genes_idx, :] (G_arm x cells) and
      X[arm_genes_idx, :].mean(ref) (G_arm,). Subtract reference mean,
      clip, then smooth along genes, then take mean over genes -> per-cell
      arm score.

    Returns:
      cnv  : (cells, n_arms) array of per-arm CNV scores
      arm_names : list[str], length n_arms (sorted)
    """
    arm_names = sorted(arm_genes.keys(),
                       key=lambda a: (int(a[:-1].replace("X", "23").replace("Y", "24")), a[-1]))
    n_cells = X.shape[1]
    cnv = np.zeros((n_cells, len(arm_names)), dtype=np.float32)
    ref_X = X[:, ref_mask]
    print(f"  reference panel: {ref_mask.sum()} non-malignant cells", flush=True)
    for k, arm in enumerate(arm_names):
        gidx = np.array([t[0] for t in arm_genes[arm]])
        Xa = X[gidx, :]                          # G_arm x cells
        ref_mean = ref_X[gidx, :].mean(axis=1, keepdims=True)
        # Center against reference
        cent = Xa - ref_mean                     # G_arm x cells
        # Clip extreme values to limit influence of single high-expr genes
        np.clip(cent, -lfc_clip, lfc_clip, out=cent)
        # Smooth along genes (axis=0). moving_average operates on last axis,
        # so transpose -> (cells, G_arm), smooth, take mean over genes.
        smoothed = moving_average(cent.T, CNV_WINDOW)  # cells x G_arm
        cnv[:, k] = smoothed.mean(axis=1)
    return cnv, arm_names


# ----------------------------------------------------------------------
# Per-tumor subclone calling
# ----------------------------------------------------------------------
def call_subclones(cnv_cells: np.ndarray, k_options=(5, 4, 3, 2)) -> np.ndarray:
    """Hierarchical Ward clustering on per-arm CNV vectors. Tries k from
    largest to smallest in k_options; picks the largest k for which every
    cluster has >= MIN_SUBCLONE cells. Returns cluster labels in [1..k].
    If no k satisfies the constraint (very small or homogeneous tumor),
    returns all-1 labels."""
    n = cnv_cells.shape[0]
    if n < 2 * MIN_SUBCLONE:
        return np.ones(n, dtype=int)
    # condensed pairwise distance for linkage
    Z = linkage(pdist(cnv_cells, metric="euclidean"), method="ward")
    for k in k_options:
        if k > n:
            continue
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        if (sizes >= MIN_SUBCLONE).all() and len(sizes) >= 2:
            return labels
    # fall back to k=2 even if one cluster is small (still returns 2 clusters)
    labels = fcluster(Z, t=2, criterion="maxclust")
    return labels


# ----------------------------------------------------------------------
# kNN graph + Ollivier-Ricci kappa
# ----------------------------------------------------------------------
def build_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
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


def ollivier_ricci_per_cell(G: nx.Graph, n_edges: int, alpha: float,
                            rng: np.random.Generator):
    """Compute OR kappa on n_edges sampled edges, then map to per-cell mean
    over incident edges."""
    n_nodes = G.number_of_nodes()
    edges_all = list(G.edges())
    if n_edges < len(edges_all):
        sel = rng.choice(len(edges_all), size=n_edges, replace=False)
        sample_edges = [edges_all[i] for i in sel]
    else:
        sample_edges = edges_all

    # Build symmetric weighted sparse matrix and get all-pairs Dijkstra
    src, dst, w = [], [], []
    for (u, v) in G.edges():
        d = G[u][v]["weight"]
        src.extend([u, v])
        dst.extend([v, u])
        w.extend([d, d])
    A = csr_matrix((w, (src, dst)), shape=(n_nodes, n_nodes))
    print(f"  computing all-pairs Dijkstra ({n_nodes} nodes) ...", flush=True)
    D = dijkstra(A, directed=False)

    neigh = {n: list(G.neighbors(n)) for n in G.nodes()}
    edge_weight = {(u, v): G[u][v]["weight"] for (u, v) in G.edges()}
    edge_weight.update({(v, u): w_ for (u, v), w_ in list(edge_weight.items())})

    print(f"  computing OR on {len(sample_edges)} sampled edges ...",
          flush=True)
    kappa: dict[tuple[int, int], float] = {}
    for (u, v) in sample_edges:
        nu = neigh[u]
        nv = neigh[v]
        if not nu or not nv:
            kappa[(u, v)] = np.nan
            continue
        sup_u = [u] + nu
        sup_v = [v] + nv
        w_u = np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu))
        w_v = np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv))
        C = D[np.ix_(sup_u, sup_v)]
        if not np.all(np.isfinite(C)):
            kappa[(u, v)] = np.nan
            continue
        W = ot.emd2(w_u, w_v, C)
        d_uv = edge_weight[(u, v)]
        kappa[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan

    by_cell: dict[int, list[float]] = {n: [] for n in G.nodes()}
    for (u, v), kv in kappa.items():
        if np.isnan(kv):
            continue
        by_cell[u].append(kv)
        by_cell[v].append(kv)
    return np.array([float(np.mean(by_cell[n])) if by_cell[n] else np.nan
                     for n in range(n_nodes)])


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u, _ = mannwhitneyu(a, b, alternative="greater")
    return 2.0 * u / (len(a) * len(b)) - 1.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    cell_ids, tumor_id, malignant, nonmal_type, gene_names, X = load_tirosh()
    pos = load_gene_positions()
    arm_genes = build_arm_index(gene_names, pos)

    mask_mal = (malignant == 2)
    mask_imm = (malignant == 1) & np.isin(nonmal_type, [1, 2, 3])  # T,B,Macro
    print(f"  malignant n={int(mask_mal.sum())}, "
          f"immune-ref n={int(mask_imm.sum())}", flush=True)

    # ------ CNV inference (only need for malignant cells, but we compute on
    # the full matrix because the moving-average is computed per-cell on the
    # full gene set; we'll subset to malignant later) ------
    print("\n[CNV] running manual moving-average CNV proxy ...", flush=True)
    cnv_all, arm_names = compute_cnv_scores(X, arm_genes, ref_mask=mask_imm,
                                            lfc_clip=3.0)
    print(f"  CNV matrix: {cnv_all.shape}  (cells x arms)", flush=True)
    print(f"  arms: {arm_names}", flush=True)

    cnv_mal = cnv_all[mask_mal]                      # (n_mal, n_arms)
    cell_ids_mal = cell_ids[mask_mal]
    tumor_id_mal = tumor_id[mask_mal].astype(int)

    # ------ per-tumor subclone calling ------
    print("\n[Subclones] hierarchical Ward clustering per patient ...",
          flush=True)
    subclone_global = -np.ones(cnv_mal.shape[0], dtype=int)  # global id
    subclone_local = -np.ones(cnv_mal.shape[0], dtype=int)   # 1..k within-patient
    next_global_id = 1
    patient_subclone_count = {}
    for pid in np.unique(tumor_id_mal):
        idx = np.where(tumor_id_mal == pid)[0]
        if len(idx) < MIN_TUMOR_CELLS:
            subclone_local[idx] = 1
            subclone_global[idx] = next_global_id
            next_global_id += 1
            patient_subclone_count[int(pid)] = 1
            continue
        labels = call_subclones(cnv_mal[idx])
        # remap labels to 1..k contiguous
        uniq = np.unique(labels)
        remap = {old: new for new, old in enumerate(uniq, start=1)}
        labels = np.array([remap[v] for v in labels])
        subclone_local[idx] = labels
        for new in range(1, len(uniq) + 1):
            subclone_global[idx[labels == new]] = next_global_id
            next_global_id += 1
        patient_subclone_count[int(pid)] = len(uniq)
        print(f"  patient {int(pid):>3d}: {len(idx)} cells -> "
              f"{len(uniq)} subclone(s), sizes "
              f"{np.bincount(labels)[1:].tolist()}",
              flush=True)
    print(f"  total subclones across cohort: {next_global_id - 1}",
          flush=True)

    # ------ PCA + kNN + OR kappa on all malignant cells ------
    print("\n[Geometry] HVG -> PCA -> kNN -> Ollivier-Ricci ...", flush=True)
    var = X[:, mask_mal].var(axis=1)
    hvg = np.argsort(var)[::-1][:N_HVG]
    Xm = X[np.ix_(hvg, np.where(mask_mal)[0])].T.astype(np.float32)  # cells x genes
    Xm = Xm - Xm.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=N_PCA, random_state=SEED).fit_transform(Xm)
    print(f"  PCA: {Xpca.shape}", flush=True)
    G = build_knn_graph(Xpca, k=K_NN)
    print(f"  kNN graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges", flush=True)
    kappa = ollivier_ricci_per_cell(G, n_edges=N_EDGES, alpha=ALPHA, rng=RNG)
    finite = np.isfinite(kappa)
    print(f"  per-cell kappa: {finite.sum()} / {len(kappa)} finite, "
          f"mean={np.nanmean(kappa):+.4f}, std={np.nanstd(kappa):.4f}",
          flush=True)

    # ------ per-cell long-format dataframe ------
    # For the "top 3 CNV arms" we report the 3 arms with largest |score| for
    # each cell (signed).
    arm_arr = np.array(arm_names)
    top3_arms = []
    for i in range(cnv_mal.shape[0]):
        scores = cnv_mal[i]
        order = np.argsort(-np.abs(scores))[:3]
        top3 = ";".join(f"{arm_arr[j]}={scores[j]:+.2f}" for j in order)
        top3_arms.append(top3)

    df_cells = pd.DataFrame({
        "cell_id": cell_ids_mal,
        "patient_id": tumor_id_mal,
        "subclone_local": subclone_local,
        "subclone_global": subclone_global,
        "kappa": kappa,
        "top3_cnv_arms": top3_arms,
    })
    cells_csv = OUT / "n4_subclone_per_cell.csv"
    df_cells.to_csv(cells_csv, index=False)
    print(f"  saved {cells_csv}", flush=True)

    # ------ per-tumor subclone test ------
    print("\n[Test] per-tumor Kruskal-Wallis on kappa across subclones ...",
          flush=True)
    summary_rows = []
    aggressive_arms = ["8q", "17p", "9p"]  # MYC amp, TP53 loss, CDKN2A loss
    aggr_idx = {a: arm_names.index(a) for a in aggressive_arms
                if a in arm_names}

    for pid in np.unique(tumor_id_mal):
        sub_pid = df_cells[df_cells["patient_id"] == pid]
        n_total = len(sub_pid)
        local_labels = sub_pid["subclone_local"].values
        kappa_pid = sub_pid["kappa"].values

        # Need >=2 subclones each with >=10 finite-kappa cells to test
        valid_clones = []
        for c in np.unique(local_labels):
            mask_c = (local_labels == c) & np.isfinite(kappa_pid)
            if mask_c.sum() >= MIN_SUBCLONE:
                valid_clones.append(c)
        n_subclones_full = patient_subclone_count[int(pid)]

        row = {
            "patient_id": int(pid),
            "n_cells": int(n_total),
            "n_subclones_called": n_subclones_full,
            "n_subclones_testable": len(valid_clones),
            "kw_p": np.nan,
            "kw_H": np.nan,
            "max_pairwise_cliff_delta": np.nan,
            "high_kappa_subclone": -1,
            "low_kappa_subclone": -1,
            "high_kappa_mean": np.nan,
            "low_kappa_mean": np.nan,
            "high_subclone_n": 0,
            "low_subclone_n": 0,
            "high_subclone_top_cnv": "",
            "low_subclone_top_cnv": "",
        }
        # also report aggressive-arm scores per subclone
        for arm in aggr_idx:
            row[f"high_{arm}"] = np.nan
            row[f"low_{arm}"] = np.nan

        if len(valid_clones) >= 2:
            groups = []
            means = {}
            ns = {}
            for c in valid_clones:
                mask_c = (local_labels == c) & np.isfinite(kappa_pid)
                vals = kappa_pid[mask_c]
                groups.append(vals)
                means[c] = float(vals.mean())
                ns[c] = int(mask_c.sum())
            try:
                H, p = kruskal(*groups)
            except ValueError:
                H, p = np.nan, np.nan
            row["kw_p"] = float(p) if not np.isnan(p) else np.nan
            row["kw_H"] = float(H) if not np.isnan(H) else np.nan

            # Identify high and low kappa subclones (max vs min mean)
            c_hi = max(means, key=means.get)
            c_lo = min(means, key=means.get)
            if c_hi != c_lo:
                a = kappa_pid[(local_labels == c_hi) & np.isfinite(kappa_pid)]
                b = kappa_pid[(local_labels == c_lo) & np.isfinite(kappa_pid)]
                row["max_pairwise_cliff_delta"] = float(cliffs_delta(a, b))
                row["high_kappa_subclone"] = int(c_hi)
                row["low_kappa_subclone"] = int(c_lo)
                row["high_kappa_mean"] = means[c_hi]
                row["low_kappa_mean"] = means[c_lo]
                row["high_subclone_n"] = ns[c_hi]
                row["low_subclone_n"] = ns[c_lo]

                # Mean per-arm CNV profile of each subclone, top arms by |mean|
                idx_hi = sub_pid.index[(sub_pid["subclone_local"] == c_hi)]
                idx_lo = sub_pid.index[(sub_pid["subclone_local"] == c_lo)]
                # cnv_mal indexed positionally; map dataframe index back
                # (df_cells row order matches cnv_mal row order)
                pos_hi = sub_pid.index[(sub_pid["subclone_local"] == c_hi)].to_numpy()
                pos_lo = sub_pid.index[(sub_pid["subclone_local"] == c_lo)].to_numpy()
                prof_hi = cnv_mal[pos_hi].mean(axis=0)
                prof_lo = cnv_mal[pos_lo].mean(axis=0)
                top_hi = np.argsort(-np.abs(prof_hi))[:3]
                top_lo = np.argsort(-np.abs(prof_lo))[:3]
                row["high_subclone_top_cnv"] = ";".join(
                    f"{arm_arr[j]}={prof_hi[j]:+.2f}" for j in top_hi)
                row["low_subclone_top_cnv"] = ";".join(
                    f"{arm_arr[j]}={prof_lo[j]:+.2f}" for j in top_lo)
                for arm, j in aggr_idx.items():
                    row[f"high_{arm}"] = float(prof_hi[j])
                    row[f"low_{arm}"] = float(prof_lo[j])

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summ_csv = OUT / "n4_subclone_summary.csv"
    df_summary.to_csv(summ_csv, index=False)
    print(f"  saved {summ_csv}", flush=True)

    # ------ per-patient stripplot ------
    testable = df_summary[df_summary["n_subclones_testable"] >= 2]
    n_panels = len(testable)
    if n_panels >= 1:
        ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k_panel, (_, prow) in enumerate(testable.iterrows()):
            ax = axes.flat[k_panel]
            ax.set_visible(True)
            pid = int(prow["patient_id"])
            sub_pid = df_cells[df_cells["patient_id"] == pid]
            for c in sorted(sub_pid["subclone_local"].unique()):
                vals = sub_pid[sub_pid["subclone_local"] == c]["kappa"].values
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                xs = np.full_like(vals, c, dtype=float) + RNG.normal(0, 0.06, len(vals))
                ax.scatter(xs, vals, s=8, alpha=0.55, label=f"sc{c} (n={len(vals)})")
            ax.set_title(f"P{pid}  KW p={prow['kw_p']:.2g}  "
                         f"delta={prow['max_pairwise_cliff_delta']:+.2f}",
                         fontsize=9)
            ax.set_xlabel("subclone")
            ax.set_ylabel("per-cell kappa")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.legend(fontsize=6, loc="best")
        plt.suptitle("N4: per-cell Ollivier-Ricci kappa across CNV-defined "
                     "subclones (Tirosh 2016 melanoma)", y=1.0)
        plt.tight_layout()
        png = OUT / "n4_subclone_distributions.png"
        plt.savefig(png, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  saved {png}", flush=True)

    # ------ summary text ------
    n_testable = int(testable.shape[0])
    sig = testable[testable["kw_p"] < 0.05]
    n_sig = int(sig.shape[0])
    deltas = testable["max_pairwise_cliff_delta"].dropna().values

    # high-kappa CNV pattern aggregation
    aggr_rows = []
    for arm in aggr_idx:
        hi = testable[f"high_{arm}"].dropna().values
        lo = testable[f"low_{arm}"].dropna().values
        if len(hi) and len(lo):
            try:
                # paired Wilcoxon would be more proper; use Mann-Whitney for
                # cross-tumor comparison since each tumor contributes one pair
                from scipy.stats import wilcoxon
                W, p_w = wilcoxon(hi, lo)
            except Exception:
                W, p_w = np.nan, np.nan
            aggr_rows.append((arm, float(np.mean(hi)), float(np.mean(lo)),
                              float(np.mean(hi - lo)), float(p_w)))

    txt = []
    txt.append("N4 - per-cell OR kappa across CNV-defined subclones, "
               "Tirosh 2016 melanoma")
    txt.append("=" * 78)
    txt.append(f"seed: {SEED}")
    txt.append(f"malignant cells loaded:    {int(mask_mal.sum())}")
    txt.append(f"immune-ref cells (T+B+M):  {int(mask_imm.sum())}")
    txt.append(f"chromosome arms used:      {len(arm_names)}  ({arm_names})")
    txt.append(f"HVG: {N_HVG}, PCA: {N_PCA}, kNN k: {K_NN}, OR alpha: {ALPHA}, "
               f"sampled edges: {N_EDGES}")
    txt.append("")
    txt.append(f"patients with malignant cells:                "
               f"{df_summary.shape[0]}")
    txt.append(f"patients with >=2 testable subclones (>=10 cells each): "
               f"{n_testable}")
    txt.append(f"... of those, KW p<0.05:                       "
               f"{n_sig}  ({100.0*n_sig/max(n_testable,1):.1f}%)")
    txt.append("")
    if len(deltas):
        txt.append(f"|Cliff's delta| (max subclone vs min subclone) "
                   f"across testable tumors:")
        txt.append(f"  median |delta|: {np.median(np.abs(deltas)):+.3f}")
        txt.append(f"  mean   |delta|: {np.mean(np.abs(deltas)):+.3f}")
        txt.append(f"  max    |delta|: {np.max(np.abs(deltas)):+.3f}")
        txt.append(f"  min    |delta|: {np.min(np.abs(deltas)):+.3f}")
        n_large = int((np.abs(deltas) >= 0.33).sum())
        txt.append(f"  fraction with |delta|>=0.33 (medium effect): "
                   f"{n_large}/{len(deltas)}")
    txt.append("")
    txt.append("Aggressive-CNV enrichment (high-kappa - low-kappa subclones)")
    txt.append("Wilcoxon paired test across testable tumors:")
    for arm, hi_m, lo_m, diff, p_w in aggr_rows:
        sign = "more amp/loss in high-kappa" if abs(hi_m) > abs(lo_m) else \
               "more amp/loss in low-kappa"
        txt.append(f"  arm {arm}:  hi-mean={hi_m:+.3f}  lo-mean={lo_m:+.3f}  "
                   f"hi-lo={diff:+.3f}  Wilcoxon p={p_w:.3g}   ({sign})")
    txt.append("")

    # Per-tumor summary table
    txt.append("Per-tumor results (testable only):")
    txt.append("-" * 78)
    cols = ["patient_id", "n_cells", "n_subclones_testable", "kw_p",
            "max_pairwise_cliff_delta", "high_kappa_subclone",
            "low_kappa_subclone", "high_kappa_mean", "low_kappa_mean",
            "high_subclone_top_cnv"]
    sub_t = testable[cols].copy()
    sub_t["kw_p"] = sub_t["kw_p"].apply(lambda x: f"{x:.3g}")
    sub_t["max_pairwise_cliff_delta"] = sub_t[
        "max_pairwise_cliff_delta"].apply(lambda x: f"{x:+.3f}")
    sub_t["high_kappa_mean"] = sub_t["high_kappa_mean"].apply(
        lambda x: f"{x:+.4f}")
    sub_t["low_kappa_mean"] = sub_t["low_kappa_mean"].apply(
        lambda x: f"{x:+.4f}")
    txt.append(sub_t.to_string(index=False))
    txt.append("")

    # Verdict
    if n_testable == 0:
        verdict = ("INCONCLUSIVE: no patient had >=2 subclones with >=10 "
                   "cells each.")
    else:
        frac_sig = n_sig / n_testable
        med_abs = float(np.median(np.abs(deltas))) if len(deltas) else 0.0
        if frac_sig >= 0.5 and med_abs >= 0.20:
            verdict = (f"STRONG ITH READOUT: {n_sig}/{n_testable} testable "
                       f"tumors show KW p<0.05 with median |Cliff's delta|"
                       f"={med_abs:.3f}. Per-cell kappa stratifies CNV-defined "
                       "subclones within tumors -> useful single-tumor ITH "
                       "diagnostic axis.")
        elif frac_sig >= 0.25 or med_abs >= 0.15:
            verdict = (f"PARTIAL ITH READOUT: {n_sig}/{n_testable} "
                       f"({100*frac_sig:.0f}%) tumors show KW p<0.05; "
                       f"median |delta|={med_abs:.3f}. Kappa stratifies "
                       "subclones in some but not all tumors.")
        else:
            verdict = (f"WITHIN-TUMOR UNIFORM: only {n_sig}/{n_testable} "
                       f"({100*frac_sig:.0f}%) tumors show KW p<0.05; "
                       f"median |delta|={med_abs:.3f}. Malignancy bumps "
                       "kappa but does not stratify CNV-defined subclones "
                       "within tumors.")
    txt.append(f"VERDICT: {verdict}")
    txt.append("")
    txt.append(f"runtime: {time.time()-t0:.1f}s")

    summary_path = OUT / "n4_summary.txt"
    summary_path.write_text("\n".join(txt) + "\n")
    print(f"  saved {summary_path}", flush=True)
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
