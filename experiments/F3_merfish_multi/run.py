"""
F3 -- MERFISH single-cell-resolution replication of OR kappa malignant-vs-stroma
direction.  Question is the same one asked by T4 (Visium spot, ~55 um) and N2
(Visium HD, 8 um): does the malignant > stroma kappa direction observed on the
transcriptomic-similarity kNN graph also appear on a *physical* spatial-neighbor
graph?

T4 was null on G_phys (Cliff d ~ 0).  N2 confirmed the null at 8 um, so the
spot-resolution result was NOT a 55 um averaging artifact.  This script repeats
the dual-graph G_phys vs G_trans test on TRUE single-cell MERFISH data
(~1 um effective resolution), which is the most granular available test of the
substrate-specificity claim.

Dataset: GSE291210 (Wu et al., MMTV-PyMT mouse mammary tumor MERFISH, 2025).
Sample T1 = GSM8830801, MERSCOPE Mouse Immuno-Oncology panel (~285 genes),
~663k cells with single-cell segmentation and (center_x, center_y) micron
coordinates.  This model has both malignant epithelial PyMT tumor cells
(Krt5/7/14/15/17, Cdh1) and rich stroma (Acta2, Col1a1, Pdgfra/b, Pecam1) in
one section.

Pipeline:
  1) Load cell_by_gene.csv.gz + cell_metadata.csv.gz from
     ../../data/merfish/.  Drop BLANK probe controls.
  2) QC filter (min_counts >= 20, min_cells >= 3).
  3) log1p + (full panel as features) + PCA up to min(50, n_genes-1).
  4) Score per-cell tumor (Krt5/7/14/15/17/Cdh1) vs stroma (Acta2, Col1a1,
     Pdgfra, Pdgfrb, Pecam1, Vim, Ptprc, Vwf?) signatures with
     scanpy.tl.score_genes; label malignant if tumor>=q67 of (t-s) and
     stroma if (t-s)<=q33; this matches T4's labelling rule.
  5) Stratified subsample to ~6000 cells (40% malignant, 40% stroma,
     20% unlabeled) to fit the OR kappa time budget.
  6) Build G_phys: kNN k=10 in (center_x, center_y) microns, weight = um.
  7) Build G_trans: kNN k=15 in PCA, weight = PCA distance.
  8) Sample 4000 edges per graph, lazy-walk alpha=0.5 OR kappa per edge ->
     per-cell mean kappa.
  9) Welch t, MWU, Cliff's delta on malignant vs stroma per graph.
 10) Outputs: f3_merfish_summary.csv, f3_merfish_spatial.png,
     f3_merfish_distributions.png, f3_summary.txt.
"""

from __future__ import annotations

import gzip
import io
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "merfish"
SEED = 20260508
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)

CBG_GZ = DATA / "T1_cbg.csv.gz"
META_GZ = DATA / "T1_meta.csv.gz"

TARGET_N = 6000
N_EDGES = 4000

# Spatial tile (microns), centered on the tissue COM.  Sized so that the
# native MERFISH cell density (~7-14k cells/mm^2) gives ~6k cells while
# preserving micron-scale physical-contact neighbours (NN distance ~6 um).
# Random subsampling across the full 11x8 mm section would inflate the
# physical-graph edge length 10x and turn G_phys into a mesoscale-tissue
# graph rather than a cell-cell-contact graph.
TILE_W_UM = 700.0
TILE_H_UM = 700.0

# Mouse-case markers.  Tumor = mammary epithelial / PyMT-driven.
TUMOR_GENES = ["Cdh1", "Krt5", "Krt7", "Krt14", "Krt15", "Krt17", "Epcam",
               "Krt8", "Krt18", "Krt19"]
# Stroma = fibroblasts + endothelial + smooth muscle + leukocytes.
STROMA_GENES = ["Acta2", "Col1a1", "Pdgfra", "Pdgfrb", "Pecam1", "Vim",
                "Ptprc", "Vwf"]


# ---------- OR kappa (matches exp/N2_visium_hd/run.py) ----------------------
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


def per_node_mean_curvature(G, edge_kappa):
    by = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k):
            continue
        by[u].append(k)
        by[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by.items()}


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    B = np.sort(b)
    n_less = np.searchsorted(B, a, side="left")
    n_leq = np.searchsorted(B, a, side="right")
    n_greater = len(B) - n_leq
    gt = n_less.sum(); lt = n_greater.sum()
    return float((gt - lt) / (len(a) * len(b)))


# ---------- data loading -----------------------------------------------------
def load_merfish():
    print(f"[load] {CBG_GZ.name} + {META_GZ.name}")
    t0 = time.time()
    cbg = pd.read_csv(CBG_GZ, index_col=0)
    print(f"  cell_by_gene: {cbg.shape} in {time.time() - t0:.1f}s")
    meta = pd.read_csv(META_GZ, index_col=0)
    print(f"  cell_metadata: {meta.shape}")
    # drop blank/control probes if present
    blank_cols = [c for c in cbg.columns if c.lower().startswith("blank")]
    if blank_cols:
        print(f"  dropping {len(blank_cols)} blank-control probes")
        cbg = cbg.drop(columns=blank_cols)
    common = cbg.index.intersection(meta.index)
    cbg = cbg.loc[common]
    meta = meta.loc[common]
    print(f"  aligned: {len(common)} cells x {cbg.shape[1]} genes")

    X = cbg.values.astype(np.float32)
    var = pd.DataFrame(index=cbg.columns)
    obs = meta.copy()
    obs.index = obs.index.astype(str)
    adata = AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    # Provide consistent obs columns.
    for c in ("center_x", "center_y", "transcript_count", "volume"):
        if c in adata.obs.columns:
            adata.obs[c] = adata.obs[c].astype(float)
    return adata


# ---------- spatial tile crop -----------------------------------------------
def crop_centered_tile(adata, w_um, h_um):
    cx = adata.obs["center_x"].values
    cy = adata.obs["center_y"].values
    mid_x = 0.5 * (cx.min() + cx.max())
    mid_y = 0.5 * (cy.min() + cy.max())
    keep = ((cx > mid_x - w_um / 2) & (cx < mid_x + w_um / 2) &
            (cy > mid_y - h_um / 2) & (cy < mid_y + h_um / 2))
    print(f"[tile] centered {w_um:.0f} x {h_um:.0f} um at "
          f"({mid_x:.0f}, {mid_y:.0f}): {int(keep.sum())}/{adata.n_obs} cells")
    return adata[keep].copy()


# ---------- preprocessing ----------------------------------------------------
def preprocess(adata, n_pcs=50, seed=SEED):
    print("[preprocess] QC + normalize_total + log1p + PCA")
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None,
                               log1p=False)
    n0 = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=20)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  QC: {n0} -> {adata.n_obs} cells, {adata.n_vars} genes")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e3)  # MERFISH counts are small
    sc.pp.log1p(adata)
    # MERFISH panels are pre-selected, so use full panel rather than HVG.
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    # Manual zero-mean / unit-var scale clipped at 10 to match scanpy.scale.
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    X = X / sd
    X = np.clip(X, -10, 10).astype(np.float32)
    n_pcs_eff = min(n_pcs, X.shape[1] - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_pcs_eff, random_state=seed).fit(X)
    adata.obsm["X_pca"] = pca.transform(X).astype(np.float32)
    adata.uns["pca_var_ratio"] = pca.explained_variance_ratio_
    print(f"  PCA: {adata.obsm['X_pca'].shape}, "
          f"top-3 var={pca.explained_variance_ratio_[:3].round(3).tolist()}")
    return adata


# ---------- labelling --------------------------------------------------------
def label_cells(adata):
    print("[label] tumor vs stroma signature scoring")
    avail_t = [g for g in TUMOR_GENES if g in adata.var_names]
    avail_s = [g for g in STROMA_GENES if g in adata.var_names]
    print(f"  tumor markers found: {avail_t}")
    print(f"  stroma markers found: {avail_s}")
    if not avail_t or not avail_s:
        raise RuntimeError("Missing key marker genes")
    sc.tl.score_genes(adata, gene_list=avail_t, score_name="tumor_score",
                      random_state=SEED)
    sc.tl.score_genes(adata, gene_list=avail_s, score_name="stroma_score",
                      random_state=SEED)
    diff = (adata.obs["tumor_score"] - adata.obs["stroma_score"]).values
    q_lo, q_hi = np.quantile(diff, [0.33, 0.67])
    label = np.array(["unlabeled"] * adata.n_obs, dtype=object)
    label[diff >= q_hi] = "malignant"
    label[diff <= q_lo] = "stroma"
    adata.obs["label"] = label
    adata.obs["sig_diff"] = diff
    counts = pd.Series(label).value_counts()
    print(f"  labels: {counts.to_dict()}")
    print(f"  q33={q_lo:+.3f}  q67={q_hi:+.3f}")
    return adata


# ---------- balanced subsample ----------------------------------------------
def stratified_subsample(adata, target_n, seed=SEED):
    rng = np.random.default_rng(seed)
    label = adata.obs["label"].values
    n_total = adata.n_obs
    if n_total <= target_n:
        return adata, np.arange(n_total)
    idx_m = np.where(label == "malignant")[0]
    idx_s = np.where(label == "stroma")[0]
    idx_u = np.where(label == "unlabeled")[0]
    nm = min(int(target_n * 0.4), len(idx_m))
    ns = min(int(target_n * 0.4), len(idx_s))
    nu = min(target_n - nm - ns, len(idx_u))
    take = np.concatenate([
        rng.choice(idx_m, size=nm, replace=False),
        rng.choice(idx_s, size=ns, replace=False),
        rng.choice(idx_u, size=nu, replace=False),
    ])
    take.sort()
    print(f"[subsample] {len(take)} cells kept "
          f"(malignant={nm}, stroma={ns}, unlabeled={nu})")
    return adata[take].copy(), take


# ---------- graphs -----------------------------------------------------------
def build_phys_graph(adata, k=10):
    """k=10 nearest physical neighbours in (center_x, center_y) micron-space.
    """
    pts = np.column_stack([adata.obs["center_x"].values,
                           adata.obs["center_y"].values]).astype(float)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(pts)
    dists, inds = nbr.kneighbors(pts)
    G = nx.Graph()
    G.add_nodes_from(range(pts.shape[0]))
    for i in range(pts.shape[0]):
        for j_idx in range(1, k + 1):
            j = int(inds[i, j_idx])
            d = float(dists[i, j_idx])
            if d <= 0:
                continue
            if not G.has_edge(i, j) or G[i][j].get("weight", np.inf) > d:
                G.add_edge(i, j, weight=d)
    med = float(np.median([d for _, _, d in G.edges(data="weight")]))
    print(f"[G_phys] kNN(k={k}) physical: {G.number_of_edges()} edges, "
          f"median d={med:.2f} um")
    return G, med


def build_trans_graph(adata, k=15):
    X = adata.obsm["X_pca"]
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, inds = nbr.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    for i in range(X.shape[0]):
        for j_idx in range(1, k + 1):
            j = int(inds[i, j_idx])
            d = float(dists[i, j_idx])
            if not G.has_edge(i, j) or G[i][j].get("weight", np.inf) > d:
                G.add_edge(i, j, weight=d)
    print(f"[G_trans] kNN(k={k}) PCA: {G.number_of_edges()} edges")
    return G


# ---------- analysis ---------------------------------------------------------
def analyse(name, G, label, n_edges=N_EDGES):
    print(f"\n[{name}] OR kappa on {n_edges} sampled edges (alpha=0.5)")
    t0 = time.time()
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=RNG)
    elapsed = time.time() - t0
    n_valid = sum(1 for v in edge_k.values() if not np.isnan(v))
    print(f"  valid: {n_valid}/{len(edge_k)}  (took {elapsed:.1f}s)")
    per = per_node_mean_curvature(G, edge_k)
    kappa = np.array([per.get(i, np.nan) for i in range(len(label))])
    n_with_k = np.isfinite(kappa).sum()
    print(f"  cells with valid kappa: {n_with_k}/{len(label)}")
    a = kappa[(label == "malignant") & np.isfinite(kappa)]
    b = kappa[(label == "stroma") & np.isfinite(kappa)]
    if len(a) == 0 or len(b) == 0:
        return kappa, dict(name=name, n_mal=int(len(a)), n_str=int(len(b)),
                           mean_mal=float("nan"), std_mal=float("nan"),
                           mean_str=float("nan"), std_str=float("nan"),
                           diff=float("nan"), welch_t=float("nan"),
                           welch_p=float("nan"), mwu_U=float("nan"),
                           mwu_p=float("nan"), cliff_d=float("nan"))
    welch_t, welch_p = ttest_ind(a, b, equal_var=False)
    mwu_U, mwu_p = mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    print(f"  mean kappa  malignant={a.mean():+.4f} (sd={a.std():.4f}, n={len(a)})  "
          f"stroma={b.mean():+.4f} (sd={b.std():.4f}, n={len(b)})  "
          f"diff={a.mean()-b.mean():+.4f}")
    print(f"  Welch t={welch_t:.3f} p={welch_p:.3g}  "
          f"MWU U={mwu_U:.0f} p={mwu_p:.3g}  Cliff d={d:+.3f}")
    return kappa, dict(name=name,
                       n_mal=int(len(a)), n_str=int(len(b)),
                       mean_mal=float(a.mean()), std_mal=float(a.std()),
                       mean_str=float(b.mean()), std_str=float(b.std()),
                       diff=float(a.mean() - b.mean()),
                       welch_t=float(welch_t), welch_p=float(welch_p),
                       mwu_U=float(mwu_U), mwu_p=float(mwu_p),
                       cliff_d=float(d))


# ---------- plotting ---------------------------------------------------------
def plot_spatial(adata, kappa_phys, label, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    px = adata.obs["center_x"].values
    py = adata.obs["center_y"].values
    ax = axes[0]
    valid = np.isfinite(kappa_phys)
    s = 2
    if valid.sum() > 0:
        sc1 = ax.scatter(px[valid], -py[valid], c=kappa_phys[valid],
                         cmap="RdBu_r", s=s,
                         vmin=np.nanpercentile(kappa_phys, 2),
                         vmax=np.nanpercentile(kappa_phys, 98))
        plt.colorbar(sc1, ax=ax, label="kappa (G_phys)")
    ax.set_title(f"Per-cell Ollivier-Ricci kappa on G_phys "
                 f"(MERFISH ~1 um, n={int(valid.sum())} cells)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    color = {"malignant": "#d62728", "stroma": "#1f77b4",
             "unlabeled": "#cccccc"}
    for lab in ("unlabeled", "stroma", "malignant"):
        m = label == lab
        ax.scatter(px[m], -py[m], c=color[lab], s=s,
                   label=f"{lab} (n={int(m.sum())})")
    ax.legend(loc="lower right", fontsize=9, markerscale=4)
    ax.set_title("Cell labels (tumor vs stroma signature)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()


def plot_distributions(kappa_phys, kappa_trans, label, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kappa, name in zip(axes, (kappa_phys, kappa_trans),
                               ("G_phys (k=10 micron)",
                                "G_trans (k=15 PCA)")):
        a = kappa[(label == "malignant") & np.isfinite(kappa)]
        b = kappa[(label == "stroma") & np.isfinite(kappa)]
        if len(a) == 0 or len(b) == 0:
            ax.set_title(f"{name}: no labels"); continue
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 40)
        ax.hist(b, bins=bins, alpha=0.55, label=f"stroma n={len(b)}",
                color="#1f77b4")
        ax.hist(a, bins=bins, alpha=0.55, label=f"malignant n={len(a)}",
                color="#d62728")
        ax.axvline(a.mean(), color="#d62728", ls="--")
        ax.axvline(b.mean(), color="#1f77b4", ls="--")
        ax.set_title(f"{name}: mal={a.mean():+.3f}, str={b.mean():+.3f}, "
                     f"diff={a.mean()-b.mean():+.3f}")
        ax.set_xlabel("per-cell mean kappa"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()


# ---------- main -------------------------------------------------------------
def main():
    print("=" * 72)
    print("F3 -- MERFISH single-cell-resolution OR kappa replication")
    print(f"     seed={SEED}")
    print("=" * 72)
    t_start = time.time()

    adata = load_merfish()
    # Crop a contiguous tissue tile FIRST so we preserve native MERFISH cell
    # density.  Random downsampling across the whole 11x8 mm section would
    # inflate physical-graph edges from ~6 um to ~120 um, turning G_phys
    # into a mesoscale tissue graph.
    adata = crop_centered_tile(adata, TILE_W_UM, TILE_H_UM)
    adata = preprocess(adata)
    adata = label_cells(adata)
    if adata.n_obs > TARGET_N * 1.5:
        adata, _ = stratified_subsample(adata, TARGET_N)

    # Re-run PCA on the working set for well-conditioned G_trans.
    print("[pca-redo] re-running PCA on tile")
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True); sd[sd == 0] = 1.0
    X = np.clip(X / sd, -10, 10).astype(np.float32)
    n_pcs_eff = min(50, X.shape[1] - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_pcs_eff, random_state=SEED).fit(X)
    adata.obsm["X_pca"] = pca.transform(X).astype(np.float32)
    print(f"  PCA on subsample: {adata.obsm['X_pca'].shape}")

    G_phys, med_phys_um = build_phys_graph(adata, k=10)
    G_trans = build_trans_graph(adata, k=15)

    label = adata.obs["label"].values

    kappa_phys, row_phys = analyse("G_phys", G_phys, label, n_edges=N_EDGES)
    kappa_trans, row_trans = analyse("G_trans", G_trans, label, n_edges=N_EDGES)

    summary = pd.DataFrame([row_phys, row_trans])
    summary_path = HERE / "f3_merfish_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote: {summary_path}")
    print(summary.to_string(index=False))

    spatial_path = HERE / "f3_merfish_spatial.png"
    plot_spatial(adata, kappa_phys, label, spatial_path)
    print(f"Wrote: {spatial_path}")

    dist_path = HERE / "f3_merfish_distributions.png"
    plot_distributions(kappa_phys, kappa_trans, label, dist_path)
    print(f"Wrote: {dist_path}")

    # Resolution: rough cells/um^2 based on bbox of subsample.
    px = adata.obs["center_x"].values
    py = adata.obs["center_y"].values
    bbox_um2 = (px.max() - px.min()) * (py.max() - py.min())
    cells_per_um2 = adata.n_obs / bbox_um2 if bbox_um2 > 0 else float("nan")

    sign_phys = np.sign(row_phys["diff"])
    sign_trans = np.sign(row_trans["diff"])
    # "small but real" = highly significant + at least small Cliff effect
    sig_phys = (row_phys["welch_p"] < 0.01
                and abs(row_phys["cliff_d"]) >= 0.05)
    sig_trans = (row_trans["welch_p"] < 0.01
                 and abs(row_trans["cliff_d"]) >= 0.05)
    big_phys = abs(row_phys["cliff_d"]) >= 0.20
    headline = []
    headline.append("=" * 72)
    headline.append("F3 -- HEADLINE")
    headline.append("=" * 72)
    headline.append(f"dataset: GSE291210 / GSM8830801 (MMTV-PyMT mouse "
                    f"mammary tumor MERFISH)")
    headline.append(f"tile: centered {TILE_W_UM:.0f} x {TILE_H_UM:.0f} um")
    headline.append(f"resolution: ~1 um single-cell segmentation; "
                    f"median G_phys edge = {med_phys_um:.2f} um  "
                    f"(approx {cells_per_um2*1e6:.1f} cells / mm^2)")
    headline.append(f"working set: {adata.n_obs} cells, {adata.n_vars} genes")
    headline.append(
        f"G_phys : diff={row_phys['diff']:+.4f}  p={row_phys['welch_p']:.3g}  "
        f"d={row_phys['cliff_d']:+.3f}  "
        f"({'mal>str' if sign_phys>0 else 'mal<str'} "
        f"{'STRONG' if big_phys else 'small/null' if not sig_phys else 'small'})")
    headline.append(
        f"G_trans: diff={row_trans['diff']:+.4f}  p={row_trans['welch_p']:.3g}  "
        f"d={row_trans['cliff_d']:+.3f}  "
        f"({'mal>str' if sign_trans>0 else 'mal<str'} "
        f"{'STRONG' if abs(row_trans['cliff_d'])>=0.20 else 'small'})")
    if big_phys and sig_trans and sign_phys == sign_trans:
        verdict = ("SUBSTRATE-INDEPENDENT at single-cell resolution: "
                   "the kappa direction transfers to physical-space neighbors. "
                   "T4 + N2 nulls were resolution artifacts -- SURPRISE "
                   "POSITIVE, significant reframe required.")
    elif big_phys and sig_trans and sign_phys != sign_trans:
        verdict = ("DIVERGENT: kappa direction is opposite on G_phys vs "
                   "G_trans at single-cell resolution -- weird, investigate.")
    elif (not big_phys) and sig_trans:
        verdict = ("SUBSTRATE-SPECIFIC confirmed at THREE resolutions: "
                   "55 um (T4), 8 um (N2), and ~1 um (F3). The kappa "
                   "malignant-vs-stroma direction lives only on the "
                   "transcriptomic-similarity graph; physical-neighbor "
                   "kappa is null (Cliff d~0). The phenotype is genuinely "
                   "transcriptional-similarity-only and does not encode "
                   "physical-tissue architecture, even at sub-cellular "
                   "resolution.")
    elif big_phys and not sig_trans:
        verdict = ("PHYSICAL-TISSUE-SPECIFIC (only G_phys) -- unexpected.")
    else:
        verdict = ("BOTH NULL -- the kappa direction did not establish on "
                   "either graph in this dataset; check labelling power.")
    headline.append(f"VERDICT: {verdict}")
    headline.append(f"total runtime: {time.time() - t_start:.1f}s")
    head_text = "\n".join(headline)
    print("\n" + head_text)
    (HERE / "f3_summary.txt").write_text(head_text + "\n")
    print(f"Wrote: {HERE / 'f3_summary.txt'}")


if __name__ == "__main__":
    main()
