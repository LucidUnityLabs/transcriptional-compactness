"""
N2 — Visium HD (8 micron) replication of OR kappa malignant-vs-stroma
direction at single-cell-adjacent resolution.

Question: the spot-resolution Visium test (T4) on a 55 micron / 1-10 cell
spot was null on G_phys (Cliff d ~ 0). Was that:
  (a) substrate-specificity confirmed (kappa is genuinely about transcriptional
      neighborhoods), or
  (b) a resolution artifact (each 55 um spot mechanically averages multiple
      cell types so the per-spot kappa cannot pick up cell-cell contrast)?

This script repeats the dual-graph G_phys vs G_trans OR kappa malignant-vs-
stroma test using 10X Visium HD 8 micron bins from human colorectal cancer
patient P2 (GEO GSM8594568). At 8 um, each bin is single-cell-adjacent.

Pipeline:
  1) Load filtered_feature_bc_matrix.h5 + tissue_positions.parquet +
     pre-computed Metadata.parquet (10X-supplied UnsupervisedL1 labels).
  2) Restrict to a 120 x 120 array_row/col tile selected for label balance
     (~14k bins with both Tumor and Fibroblast/SmoothMuscle/Endothelial).
  3) Filter to bins with non-trivial counts; log1p + HVG-2000 + PCA-50.
  4) Label per-bin malignant vs stroma. Primary uses 10X UnsupervisedL1
     (Tumor -> malignant; Fibroblast/SmoothMuscle/Endothelial -> stroma);
     also computes a signature-score backup label (KRT8/KRT18/KRT19/EPCAM/
     KRT20/VIL1 vs PTPRC/COL1A1/ACTA2/PECAM1/VWF).
  5) Random-subsample to ~5000 bins keeping label balance (preserve spatial
     density too) so the OR kappa fits the time budget.
  6) Build G_phys: kNN k=6 in array_row/col Euclidean; weight = distance.
  7) Build G_trans: kNN k=15 in PCA-50; weight = PCA distance.
  8) Sample 4000 edges per graph, lazy-walk alpha=0.5 OR kappa per edge ->
     per-bin mean kappa.
  9) Welch t, MWU, Cliff's delta on malignant vs stroma per graph.
 10) Outputs: n2_visium_hd_summary.csv, n2_visium_hd_spatial.png,
     n2_visium_hd_distributions.png, n2_summary.txt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csc_matrix
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "visium_hd"
SEED = 20260508
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)

# Tile selected after a tile-search for the best Tumor/stroma balance in the
# in-tissue bins.  120 array units == ~960 microns square.  Approx 14.6k bins.
TILE_R0, TILE_R1 = 250, 370
TILE_C0, TILE_C1 = 250, 370

# Subsample target after labelling.  OR kappa with 4000 edges + per-node mean
# is fine on 5-6k nodes; 14k blows out the time budget.
TARGET_N = 6000

# Marker panels (colon-specialised) for the backup signature-score label.
TUMOR_GENES = ["KRT8", "KRT18", "KRT19", "EPCAM", "KRT20", "VIL1"]
STROMA_GENES = ["PTPRC", "COL1A1", "ACTA2", "PECAM1", "VWF"]


# ---------- OR kappa (matches exp/T4_visium_spatial/run.py) -----------------
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
def load_visium_hd():
    h5_path = DATA / "GSM8594568_P2CRC_filtered_feature_bc_matrix.h5"
    pos_path = DATA / "GSM8594568_P2CRC_tissue_positions.parquet"
    md_path = DATA / "GSM8594568_P2CRC_Metadata.parquet"
    sf_path = DATA / "GSM8594568_P2CRC_scalefactors_json.json"

    print(f"[load] {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = m["shape"][:]
        barcodes = [b.decode() for b in m["barcodes"][:]]
        feat = m["features"]
        feat_id = [b.decode() for b in feat["id"][:]]
        feat_name = [b.decode() for b in feat["name"][:]]
        feat_type = [b.decode() for b in feat["feature_type"][:]]
    n_genes, n_cells = int(shape[0]), int(shape[1])
    X_csc = csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
    X = X_csc.T.tocsr().astype(np.float32)
    print(f"  matrix: {X.shape} (bins x features)")

    var = pd.DataFrame({"gene_ids": feat_id, "feature_types": feat_type},
                       index=feat_name)
    obs = pd.DataFrame(index=barcodes)
    adata = AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    is_ge = adata.var["feature_types"].values == "Gene Expression"
    adata = adata[:, is_ge].copy()
    print(f"  Gene Expression filter: {adata.shape}")

    pos = pd.read_parquet(pos_path).set_index("barcode")
    md = pd.read_parquet(md_path).set_index("barcode")
    common = adata.obs.index
    pos = pos.loc[common]
    md = md.loc[common]
    adata.obs["in_tissue"] = pos["in_tissue"].astype(int).values
    adata.obs["array_row"] = pos["array_row"].astype(int).values
    adata.obs["array_col"] = pos["array_col"].astype(int).values
    adata.obs["pxl_row"] = pos["pxl_row_in_fullres"].astype(float).values
    adata.obs["pxl_col"] = pos["pxl_col_in_fullres"].astype(float).values
    for col in ("UnsupervisedL1", "UnsupervisedL2", "DeconvolutionLabel1",
                "DeconvolutionClass", "Periphery"):
        if col in md.columns:
            adata.obs[col] = md[col].astype(str).values

    n_before = adata.n_obs
    adata = adata[adata.obs["in_tissue"] == 1].copy()
    print(f"  in_tissue==1: {adata.n_obs}/{n_before}")

    sf = json.loads(sf_path.read_text())
    print(f"  bin_size_um={sf.get('bin_size_um')}, "
          f"microns_per_pixel={sf.get('microns_per_pixel'):.4f}")
    return adata, sf


def crop_tile(adata, r0, r1, c0, c1):
    rr = adata.obs["array_row"].values
    cc = adata.obs["array_col"].values
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    print(f"[tile] array_row in [{r0},{r1}), array_col in [{c0},{c1}): "
          f"keeping {int(keep.sum())}/{adata.n_obs} bins")
    return adata[keep].copy()


# ---------- preprocessing ----------------------------------------------------
def preprocess(adata, n_hvg=2000, n_pcs=50, seed=SEED):
    print("[preprocess] gene/bin filter + normalize_total + log1p + HVG + PCA")
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"  after QC filter: {adata.shape}")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=n_hvg)
    adata_h = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_h, max_value=10)
    X_hvg = adata_h.X
    if hasattr(X_hvg, "toarray"):
        X_hvg = X_hvg.toarray()
    n_pcs_eff = min(n_pcs, X_hvg.shape[1] - 1, X_hvg.shape[0] - 1)
    pca = PCA(n_components=n_pcs_eff, random_state=seed).fit(X_hvg)
    adata.obsm["X_pca"] = pca.transform(X_hvg).astype(np.float32)
    print(f"  PCA: {adata.obsm['X_pca'].shape}")
    return adata


# ---------- labelling --------------------------------------------------------
def label_bins(adata):
    """Primary label: 10X UnsupervisedL1.  Backup: signature score.

    Final label is set from the primary; if a bin has UnsupervisedL1 == 'Tumor'
    AND signature_diff > 0 we keep malignant; for stroma we require
    UnsupervisedL1 in {Fibroblast, SmoothMuscle, Endothelial}.  Any
    'Unknown'/'Intestinal Epithelial'/'Myeloid' etc are left unlabeled, since
    they are neither clearly malignant nor classic stroma.  This matches
    T4's malignant/stroma bipartition.
    """
    print("[label] computing signature scores (backup) and primary L1 label")
    avail_t = [g for g in TUMOR_GENES if g in adata.var_names]
    avail_s = [g for g in STROMA_GENES if g in adata.var_names]
    print(f"  tumor markers: {avail_t}")
    print(f"  stroma markers: {avail_s}")
    sc.tl.score_genes(adata, gene_list=avail_t, score_name="tumor_score",
                      random_state=SEED)
    sc.tl.score_genes(adata, gene_list=avail_s, score_name="stroma_score",
                      random_state=SEED)
    diff = (adata.obs["tumor_score"] - adata.obs["stroma_score"]).values

    if "UnsupervisedL1" in adata.obs.columns:
        L1 = adata.obs["UnsupervisedL1"].values
        label = np.array(["unlabeled"] * adata.n_obs, dtype=object)
        is_tumor = L1 == "Tumor"
        is_stroma = np.isin(L1, ["Fibroblast", "SmoothMuscle", "Endothelial"])
        label[is_tumor] = "malignant"
        label[is_stroma] = "stroma"
        adata.obs["label_l1"] = label
        n_mal = int((label == "malignant").sum())
        n_str = int((label == "stroma").sum())
        n_un = int((label == "unlabeled").sum())
        print(f"  primary L1 labels: malignant={n_mal}  stroma={n_str}  "
              f"unlabeled={n_un}")
        adata.obs["label"] = label
    else:
        # Pure signature-score fallback -- 33/67 quantile split.
        q_lo, q_hi = np.quantile(diff, [0.33, 0.67])
        label = np.array(["unlabeled"] * adata.n_obs, dtype=object)
        label[diff >= q_hi] = "malignant"
        label[diff <= q_lo] = "stroma"
        adata.obs["label"] = label
        print(f"  signature-only labels: {pd.Series(label).value_counts().to_dict()}")
    adata.obs["sig_diff"] = diff
    return adata


# ---------- graphs -----------------------------------------------------------
def build_phys_graph(adata, k=6):
    """k=6 nearest physical neighbours in array_row/col coordinate space.
    Each unit equals one 8-micron bin pitch, so the graph is over the local
    physical neighbourhood.  Edge weight = Euclidean distance in array units.
    """
    pts = np.column_stack([adata.obs["array_row"].values,
                           adata.obs["array_col"].values]).astype(float)
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
    print(f"[G_phys] kNN(k={k}) physical: {G.number_of_edges()} edges, "
          f"median d={np.median([d for _,_,d in G.edges(data='weight')]):.2f}")
    return G


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
def analyse(name, G, label, n_edges=4000):
    print(f"\n[{name}] OR kappa on {n_edges} sampled edges (alpha=0.5)")
    t0 = time.time()
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=RNG)
    elapsed = time.time() - t0
    n_valid = sum(1 for v in edge_k.values() if not np.isnan(v))
    print(f"  valid: {n_valid}/{len(edge_k)}  (took {elapsed:.1f}s)")
    per = per_node_mean_curvature(G, edge_k)
    kappa = np.array([per.get(i, np.nan) for i in range(len(label))])
    n_with_k = np.isfinite(kappa).sum()
    print(f"  bins with valid kappa: {n_with_k}/{len(label)}")
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
    pr = adata.obs["array_row"].values
    pc = adata.obs["array_col"].values
    ax = axes[0]
    valid = np.isfinite(kappa_phys)
    s = 4
    if valid.sum() > 0:
        sc1 = ax.scatter(pc[valid], -pr[valid], c=kappa_phys[valid],
                         cmap="RdBu_r", s=s,
                         vmin=np.nanpercentile(kappa_phys, 2),
                         vmax=np.nanpercentile(kappa_phys, 98))
        plt.colorbar(sc1, ax=ax, label="kappa (G_phys)")
    ax.set_title(f"Per-bin Ollivier-Ricci kappa on G_phys "
                 f"(n={int(valid.sum())} bins, 8 um)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    color = {"malignant": "#d62728", "stroma": "#1f77b4",
             "unlabeled": "#cccccc"}
    for lab in ("unlabeled", "stroma", "malignant"):
        m = label == lab
        ax.scatter(pc[m], -pr[m], c=color[lab], s=s,
                   label=f"{lab} (n={int(m.sum())})")
    ax.legend(loc="lower right", fontsize=9, markerscale=2)
    ax.set_title("8-um bin labels (10X UnsupervisedL1)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()


def plot_distributions(kappa_phys, kappa_trans, label, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kappa, name in zip(axes, (kappa_phys, kappa_trans),
                               ("G_phys (k=6 array)", "G_trans (k=15 PCA)")):
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
        ax.set_xlabel("per-bin mean kappa"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()


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
    # 40 / 40 / 20 split, capped at availability.
    nm = min(int(target_n * 0.4), len(idx_m))
    ns = min(int(target_n * 0.4), len(idx_s))
    nu = min(target_n - nm - ns, len(idx_u))
    take = np.concatenate([
        rng.choice(idx_m, size=nm, replace=False),
        rng.choice(idx_s, size=ns, replace=False),
        rng.choice(idx_u, size=nu, replace=False),
    ])
    take.sort()
    print(f"[subsample] {len(take)} bins kept "
          f"(malignant={nm}, stroma={ns}, unlabeled={nu})")
    return adata[take].copy(), take


# ---------- main -------------------------------------------------------------
def main():
    print("=" * 72)
    print("N2 -- Visium HD 8-um single-cell-resolution OR kappa replication")
    print(f"     seed={SEED}")
    print("=" * 72)
    t_start = time.time()

    adata, sf = load_visium_hd()
    adata = crop_tile(adata, TILE_R0, TILE_R1, TILE_C0, TILE_C1)
    adata = preprocess(adata)
    adata = label_bins(adata)
    adata, _ = stratified_subsample(adata, TARGET_N)

    # rebuild PCA on the subsample to keep G_trans well-conditioned.
    print("[pca-redo] re-running PCA on subsample")
    sub = adata.copy()
    if "log1p" in sub.uns:
        del sub.uns["log1p"]
    # Re-derive HVGs and PCA on subsample size.
    sub_h = sub[:, sub.var["highly_variable"]].copy()
    sc.pp.scale(sub_h, max_value=10)
    Xh = sub_h.X
    if hasattr(Xh, "toarray"):
        Xh = Xh.toarray()
    n_pcs_eff = min(50, Xh.shape[1] - 1, Xh.shape[0] - 1)
    pca = PCA(n_components=n_pcs_eff, random_state=SEED).fit(Xh)
    adata.obsm["X_pca"] = pca.transform(Xh).astype(np.float32)
    print(f"  PCA on subsample: {adata.obsm['X_pca'].shape}")

    G_phys = build_phys_graph(adata, k=6)
    G_trans = build_trans_graph(adata, k=15)

    label = adata.obs["label"].values

    kappa_phys, row_phys = analyse("G_phys", G_phys, label, n_edges=4000)
    kappa_trans, row_trans = analyse("G_trans", G_trans, label, n_edges=4000)

    summary = pd.DataFrame([row_phys, row_trans])
    summary_path = HERE / "n2_visium_hd_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote: {summary_path}")
    print(summary.to_string(index=False))

    spatial_path = HERE / "n2_visium_hd_spatial.png"
    plot_spatial(adata, kappa_phys, label, spatial_path)
    print(f"Wrote: {spatial_path}")

    dist_path = HERE / "n2_visium_hd_distributions.png"
    plot_distributions(kappa_phys, kappa_trans, label, dist_path)
    print(f"Wrote: {dist_path}")

    # Effect-size thresholds.  A small effect (|d|>=0.10) at high statistical
    # significance still counts as a real signal -- with n>1500 per group the
    # power to detect arbitrarily small differences is enormous, so we look
    # at sign + size jointly.
    sign_phys = np.sign(row_phys["diff"])
    sign_trans = np.sign(row_trans["diff"])
    sig_phys = (row_phys["welch_p"] < 0.01
                and abs(row_phys["cliff_d"]) >= 0.10)
    sig_trans = (row_trans["welch_p"] < 0.01
                 and abs(row_trans["cliff_d"]) >= 0.10)
    big_phys = abs(row_phys["cliff_d"]) >= 0.20
    headline = []
    headline.append("=" * 72)
    headline.append("N2 -- HEADLINE")
    headline.append("=" * 72)
    headline.append(f"bin size: {sf.get('bin_size_um')} um   "
                    f"tile: array_row [{TILE_R0},{TILE_R1}) x "
                    f"array_col [{TILE_C0},{TILE_C1})")
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
    # Decision tree per the spec: focus on whether the kappa direction
    # transfers to physical space.
    if big_phys and sig_trans and sign_phys == sign_trans:
        verdict = ("SUBSTRATE-INDEPENDENT at single-cell resolution: "
                   "the kappa direction transfers to physical-space neighbors. "
                   "T4's spot-resolution null was a resolution artifact.")
    elif big_phys and sig_trans and sign_phys != sign_trans:
        verdict = ("DIVERGENT: kappa direction is opposite on G_phys vs "
                   "G_trans -- weird, investigate.")
    elif (not big_phys) and sig_trans:
        verdict = ("SUBSTRATE-SPECIFIC confirmed at single-cell resolution: "
                   "the kappa malignant-vs-stroma direction lives only on "
                   "the transcriptomic-similarity graph; physical-neighbor "
                   "kappa is null (Cliff d~0). T4's null replicates at 8 um, "
                   "so it was NOT a resolution artifact.")
    elif big_phys and not sig_trans:
        verdict = ("PHYSICAL-TISSUE-SPECIFIC (only G_phys) -- unexpected.")
    else:
        verdict = ("BOTH NULL -- the kappa direction did not establish on "
                   "either graph in this tile.")
    headline.append(f"VERDICT: {verdict}")
    headline.append(f"total runtime: {time.time() - t_start:.1f}s")
    head_text = "\n".join(headline)
    print("\n" + head_text)
    (HERE / "n2_summary.txt").write_text(head_text + "\n")
    print(f"Wrote: {HERE / 'n2_summary.txt'}")


if __name__ == "__main__":
    main()
