"""
LABEL_VALIDATION -- Sensitivity of the kappa-malignancy finding to the
cell-type annotation scheme, for the two marker-derived cohorts:

  * Chen 2021 prostate   (GSE176031)  -> exp/N5b_prostate (headline Cliff d = +0.21)
  * Olalekan 2021 ovarian (GSE147082) -> exp/N5c_ovarian  (headline Cliff d = +0.48)

The other 5 cohorts use original-author labels and are out of scope here.
The reviewer concern: for Chen and Olalekan the malignant / non-malignant
split is derived in-house from marker genes, so the result could in
principle be an artefact of the specific annotation. We test that by
re-deriving the labels three independent ways and recomputing Cliff's
delta each time:

  scheme A "original"            -- reproduces the N5b / N5c marker panels
                                    exactly (sanity check: must recover
                                    the published delta).
  scheme B "alt1_broad"          -- a DIFFERENT marker panel (generic
                                    broad epithelial for prostate, no
                                    prostate-secretory markers; ovarian-
                                    specific Müllerian markers for ovarian).
  scheme C "alt2_cluster"        -- UNSUPERVISED Louvain communities on the
                                    PCA-50 kNN graph, then annotate each
                                    community by marker enrichment. No
                                    per-cell marker argmax at all.

For each scheme we rebuild the 600+600 subsample, HVG-2000, PCA-50,
kNN k=15, Ollivier-Ricci alpha=0.5 on 4000 sampled edges, and report
Cliff's delta (malignant vs immune/T comparator).

If all three schemes agree in sign and rough magnitude, the labels are
robust. If they disagree, the kappa finding is annotation-dependent for
that cohort.

Seed = 20260508 (matches N5b / N5c).
"""
import os
import re
import sys
import json
import time
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, mannwhitneyu
from networkx.algorithms.community import louvain_communities

OUT = Path(__file__).parent
DATA_ROOT = OUT.parent.parent / "data"
CHEN_DIR = DATA_ROOT / "GSE176031_chen"
OLALEKAN_DIR = DATA_ROOT / "GSE147082"
SEED = 20260508

PUBLISHED = {
    "prostate_chen_GSE176031": 0.21,
    "ovarian_olalekan_GSE147082": 0.48,
}


# =====================================================================
# Shared Ollivier-Ricci machinery (mirrors N5b/N5c exactly)
# =====================================================================
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
        # Guard degenerate edges: when two cells are near-identical in PCA
        # space (d_uv -> 0) the curvature 1 - W/d is numerically undefined
        # and can blow up to -1e3. Treat such edges as missing (nan).
        if d_uv > 1e-6:
            out[(u, v)] = float(1.0 - W / d_uv)
        else:
            out[(u, v)] = np.nan
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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    gt = lt = 0
    bsz = 500
    for i in range(0, na, bsz):
        chunk = a[i:i + bsz][:, None]
        gt += int(np.sum(chunk > b[None, :]))
        lt += int(np.sum(chunk < b[None, :]))
    return float(gt - lt) / (na * nb)


# =====================================================================
# Normalisation + marker scoring helpers
# =====================================================================
def normalize_cp10k_log(counts_genes_x_cells_df):
    X = counts_genes_x_cells_df.values.astype(np.float64)
    libsize = X.sum(axis=0)
    libsize = np.where(libsize == 0, 1, libsize)
    Xn = X / libsize[None, :] * 1e4
    Xlog = np.log1p(Xn).astype(np.float32)
    return pd.DataFrame(Xlog, index=counts_genes_x_cells_df.index,
                        columns=counts_genes_x_cells_df.columns)


def panel_mean_score(expr_log, panel):
    """Per-cell mean log-expression over the available markers in a panel.
    Returns a length-n_cells numpy array."""
    avail = [g for g in panel if g in expr_log.index]
    if not avail:
        return np.full(expr_log.shape[1], -np.inf, dtype=np.float64)
    return expr_log.loc[avail].mean(axis=0).values.astype(np.float64)


def zscore_argmax_assign(expr_log, panel_dict, min_top_z=0.5, min_margin=0.25):
    """Prostate-style: z-score each panel across cells, argmax, confidence
    gate on top z and margin over runner-up. Returns per-cell call array."""
    scores = {ct: panel_mean_score(expr_log, p) for ct, p in panel_dict.items()}
    sdf = pd.DataFrame(scores, index=expr_log.columns)
    zs = (sdf - sdf.mean(axis=0)) / (sdf.std(axis=0) + 1e-9)
    call = zs.idxmax(axis=1).values.astype(object)
    top_z = zs.max(axis=1).values
    sorted_z = np.sort(zs.values, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]
    confident = (top_z >= min_top_z) & (margin >= min_margin)
    call = np.where(confident, call, "Unknown")
    return call


def tirosh_score(Xc_cells_x_genes, gene_index, markers, n_ctrl=50, rng=None):
    """Ovarian-style Tirosh signature: mean marker log-expression minus mean
    of expression-binned control genes. Returns per-cell score array."""
    if rng is None:
        rng = np.random.default_rng(0)
    present = [g for g in markers if g in gene_index]
    if not present:
        return np.full(Xc_cells_x_genes.shape[0], np.nan, dtype=np.float32)
    sig_idx = np.array([gene_index[g] for g in present], dtype=int)
    means = Xc_cells_x_genes.mean(axis=0)
    n_bins = 25
    quantiles = np.quantile(means, np.linspace(0, 1, n_bins + 1))
    quantiles[-1] += 1e-9
    bin_id = np.digitize(means, quantiles[1:-1])
    sig_bins = bin_id[sig_idx]
    ctrl_pool = []
    for b in np.unique(sig_bins):
        cands = np.where(bin_id == b)[0]
        cands = np.setdiff1d(cands, sig_idx, assume_unique=False)
        if len(cands) == 0:
            continue
        pick = rng.choice(cands, size=min(n_ctrl, len(cands)), replace=False)
        ctrl_pool.append(pick)
    ctrl_idx = np.concatenate(ctrl_pool) if ctrl_pool else np.array([], dtype=int)
    sig_score = Xc_cells_x_genes[:, sig_idx].mean(axis=1)
    ctrl_score = (Xc_cells_x_genes[:, ctrl_idx].mean(axis=1)
                  if len(ctrl_idx) else np.zeros_like(sig_score))
    return sig_score - ctrl_score


def tirosh_argmax_assign(scores_dict, min_score=0.0, min_margin=0.05):
    """Ovarian-style argmax with score>0 and margin>=0.05 gates."""
    types = list(scores_dict.keys())
    M = np.stack([scores_dict[t] for t in types], axis=1)
    M = np.where(np.isfinite(M), M, -np.inf)
    order = np.argsort(M, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    top_score = M[np.arange(len(M)), top]
    margin = top_score - M[np.arange(len(M)), second]
    call = np.array(["Unknown"] * len(M), dtype=object)
    ok = (top_score > min_score) & (margin >= min_margin)
    call[ok] = np.array(types)[top[ok]]
    return call


# =====================================================================
# Chen GSE176031 loader (mirrors N5b)
# =====================================================================
def parse_sample_meta(fname):
    m = re.search(r"PR(\d+)_([TN])(\d)?", fname)
    if not m:
        return None, None
    return f"PR{m.group(1)}", m.group(2)


def load_chen():
    print("Loading Chen GSE176031 ...")
    files = sorted(f for f in os.listdir(CHEN_DIR) if f.endswith(".txt.gz"))
    dfs, meta_rows = [], []
    for f in files:
        patient, tissue = parse_sample_meta(f)
        if patient is None:
            continue
        df = pd.read_csv(CHEN_DIR / f, sep="\t", index_col=0)
        sample = f.split("_dge")[0]
        df.columns = [f"{sample}__{c}" for c in df.columns]
        dfs.append(df)
        for c in df.columns:
            meta_rows.append({"cell_id": c, "patient": patient,
                              "tissue": tissue, "sample": sample})
    full = pd.concat(dfs, axis=1, join="outer").fillna(0).astype(np.float32)
    meta = pd.DataFrame(meta_rows).set_index("cell_id").loc[full.columns]

    n_genes = (full.values > 0).sum(axis=0)
    n_umi = full.values.sum(axis=0)
    keep = (n_genes >= 200) & (n_umi >= 500)
    print(f"  QC: {keep.sum()}/{len(keep)} cells pass")
    full = full.iloc[:, keep]
    meta = meta.iloc[keep]
    expr_log = normalize_cp10k_log(full)
    print(f"  {expr_log.shape[0]} genes x {expr_log.shape[1]} cells (log-CP10K)")
    return expr_log, meta


# =====================================================================
# Olalekan GSE147082 loader (mirrors N5c)
# =====================================================================
def load_olalekan():
    print("Loading Olalekan GSE147082 ...")
    files = sorted(OLALEKAN_DIR.glob("*.csv"))
    files = [f for f in files if not f.name.endswith(".csv.gz")]
    dfs, pids = [], []
    for f in files:
        pid = f.stem.split("_")[-1]
        df = pd.read_csv(f, index_col=0, low_memory=False)
        dfs.append(df)
        pids.append(pid)
    gene_sets = [set(d.index) for d in dfs]
    common = sorted(set.intersection(*gene_sets))
    parts, cell_pids, barcodes = [], [], []
    for d, pid in zip(dfs, pids):
        sub = d.loc[common]
        parts.append(sub.values.astype(np.float32))
        cell_pids.extend([pid] * sub.shape[1])
        barcodes.extend([f"{pid}__{c}" for c in sub.columns])
    X_counts = np.concatenate(parts, axis=1)
    cell_pids = np.array(cell_pids)
    barcodes = np.array(barcodes)

    libsize = X_counts.sum(axis=0)
    keep = libsize >= 500
    X_counts = X_counts[:, keep]
    cell_pids = cell_pids[keep]
    barcodes = barcodes[keep]
    detected = (X_counts > 0).sum(axis=1)
    keep_g = detected >= 10
    X_counts = X_counts[keep_g]
    gene_names = np.array(common)[keep_g]

    libsize = X_counts.sum(axis=0)
    libsize[libsize == 0] = 1
    X_norm = X_counts / libsize[None, :] * 1e4
    X_log = np.log1p(X_norm).astype(np.float32)

    expr_log = pd.DataFrame(X_log, index=gene_names, columns=barcodes)
    meta = pd.DataFrame({"patient": cell_pids}, index=barcodes)
    print(f"  {expr_log.shape[0]} genes x {expr_log.shape[1]} cells (log-CP10K)")
    return expr_log, meta


# =====================================================================
# Annotation schemes
# =====================================================================
# ---- Chen prostate panels ----
CHEN_ORIGINAL = {
    "luminal": ["KLK3", "KLK2", "AR", "MSMB", "ACPP", "NKX3-1", "KRT8", "KRT18", "EPCAM"],
    "basal":   ["KRT5", "KRT14", "TP63", "KRT15", "DST"],
    "tcell":   ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "PTPRC", "TRAC"],
    "myeloid": ["CD14", "CD68", "LYZ", "C1QA", "C1QB", "AIF1", "CSF1R"],
    "fibro":   ["DCN", "COL1A1", "COL1A2", "LUM", "PDGFRA", "ACTA2"],
    "endo":    ["VWF", "PECAM1", "CDH5", "ENG", "CLDN5"],
}
# Alt1: GENERIC broad epithelial panel (drops prostate-secretory markers
# KLK3/KLK2/MSMB/ACPP/NKX3-1; pools luminal+basal keratins). Comparator
# broad immune T (CD3/PTPRC).
CHEN_ALT1 = {
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT5", "KRT14", "KRT19", "CDH1", "AR"],
    "immune_t":   ["PTPRC", "CD3D", "CD3E", "CD3G", "CD8A", "CD4", "TRAC"],
    "immune_myel": ["PTPRC", "CD68", "LYZ", "CD14", "C1QA", "AIF1"],
    "stromal":    ["DCN", "COL1A1", "COL1A2", "LUM", "PDGFRA", "ACTA2"],
    "endo":       ["PECAM1", "VWF", "CDH5", "ENG"],
}
# Alt2 cluster-annotation markers. Prostate needs a 4-way panel so that
# Louvain communities are annotated into luminal vs basal separately (the
# malignant-of-origin compartment in PRAD is luminal-secretory; pooling
# basal into "epithelial" would unfairly dilute the malignant group).
CHEN_ALT2_MARKERS = {
    "luminal": ["KLK3", "AR", "MSMB", "NKX3-1", "KRT8", "KRT18", "EPCAM"],
    "basal":   ["KRT5", "KRT14", "TP63", "KRT15", "DST"],
    "immune":  ["PTPRC", "CD3D", "CD3G", "CD68", "LYZ", "CD3E"],
    "stromal": ["DCN", "COL1A1", "COL1A2", "ACTA2", "PECAM1", "VWF"],
}

# ---- Olalekan ovarian panels ----
OV_ORIGINAL = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1",
                   "MUC16", "PAX8", "WT1", "FOLR1", "MSLN", "CLDN3", "CLDN4"],
    "Tcell":      ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "TRAC", "TRBC1", "TRBC2"],
    "Bcell":      ["CD79A", "CD79B", "MS4A1", "CD19", "IGKC", "IGHG1"],
    "Macrophage": ["CD68", "CD163", "AIF1", "LYZ", "C1QA", "C1QB", "C1QC", "MARCO"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "PDGFRA", "PDGFRB", "ACTA2", "FAP", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "ENG", "KDR"],
}
# Alt1: Müllerian / ovarian-specific epithelial identity (PAX8, WT1, MUC16/CA125)
# plus pan-epithelial backbone -- deliberately DROPS the generic EPCAM/KRT
# bias to test whether ovarian-lineage markers alone recover the same cells.
OV_ALT1 = {
    "Epithelial_muellerian": ["PAX8", "WT1", "MUC16", "FOLR1", "MSLN", "CLDN3", "CLDN4", "KRT7"],
    "Tcell":      ["PTPRC", "CD3D", "CD3E", "CD3G", "CD8A", "CD4", "TRAC"],
    "Bcell":      ["PTPRC", "CD79A", "MS4A1", "CD19"],
    "Macrophage": ["PTPRC", "CD68", "CD163", "LYZ", "C1QA"],
    "Fibroblast": ["DCN", "COL1A1", "COL1A2", "PDGFRA", "ACTA2", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
}
OV_ALT2_MARKERS = {
    "epi":    ["EPCAM", "KRT8", "KRT18", "PAX8", "WT1", "MUC16", "KRT7"],
    "immune": ["PTPRC", "CD3D", "CD3G", "CD68", "LYZ", "CD3E"],
    "stromal": ["DCN", "COL1A1", "COL1A2", "ACTA2", "PECAM1", "VWF"],
}


def annotate_chen(expr_log, meta, scheme):
    """Return (mal_mask, comp_mask, celltype_calls) for Chen prostate.
    malignant = epithelial/luminal lineage in T (tumor) tissue.
    comparator = T cells."""
    is_T = (meta["tissue"].values == "T")
    if scheme == "original":
        call = zscore_argmax_assign(expr_log, CHEN_ORIGINAL)
        mal = (call == "luminal") & is_T
        comp = (call == "tcell")
        return mal, comp, call
    elif scheme == "alt1_broad":
        call = zscore_argmax_assign(expr_log, CHEN_ALT1)
        mal = (call == "epithelial") & is_T
        comp = (call == "immune_t")
        return mal, comp, call
    elif scheme == "alt2_cluster":
        call = annotate_by_community(expr_log, CHEN_ALT2_MARKERS)
        mal = (call == "luminal") & is_T
        comp = (call == "immune")
        return mal, comp, call
    raise ValueError(scheme)


def annotate_olalekan(expr_log, meta, scheme):
    """Return (mal_mask, comp_mask, calls) for Olalekan ovarian.
    malignant = Epithelial; comparator = T cells."""
    gene_index = {g: i for i, g in enumerate(expr_log.index)}
    Xc = expr_log.values.T  # cells x genes
    if scheme == "original":
        scores = {ct: tirosh_score(Xc, gene_index, m,
                                   rng=np.random.default_rng(SEED))
                  for ct, m in OV_ORIGINAL.items()}
        call = tirosh_argmax_assign(scores)
        mal = (call == "Epithelial")
        comp = (call == "Tcell")
        return mal, comp, call
    elif scheme == "alt1_broad":
        scores = {ct: tirosh_score(Xc, gene_index, m,
                                   rng=np.random.default_rng(SEED))
                  for ct, m in OV_ALT1.items()}
        call = tirosh_argmax_assign(scores)
        mal = (call == "Epithelial_muellerian")
        comp = (call == "Tcell")
        return mal, comp, call
    elif scheme == "alt2_cluster":
        call = annotate_by_community(expr_log, OV_ALT2_MARKERS)
        mal = (call == "epi")
        comp = (call == "immune")
        return mal, comp, call
    raise ValueError(scheme)


def annotate_by_community(expr_log, marker_3way):
    """Unsupervised Louvain community detection on the global PCA-50 kNN
    graph; annotate each community by the argmax of mean (z-scored) marker
    enrichment over a coarse {epi, immune, stromal} panel. Returns per-cell
    call in {epi, immune, stromal, Unknown}."""
    Xc = expr_log.values.T.astype(np.float32)
    var = Xc.var(axis=0)
    hvg = np.argsort(var)[::-1][:2000]
    Xh = Xc[:, hvg]
    Xh = Xh - Xh.mean(axis=0, keepdims=True)
    n_comp = min(50, Xh.shape[0] - 1, Xh.shape[1])
    Xpca = PCA(n_components=n_comp, random_state=SEED).fit_transform(Xh)
    G = build_knn_graph(Xpca, k=15)
    print(f"    [cluster] kNN graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges -> Louvain ...")
    communities = list(louvain_communities(G, seed=SEED))
    print(f"    [cluster] {len(communities)} communities "
          f"(sizes {sorted([len(c) for c in communities], reverse=True)[:12]})")

    # per-community mean marker score (z-scored across cells for comparability)
    panel_scores = {ct: panel_mean_score(expr_log, m)
                    for ct, m in marker_3way.items()}
    sdf = pd.DataFrame(panel_scores, index=expr_log.columns)
    zs = (sdf - sdf.mean(axis=0)) / (sdf.std(axis=0) + 1e-9)

    call = np.array(["Unknown"] * Xc.shape[0], dtype=object)
    cell_idx = np.arange(Xc.shape[0])
    for com in communities:
        members = np.array(sorted(com))
        mean_z = zs.iloc[members].mean(axis=0)
        top = mean_z.idxmax()
        # only assign if the community is clearly enriched (mean z > 0.1)
        if mean_z[top] > 0.1:
            call[members] = top
    return call


# =====================================================================
# kappa contrast runner (matches N5b/N5c subsample + HVG + PCA + kNN + OR)
# =====================================================================
def run_kappa_contrast(expr_log, mal_mask, comp_mask, label_mal, label_comp,
                       n_per=600, n_edges=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    n_mal = int(mal_mask.sum())
    n_comp = int(comp_mask.sum())
    if n_mal < 30 or n_comp < 30:
        return dict(ok=False, n_mal=n_mal, n_comp=n_comp,
                    label_mal=label_mal, label_comp=label_comp)
    idx_mal = rng.choice(np.where(mal_mask)[0],
                         size=min(n_per, n_mal), replace=False)
    idx_comp = rng.choice(np.where(comp_mask)[0],
                          size=min(n_per, n_comp), replace=False)
    sel = np.concatenate([idx_mal, idx_comp])
    n_m = len(idx_mal)

    Xs = expr_log.values[:, sel]  # genes x cells (log-CP10K)
    var = Xs.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xs = Xs[hvg].T.astype(np.float32)  # cells x genes
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=min(50, Xs.shape[0] - 1, Xs.shape[1]),
               random_state=seed).fit_transform(Xs)

    G = build_knn_graph(Xpca, k=15)
    ne = min(n_edges, G.number_of_edges())
    ek = ollivier_ricci_edges(G, alpha=0.5, n_edges=ne, rng=rng)
    per_cell = per_cell_mean_curvature(G, ek)

    kap_mal = np.array([per_cell[i] for i in range(n_m)
                        if not np.isnan(per_cell[i])])
    kap_comp = np.array([per_cell[i] for i in range(n_m, len(sel))
                         if not np.isnan(per_cell[i])])
    if len(kap_mal) < 5 or len(kap_comp) < 5:
        return dict(ok=False, n_mal=n_mal, n_comp=n_comp,
                    label_mal=label_mal, label_comp=label_comp)
    t_stat, p_t = ttest_ind(kap_mal, kap_comp, equal_var=False)
    u_stat, p_u = mannwhitneyu(kap_mal, kap_comp)
    d = cliffs_delta(kap_mal, kap_comp)
    print(f"    {label_mal} (n={len(kap_mal)}) kappa={kap_mal.mean():+.4f}"
          f"+/-{kap_mal.std():.4f}   {label_comp} (n={len(kap_comp)})"
          f" kappa={kap_comp.mean():+.4f}+/-{kap_comp.std():.4f}")
    print(f"    Cliff's delta = {d:+.4f}   Welch p={p_t:.3g}   MWU p={p_u:.3g}")
    return dict(ok=True, label_mal=label_mal, label_comp=label_comp,
                n_mal=len(kap_mal), n_comp=len(kap_comp),
                mean_mal=float(kap_mal.mean()), mean_comp=float(kap_comp.mean()),
                std_mal=float(kap_mal.std()), std_comp=float(kap_comp.std()),
                delta=float(d), sign="+" if d > 0 else ("-"
                          if d < 0 else "0"),
                p_welch=float(p_t), p_mwu=float(p_u))


def run_kappa_on_embedding(Xpca_all, mal_mask, comp_mask, label_mal, label_comp,
                           n_per=600, n_edges=4000, seed=SEED):
    """Like run_kappa_contrast but on a FIXED precomputed global PCA
    embedding (no per-subsample HVG). Used for the decomposition where we
    hold geometry fixed and vary only the labels. Standardises PC coords
    (divide by per-axis std) for numerical stability."""
    rng = np.random.default_rng(seed)
    n_mal = int(mal_mask.sum())
    n_comp = int(comp_mask.sum())
    if n_mal < 30 or n_comp < 30:
        return dict(ok=False, n_mal=n_mal, n_comp=n_comp,
                    label_mal=label_mal, label_comp=label_comp)
    idx_mal = rng.choice(np.where(mal_mask)[0],
                         size=min(n_per, n_mal), replace=False)
    idx_comp = rng.choice(np.where(comp_mask)[0],
                          size=min(n_per, n_comp), replace=False)
    sel = np.concatenate([idx_mal, idx_comp])
    n_m = len(idx_mal)
    Xs = Xpca_all[sel].astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    sd = Xs.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Xs = Xs / sd
    G = build_knn_graph(Xs, k=15)
    ne = min(n_edges, G.number_of_edges())
    ek = ollivier_ricci_edges(G, alpha=0.5, n_edges=ne, rng=rng)
    per_cell = per_cell_mean_curvature(G, ek)
    kap_mal = np.array([per_cell[i] for i in range(n_m)
                        if not np.isnan(per_cell[i])])
    kap_comp = np.array([per_cell[i] for i in range(n_m, len(sel))
                         if not np.isnan(per_cell[i])])
    if len(kap_mal) < 5 or len(kap_comp) < 5:
        return dict(ok=False, n_mal=n_mal, n_comp=n_comp,
                    label_mal=label_mal, label_comp=label_comp)
    t_stat, p_t = ttest_ind(kap_mal, kap_comp, equal_var=False)
    u_stat, p_u = mannwhitneyu(kap_mal, kap_comp)
    d = cliffs_delta(kap_mal, kap_comp)
    print(f"    {label_mal} (n={len(kap_mal)}) kappa={kap_mal.mean():+.4f}"
          f"+/-{kap_mal.std():.4f}   {label_comp} (n={len(kap_comp)})"
          f" kappa={kap_comp.mean():+.4f}+/-{kap_comp.std():.4f}")
    print(f"    Cliff's delta = {d:+.4f}   Welch p={p_t:.3g}   MWU p={p_u:.3g}")
    return dict(ok=True, label_mal=label_mal, label_comp=label_comp,
                n_mal=len(kap_mal), n_comp=len(kap_comp),
                mean_mal=float(kap_mal.mean()), mean_comp=float(kap_comp.mean()),
                std_mal=float(kap_mal.std()), std_comp=float(kap_comp.std()),
                delta=float(d), sign="+" if d > 0 else ("-"
                          if d < 0 else "0"),
                p_welch=float(p_t), p_mwu=float(p_u))


# =====================================================================
# Driver
# =====================================================================
SCHEMES = ["original", "alt1_broad", "alt2_cluster"]


def run_cohort(name, loader, annotator, published_delta):
    print("\n" + "#" * 70)
    print(f"# {name}   (published delta = {published_delta:+.2f})")
    print("#" * 70)
    expr_log, meta = loader()
    scheme_results = {}
    mal_masks = {}
    for scheme in SCHEMES:
        print(f"\n--- {name}: scheme = {scheme} ---")
        mal_mask, comp_mask, calls = annotator(expr_log, meta, scheme)
        mal_masks[scheme] = mal_mask
        n_mal = int(mal_mask.sum())
        n_comp = int(comp_mask.sum())
        # call distribution
        uniq, cnts = np.unique(calls[calls != "Unknown"], return_counts=True)
        dist = dict(zip(uniq.tolist(), cnts.tolist()))
        print(f"  call distribution (non-Unknown): {dist}")
        print(f"  malignant mask = {n_mal}   comparator mask = {n_comp}")
        t0 = time.time()
        r = run_kappa_contrast(expr_log, mal_mask, comp_mask,
                               label_mal=f"mal_{scheme}",
                               label_comp=f"comp_{scheme}",
                               n_per=600, n_edges=4000, seed=SEED)
        r["wallclock_s"] = round(time.time() - t0, 1)
        r["n_mal_pool"] = n_mal
        r["n_comp_pool"] = n_comp
        r["call_distribution"] = dist
        scheme_results[scheme] = r

    # ---- decomposition: fix the comparator to the ORIGINAL-scheme immune/T
    # cells, and re-test each scheme's MALIGNANT label against it. This
    # isolates whether any delta change is driven by the malignant label or
    # by the comparator also being re-derived. Geometry is held FIXED via a
    # single global PCA-50 embedding (standardised) so that only the label
    # masks vary -- this avoids subsample-HVG degeneracies and gives a clean
    # "label-only" sensitivity read.
    print(f"\n--- {name}: fixed-comparator decomposition "
          f"(malignant_S vs original immune/T, fixed global embedding) ---")
    _, comp_fixed, _ = annotator(expr_log, meta, "original")
    Xc_all = expr_log.values.T.astype(np.float32)
    var_all = Xc_all.var(axis=0)
    hvg_all = np.argsort(var_all)[::-1][:2000]
    Xh_all = Xc_all[:, hvg_all]
    Xh_all = Xh_all - Xh_all.mean(axis=0, keepdims=True)
    Xpca_all = PCA(n_components=min(50, Xh_all.shape[0] - 1, Xh_all.shape[1]),
                   random_state=SEED).fit_transform(Xh_all)
    decomp = {}
    for scheme in ["original", "alt1_broad", "alt2_cluster"]:
        t0 = time.time()
        r = run_kappa_on_embedding(Xpca_all, mal_masks[scheme], comp_fixed,
                                   label_mal=f"mal_{scheme}",
                                   label_comp="comp_original_fixed",
                                   n_per=600, n_edges=4000, seed=SEED)
        r["wallclock_s"] = round(time.time() - t0, 1)
        decomp[scheme] = r

    deltas = [scheme_results[s]["delta"] for s in SCHEMES
              if scheme_results[s].get("ok")]
    signs = [("+" if d > 0 else "-") for d in deltas]
    sign_agree = len(set(signs)) == 1 and signs and signs[0] == "+"
    summary = {
        "published_delta": published_delta,
        "schemes": scheme_results,
        "decomposition_fixed_comparator": decomp,
        "recovered_deltas": deltas,
        "sign_agreement_positive": bool(sign_agree),
        "delta_min": float(min(deltas)) if deltas else None,
        "delta_max": float(max(deltas)) if deltas else None,
        "delta_span": float(max(deltas) - min(deltas)) if deltas else None,
    }
    return summary


def main():
    t0 = time.time()
    print("=" * 70)
    print("LABEL_VALIDATION -- annotation-scheme sensitivity for the two")
    print("marker-derived cohorts (Chen prostate, Olalekan ovarian).")
    print("=" * 70)
    print(f"seed = {SEED}")

    out = {}
    out["prostate_chen_GSE176031"] = run_cohort(
        "Chen prostate (GSE176031)", load_chen, annotate_chen,
        PUBLISHED["prostate_chen_GSE176031"])
    out["ovarian_olalekan_GSE147082"] = run_cohort(
        "Olalekan ovarian (GSE147082)", load_olalekan, annotate_olalekan,
        PUBLISHED["ovarian_olalekan_GSE147082"])

    # ---- verdict ----
    # Classify each scheme's delta as positive / null / negative using both
    # sign and significance (a |delta| with p>0.05 is null, not a reversal).
    verdicts = {}
    for cohort, res in out.items():
        ds = res["recovered_deltas"]
        if not ds:
            verdicts[cohort] = "INCONCLUSIVE (a scheme failed)"
            res["verdict"] = verdicts[cohort]
            continue
        classes = []
        for s in SCHEMES:
            r = res["schemes"][s]
            d = r["delta"]
            p = r.get("p_welch", 1.0)
            if d < -0.05 and p < 0.05:
                classes.append("negative")
            elif d > 0.05 and p < 0.05:
                classes.append("positive")
            else:
                classes.append("null")
        n_pos = classes.count("positive")
        n_neg = classes.count("negative")
        n_null = classes.count("null")
        span = res["delta_span"]

        # fold in the fixed-comparator decomposition: if every scheme is
        # positive there (comparator controlled), the direction is preserved
        # even when a self-consistent scheme is null.
        dec = res.get("decomposition_fixed_comparator", {})
        dec_all_pos = all(dec[s].get("delta", 0) > 0
                          for s in SCHEMES if dec.get(s, {}).get("ok"))

        if n_neg > 0:
            v = (f"ANNOTATION-DEPENDENT (sign reversal: a scheme gives a"
                 f" significant negative delta). deltas "
                 f"{[round(x, 3) for x in ds]}")
        elif n_null == 0:
            v = (f"ROBUST: all 3 schemes significantly positive "
                 f"(delta in [{res['delta_min']:+.3f}, "
                 f"{res['delta_max']:+.3f}], span {span:.3f})")
        else:
            # some null, none negative -> direction preserved, weakest
            # scheme(s) lose significance
            null_schemes = [SCHEMES[i] for i, c in enumerate(classes)
                            if c == "null"]
            extra = ("; fixed-comparator decomposition confirms all schemes"
                     " positive when comparator is controlled"
                     if dec_all_pos else "")
            v = (f"ROBUST IN DIRECTION: {n_pos}/{len(SCHEMES)} schemes"
                 f" significantly positive, {n_null} null ({', '.join(null_schemes)}"
                 f"), 0 reversed. deltas "
                 f"{[round(x, 3) for x in ds]}{extra}")
        verdicts[cohort] = v
        res["verdict"] = v

    out["overall_verdict"] = verdicts
    out["seed"] = SEED
    out["wallclock_total_s"] = round(time.time() - t0, 1)

    # ---- save JSON ----
    jpath = OUT / "results.json"
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {jpath}")

    # ---- print + save text summary ----
    lines = []
    lines.append("LABEL_VALIDATION -- annotation-scheme sensitivity summary")
    lines.append("=" * 64)
    lines.append(f"Seed: {SEED}")
    lines.append("")
    for cohort in ["prostate_chen_GSE176031", "ovarian_olalekan_GSE147082"]:
        res = out[cohort]
        lines.append(f"{cohort}  (published delta = {res['published_delta']:+.2f})")
        lines.append("-" * 64)
        for scheme in SCHEMES:
            r = res["schemes"][scheme]
            if not r.get("ok"):
                lines.append(f"  {scheme:16s}  SKIPPED (mal={r.get('n_mal')}"
                             f" comp={r.get('n_comp')})")
                continue
            lines.append(
                f"  {scheme:16s}  delta={r['delta']:+.3f}  "
                f"({r['label_mal']} kappa {r['mean_mal']:+.4f} vs "
                f"{r['label_comp']} {r['mean_comp']:+.4f};  "
                f"n={r['n_mal']}/{r['n_comp']}; p_welch={r['p_welch']:.2g})")
        ds = res["recovered_deltas"]
        lines.append(f"  -> recovered deltas: "
                     f"{[round(x,3) for x in ds]}  "
                     f"sign_agree={res['sign_agreement_positive']}  "
                     f"span={res['delta_span']:.3f}")
        # decomposition: each scheme's malignant label vs FIXED original
        # immune/T comparator (isolates malignant-label effect)
        lines.append(f"  fixed-comparator decomposition "
                     f"(mal_S vs original immune/T, fixed embedding):")
        for scheme in ["original", "alt1_broad", "alt2_cluster"]:
            dr = res["decomposition_fixed_comparator"].get(scheme, {})
            if dr.get("ok"):
                lines.append(
                    f"    {scheme:16s}  delta={dr['delta']:+.3f}  "
                    f"({dr['label_mal']} kappa {dr['mean_mal']:+.4f} vs "
                    f"{dr['label_comp']} {dr['mean_comp']:+.4f};  "
                    f"n={dr['n_mal']}/{dr['n_comp']}; "
                    f"p_welch={dr['p_welch']:.2g})")
        lines.append(f"  VERDICT: {res['verdict']}")
        lines.append("")
    lines.append(f"Total wallclock: {out['wallclock_total_s']}s")
    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)
    tpath = OUT / "label_validation_summary.txt"
    with open(tpath, "w") as fh:
        fh.write(summary_txt + "\n")
    print(f"Saved {tpath}")


if __name__ == "__main__":
    main()
