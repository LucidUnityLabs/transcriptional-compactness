"""
CLASSIFIER_AUC -- Operational test of the "kappa distinguishes malignant
from non-malignant" phenotype claim.

Reviewer question: "Can per-cell Ollivier-Ricci curvature kappa ALONE
classify malignant vs non-malignant cells?" This is the operational test
of the per-cell kappa phenotype claim in RESEARCH_NOTE.md Sec 3.1.

Pipeline:
  (A) Per-cohort (7 solid-tumour cohorts):
      For each cohort, replicate the E1 kappa pipeline exactly:
        HVG-2000 -> mean-center -> PCA-50 -> kNN(k=15) ->
        Ollivier-Ricci alpha=0.5 lazy-walk on 4000 sampled edges ->
        per-cell mean kappa over incident edges.
      Contrast per cohort (matches RESEARCH_NOTE.md Sec 3.1):
        Tirosh  melanoma  : Mal vs T cells
        Darmanis GBM      : Neoplastic vs Immune
        Puram  HNSCC      : Mal vs Fibroblast
        Li     CRC        : Epithelial vs T cells
        Peng   PDAC       : Mal vs Ductal-non-mal
        Chen   prostate   : Mal vs Immune (T)
        Olalekan HGSOC    : Epithelial vs T cells
      Sample 600 + 600 cells per cohort (or all available if fewer).
      Single-feature logistic regression on kappa, stratified 80/20.
      10 random seeds -> AUC-ROC, AUC-PR, accuracy, F1 (mean +/- std).
      Baselines: random feature (sanity), UMI alone (depth confound).
      Multi-feature: kappa + UMI + n_genes + pct_mito.

  (B) Leave-one-cohort-out (LOCO):
      Pool per-cohort features (kappa, UMI, n_genes, pct_mito, labels)
      across all 7 cohorts. Train on 6 cohorts, test on held-out 1.
      Per-cohort held-out AUC for kappa-only and multi-feature.

  (C) Output:
      results.json           -- nested dict of all metrics
      per_cohort_auc.csv     -- per-cohort within-cohort AUC table
      leave_one_out_auc.csv  -- LOCO held-out AUC table

Method choices (with rationale):
  - Logistic regression: linear, interpretable, no hyperparameter tuning,
    appropriate for single-feature evaluation. class_weight='balanced'
    so any minor class-imbalance does not inflate accuracy.
  - Stratified train/test split: preserves class balance.
  - 10 seeds: bounds Monte-Carlo variance from the split.
  - AUC-ROC primary (pre-registered in the prompt); AUC-PR secondary
    (more honest under class imbalance).
  - kappa is computed ONCE per cohort on the full 600+600 sample
    (the standard pipeline), then reused for every seed / baseline /
    LOCO fold. This isolates classifier variance from kappa variance.
  - UMI = per-cell total linear expression (TPM for log-TPM datasets,
    raw counts for count datasets, FPKM for Li). Used as a depth proxy.
  - pct_mito = fraction of per-cell expression on MT-* genes (where
    available; 0 if no MT genes in panel).

Memory: cohorts are loaded and processed one at a time; only per-cell
features (kappa, UMI, n_genes, pct_mito, label) are retained in memory.
"""

import gc
import json
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import networkx as nx
import ot
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
SEED = 20260507
RNG = np.random.default_rng(SEED)
N_PER = 600            # per-arm cell cap (matches existing pipeline)
N_EDGES = 4000         # sampled edges for OR (matches existing pipeline)
K_NN = 15              # kNN neighbours (matches existing pipeline)
ALPHA = 0.5            # lazy-walk mass at source (matches existing pipeline)
N_CLASSIFIER_SEEDS = 10
TEST_SIZE = 0.20


# ==============================================================
# Ollivier-Ricci primitives (verbatim from exp/E1_within_patient/run.py)
# ==============================================================
def build_knn_graph(X, k=K_NN):
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


def ollivier_ricci_edges(G, alpha=ALPHA, n_edges=N_EDGES, rng=None):
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
    return {n: float(np.mean(vs)) if vs else np.nan
            for n, vs in by_cell.items()}


# ==============================================================
# QC feature helper
# ==============================================================
def compute_qc_features(X_gene_x_cell, gene_names, is_log_tpm=False,
                        is_log1p=False):
    """
    Return per-cell (libsize, n_genes, pct_mito).

    X_gene_x_cell: genes x cells matrix.
    gene_names:    iterable of gene symbols aligned with rows.
    is_log_tpm:    if True, values are log2(TPM/10 + 1); invert to TPM.
    is_log1p:      if True, values are log1p(counts); invert to counts.
    """
    X = np.asarray(X_gene_x_cell, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=object)

    if is_log_tpm:
        X_lin = (np.power(2.0, X) - 1.0) * 10.0
        X_lin = np.clip(X_lin, 0, None)
    elif is_log1p:
        X_lin = np.expm1(X)
        X_lin = np.clip(X_lin, 0, None)
    else:
        X_lin = X

    libsize = X_lin.sum(axis=0)
    n_genes = (X_lin > 0).sum(axis=0)

    # Mitochondrial genes (human: MT- prefix; fall back to MITO set)
    upper = np.char.upper(gene_names.astype(str))
    mito_mask = np.fromiter(
        (g.startswith("MT-") or g.startswith("MT.") for g in upper),
        dtype=bool, count=len(upper),
    )
    if mito_mask.any():
        mito_counts = X_lin[mito_mask].sum(axis=0)
        safe = np.where(libsize > 0, libsize, 1.0)
        pct_mito = mito_counts / safe * 100.0
    else:
        pct_mito = np.zeros(X.shape[1], dtype=np.float64)

    return libsize, n_genes, pct_mito


# ==============================================================
# Stratified sampler (matches exp/E1_within_patient/stratified_sample)
# ==============================================================
def simple_sample(mask_a, mask_b, n_per, rng):
    """Random balanced subsample: take min(|a|, |b|, n_per) from each
    group so neither dominates the kNN graph. Matches the per-cohort
    pipeline (E1 stratified_sample aims at balanced n_per_group; E4 / N5a
    effectively balance when one arm is the bottleneck)."""
    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]
    take = min(n_per, len(idx_a), len(idx_b))
    if take == 0:
        raise ValueError(f"empty arm: |a|={len(idx_a)} |b|={len(idx_b)}")
    sel_a = rng.choice(idx_a, size=take, replace=False)
    sel_b = rng.choice(idx_b, size=take, replace=False)
    sel = np.concatenate([sel_a, sel_b])
    labels = np.concatenate([
        np.ones(take, dtype=int),
        np.zeros(take, dtype=int),
    ])
    return sel, labels


# ==============================================================
# Per-cohort driver: given selected cells' expression matrix in linear
# space (genes x cells), HVG-2000 + PCA-50, kNN, OR, per-cell kappa.
# Also returns QC features in the original cell order.
# ==============================================================
def embed_and_curvature(X_log_genes_x_sel, rng):
    """HVG-2000 by variance, mean-center, PCA-50, kNN k=15, OR alpha=0.5
    on 4000 edges, per-cell mean kappa.

    X_log_genes_x_sel: genes x cells matrix in *log space*, matching each
    dataset's canonical preprocessing (log2(TPM/10+1) for Tirosh/Puram,
    log1p(counts) for Darmanis/PDAC, log1p(CP10K) for Chen/Olalekan,
    log1p(FPKM) for Li). No further transform is applied -- this matches
    exp/E1_within_patient/run.py and the per-cohort N5/E3/E4 scripts.
    """
    Xlog = X_log_genes_x_sel.astype(np.float32)
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xs = Xlog[hvg].T.astype(np.float32)  # cells x genes
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    n_comp = min(50, Xs.shape[0] - 1, Xs.shape[1])
    Xpca = PCA(n_components=n_comp, random_state=SEED).fit_transform(Xs)

    G = build_knn_graph(Xpca, k=K_NN)
    n_edges = min(N_EDGES, G.number_of_edges())
    edge_k = ollivier_ricci_edges(G, alpha=ALPHA, n_edges=n_edges, rng=rng)
    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa = np.array([per_cell.get(i, np.nan) for i in range(Xpca.shape[0])],
                     dtype=np.float64)
    return kappa, Xpca


# ==============================================================
# Cohort loaders
# Each returns a dict with:
#   name, label_a (malignant label str), label_b (non-mal label str),
#   kappa (per-cell, length n_cells),
#   libsize, n_genes, pct_mito (per-cell, length n_cells),
#   y (per-cell 1/0 malignant flag, length n_cells)
# The kappa pipeline is identical across cohorts (HVG-2000/PCA-50/kNN15/
# OR alpha=0.5/4000 edges); only loading + label assignment differs.
# ==============================================================

def _to_linear(X, mode):
    if mode == "log_tpm":
        return (np.power(2.0, X) - 1.0) * 10.0
    if mode == "log1p":
        return np.expm1(X)
    return X  # already linear (counts / fpkm)


def _cohort_common(name, X_log_sel, X_lin_sel, gene_names_sel,
                   labels, label_a, label_b):
    """Run embed_and_curvature on the selected cells (log matrix) and
    compute QC features in linear space. Returns the cohort dict."""
    rng = np.random.default_rng(SEED)
    kappa, _Xpca = embed_and_curvature(X_log_sel, rng)
    libsize, n_genes, pct_mito = compute_qc_features(
        X_lin_sel, gene_names_sel,
    )
    return {
        "name": name,
        "label_a": label_a,
        "label_b": label_b,
        "kappa": kappa,
        "libsize": libsize,
        "n_genes": n_genes.astype(np.float64),
        "pct_mito": pct_mito,
        "y": labels.astype(int),
    }


def load_tirosh():
    """Tirosh 2016 melanoma (GSE72056). log2(TPM/10+1). Mal vs T cells.
    Header rows: 0 cell_id, 1 patient, 2 malignant (0/1/2), 3 nonmal_type.
    Log matrix for embedding = input log2(TPM/10+1); linear matrix for QC
    features = TPM recovered by 2^x - 1)*10."""
    print("[Tirosh] loading ...")
    EXPR = DATA / "GSE72056_melanoma.txt"
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    gene_names = expr.iloc[:, 0].astype(str).values
    X_log_full = expr.iloc[:, 1:].values.astype(np.float32)  # log2(TPM/10+1)

    mask_mal = (malignant == 2)
    mask_T = (malignant == 1) & (nonmal_type == 1)
    print(f"  total: mal={mask_mal.sum()}  T={mask_T.sum()}")

    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_mal, mask_T, N_PER, rng)
    X_log = X_log_full[:, sel]
    X_lin = _to_linear(X_log, "log_tpm")
    return _cohort_common("Tirosh_melanoma", X_log, X_lin, gene_names,
                          labels, "Malignant", "T_cells")


def load_darmanis():
    """Darmanis 2017 GBM (GSE84465). Raw counts. Neoplastic vs Immune.
    Log matrix for embedding = log1p(counts); linear = raw counts."""
    print("[Darmanis] loading ...")
    EXPR = DATA / "GSE84465_GBM.csv"
    META = DATA / "GSE84465_meta.txt"
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

    df_expr = pd.read_csv(EXPR, sep=r"\s+", header=0, index_col=0, engine="c")
    df_expr.columns = [c.strip('"') for c in df_expr.columns]
    meta_idx = meta.set_index("cell_id").loc[df_expr.columns]
    cell_type = meta_idx["cell type"].values
    gene_names = df_expr.index.astype(str).values
    X_counts = df_expr.values.astype(np.float32)  # genes x cells

    mask_neo = (cell_type == "Neoplastic")
    mask_imm = (cell_type == "Immune cell")
    print(f"  total: neo={mask_neo.sum()}  immune={mask_imm.sum()}")

    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_neo, mask_imm, N_PER, rng)
    X_lin = X_counts[:, sel]
    X_log = np.log1p(X_lin)
    return _cohort_common("Darmanis_GBM", X_log, X_lin, gene_names,
                          labels, "Neoplastic", "Immune")


def load_puram():
    """Puram 2017 HNSCC (GSE103322). log2(TPM/10+1). Mal vs Fibroblast.
    Header: row3 cancer flag, row4 non-cancer flag, row5 non-cancer subtype.
    Log matrix = input log2(TPM/10+1); linear = TPM."""
    print("[Puram] loading ...")
    EXPR = DATA / "GSE103322_HNSCC.txt"
    header = pd.read_csv(EXPR, sep="\t", nrows=6, header=None, low_memory=False)
    cancer = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    noncancer = pd.to_numeric(header.iloc[4, 1:], errors="coerce").values
    nonmal_type = header.iloc[5, 1:].astype(str).values
    nonmal_type = np.array([t.lstrip("-").strip() for t in nonmal_type])
    expr = pd.read_csv(EXPR, sep="\t", skiprows=6, header=None, low_memory=False)
    gene_names = expr.iloc[:, 0].astype(str).values
    X_log_full = expr.iloc[:, 1:].values.astype(np.float32)

    mask_mal = (cancer == 1)
    counts = pd.Series(nonmal_type[noncancer == 1]).value_counts()
    largest = counts.index[0]
    mask_fib = (noncancer == 1) & (nonmal_type == largest)
    print(f"  total: mal={mask_mal.sum()}  {largest}={mask_fib.sum()}")

    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_mal, mask_fib, N_PER, rng)
    X_log = X_log_full[:, sel]
    X_lin = _to_linear(X_log, "log_tpm")
    return _cohort_common("Puram_HNSCC", X_log, X_lin, gene_names,
                          labels, "Malignant", largest)


def load_li_crc():
    """Li 2017 CRC (GSE81861). FPKM. Column headers encode cell type as
    <barcode>__<cellType>__<color>. Mal=Epithelial vs T cells.
    Log matrix = log1p(FPKM); linear = FPKM."""
    print("[Li_CRC] loading ...")
    csv_path = DATA / "GSE81861_CRC_tumor_FPKM.csv"
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    cols = df.columns.tolist()
    cell_types = np.array(
        [c.split("__")[1] if "__" in c else "NA" for c in cols],
        dtype=object,
    )
    keep = cell_types != "NA"
    df = df.loc[:, keep]
    cell_types = cell_types[keep]
    gene_names = df.index.astype(str).values
    X_fpkm = df.values.astype(np.float32)

    mask_epi = (cell_types == "Epithelial")
    mask_t = (cell_types == "Tcell")
    print(f"  total: epi={mask_epi.sum()}  t={mask_t.sum()}")

    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_epi, mask_t, N_PER, rng)
    X_lin = X_fpkm[:, sel]
    X_log = np.log1p(X_lin)
    return _cohort_common("Li_CRC", X_log, X_lin, gene_names,
                          labels, "Epithelial", "Tcell")


def load_pdac():
    """Peng/Moncada 2019 PDAC (GSE111672). Raw inDrop counts.
    Mal = Cancer clone A+B; non-mal = non-malignant ductal (closest match).
    Log matrix = log1p(counts); linear = counts."""
    import gzip
    print("[PDAC] loading ...")
    PDAC_A = DATA / "GSE111672_PDAC-A-indrop-filtered-expMat.txt.gz"
    PDAC_B = DATA / "GSE111672_PDAC-B-indrop-filtered-expMat.txt.gz"

    parts = []
    labels_all = []
    gene_lists = []
    for path in [PDAC_A, PDAC_B]:
        with gzip.open(path, "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
        cell_labels_file = np.array([h.strip() for h in header[1:]],
                                    dtype=object)
        df = pd.read_csv(path, sep="\t", compression="gzip", header=0,
                         low_memory=False)
        gene_lists.append(df.iloc[:, 0].astype(str).values)
        parts.append(df.iloc[:, 1:].values.astype(np.float32))
        labels_all.append(cell_labels_file)

    if np.array_equal(gene_lists[0], gene_lists[1]):
        gene_names = gene_lists[0]
        X_all = np.concatenate(parts, axis=1)
    else:
        common = pd.Index(gene_lists[0]).intersection(pd.Index(gene_lists[1]))
        gene_names = common.astype(str).values
        ia = pd.Index(gene_lists[0]).get_indexer(common)
        ib = pd.Index(gene_lists[1]).get_indexer(common)
        X_all = np.concatenate([parts[0][ia], parts[1][ib]], axis=1)

    cell_labels = np.concatenate(labels_all)
    mal_set = {"Cancer clone A", "Cancer clone B"}
    ductal_set = {
        "Ductal - terminal ductal like",
        "Ductal - CRISP3 high/centroacinar like",
        "Ductal - MHC Class II",
        "Ductal - APOL1 high/hypoxic",
    }
    mask_mal = np.array([l in mal_set for l in cell_labels])
    mask_duct = np.array([l in ductal_set for l in cell_labels])
    print(f"  total: mal={mask_mal.sum()}  ductal={mask_duct.sum()}")

    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_mal, mask_duct, N_PER, rng)
    X_lin = X_all[:, sel]
    X_log = np.log1p(X_lin)
    return _cohort_common("PDAC_peng", X_log, X_lin, gene_names,
                          labels, "Malignant", "Ductal-non-mal")


# --- Chen prostate marker panels (copied from exp/N5b_prostate/run.py) ---
_CHEN_MARKERS = {
    "luminal":  ["KLK3", "KLK2", "AR", "MSMB", "ACPP", "NKX3-1",
                 "KRT8", "KRT18", "EPCAM"],
    "basal":    ["KRT5", "KRT14", "TP63", "KRT15", "DST"],
    "tcell":    ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4",
                 "PTPRC", "TRAC"],
    "myeloid":  ["CD14", "CD68", "LYZ", "C1QA", "C1QB", "AIF1", "CSF1R"],
    "fibro":    ["DCN", "COL1A1", "COL1A2", "LUM", "PDGFRA", "ACTA2"],
    "endo":     ["VWF", "PECAM1", "CDH5", "ENG", "CLDN5"],
}


def _chen_assign_celltypes(expr_log_df, meta_df):
    """Score cells by marker panels; argmax -> call. Copied from N5b."""
    scores = {}
    for ct, markers in _CHEN_MARKERS.items():
        avail = [g for g in markers if g in expr_log_df.index]
        if not avail:
            scores[ct] = np.full(expr_log_df.shape[1], -np.inf)
            continue
        scores[ct] = expr_log_df.loc[avail].mean(axis=0).values
    score_df = pd.DataFrame(scores, index=expr_log_df.columns)
    zs = (score_df - score_df.mean(axis=0)) / (score_df.std(axis=0) + 1e-9)
    call = zs.idxmax(axis=1).values
    top_z = zs.max(axis=1).values
    sorted_z = np.sort(zs.values, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]
    confident = (top_z >= 0.5) & (margin >= 0.25)
    call = np.where(confident, call, "Unknown")
    out = meta_df.copy()
    out["celltype"] = call
    return out


def load_chen_prostate():
    """Chen 2021 PRAD (GSE176031). Raw UMI counts. Mal = luminal in Tissue
    T; non-mal = T cells (any tissue)."""
    import os
    print("[Chen prostate] loading ...")
    DATA_DIR = DATA / "GSE176031_chen"
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt.gz"))

    dfs = []
    meta_rows = []
    for f in files:
        m = re.search(r"PR(\d+)_([TN])(\d)?", f)
        if not m:
            continue
        patient = f"PR{m.group(1)}"
        tissue = m.group(2)
        path = DATA_DIR / f
        df = pd.read_csv(path, sep="\t", index_col=0)
        sample_tag = f.split("_dge")[0]
        df.columns = [f"{sample_tag}__{c}" for c in df.columns]
        dfs.append(df)
        for c in df.columns:
            meta_rows.append({"cell_id": c, "patient": patient,
                              "tissue": tissue})
    print("  concatenating samples (gene union, fillna=0) ...")
    full = pd.concat(dfs, axis=1, join="outer").fillna(0).astype(np.float32)
    meta = pd.DataFrame(meta_rows).set_index("cell_id").loc[full.columns]
    gene_names = full.index.astype(str).values
    X_counts = full.values  # genes x cells
    print(f"  combined: {X_counts.shape[0]} genes x {X_counts.shape[1]} cells")

    # QC: >=200 genes, >=500 UMI
    n_genes_all = (X_counts > 0).sum(axis=0)
    n_umi_all = X_counts.sum(axis=0)
    keep = (n_genes_all >= 200) & (n_umi_all >= 500)
    X_counts = X_counts[:, keep]
    meta = meta.iloc[keep]
    print(f"  after QC: {X_counts.shape[1]} cells")

    # CP10K + log1p for cell-type assignment
    libsize = X_counts.sum(axis=0)
    libsize = np.where(libsize == 0, 1, libsize)
    Xn = X_counts / libsize[None, :] * 1e4
    Xlog_norm = np.log1p(Xn).astype(np.float32)
    expr_log_df = pd.DataFrame(Xlog_norm, index=full.index, columns=meta.index)
    meta = _chen_assign_celltypes(expr_log_df, meta)

    is_lum = (meta["celltype"].values == "luminal")
    is_tissue_t = (meta["tissue"].values == "T")
    is_tcell = (meta["celltype"].values == "tcell")
    mask_mal = is_lum & is_tissue_t
    mask_immune = is_tcell
    print(f"  mal(luminal,T)={mask_mal.sum()}  immune(tcell)={mask_immune.sum()}")

    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_mal, mask_immune, N_PER, rng)
    X_lin = X_counts[:, sel]
    # CP10K + log1p for embedding (matches N5b pipeline)
    libsize_sel = X_lin.sum(axis=0)
    libsize_sel = np.where(libsize_sel == 0, 1, libsize_sel)
    X_log = np.log1p(X_lin / libsize_sel[None, :] * 1e4).astype(np.float32)
    return _cohort_common("Chen_prostate", X_log, X_lin, gene_names,
                          labels, "Malignant", "Immune")


# --- Olalekan HGSOC marker panels (copied from exp/N5c_ovarian/run.py) ---
_HGSOC_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1",
                   "MUC16", "PAX8", "WT1", "FOLR1", "MSLN", "CLDN3", "CLDN4"],
    "Tcell": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4",
              "TRAC", "TRBC1", "TRBC2"],
    "Bcell": ["CD79A", "CD79B", "MS4A1", "CD19", "IGKC", "IGHG1"],
    "Macrophage": ["CD68", "CD163", "AIF1", "LYZ", "C1QA", "C1QB",
                   "C1QC", "MARCO"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "PDGFRA",
                   "PDGFRB", "ACTA2", "FAP", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "ENG", "KDR"],
}


def _hgsc_score_marker(X_log_cp10k_cells_x_genes, gene_index, marker_genes,
                       n_ctrl=50, rng=None):
    """Tirosh-style signature score (marker mean - binned control mean)."""
    if rng is None:
        rng = np.random.default_rng(0)
    present = [g for g in marker_genes if g in gene_index]
    if not present:
        return np.full(X_log_cp10k_cells_x_genes.shape[0], np.nan,
                       dtype=np.float32)
    sig_idx = np.array([gene_index[g] for g in present], dtype=int)
    means = X_log_cp10k_cells_x_genes.mean(axis=0)
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
        n_pick = min(n_ctrl, len(cands))
        pick = rng.choice(cands, size=n_pick, replace=False)
        ctrl_pool.append(pick)
    ctrl_idx = (np.concatenate(ctrl_pool) if ctrl_pool
                else np.array([], dtype=int))
    sig_score = X_log_cp10k_cells_x_genes[:, sig_idx].mean(axis=1)
    if len(ctrl_idx):
        ctrl_score = X_log_cp10k_cells_x_genes[:, ctrl_idx].mean(axis=1)
    else:
        ctrl_score = np.zeros_like(sig_score)
    return sig_score - ctrl_score


def _hgsc_assign(scores):
    types = list(scores.keys())
    M = np.stack([scores[t] for t in types], axis=1)
    M = np.where(np.isfinite(M), M, -np.inf)
    order = np.argsort(M, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    top_score = M[np.arange(len(M)), top]
    second_score = M[np.arange(len(M)), second]
    margin = top_score - second_score
    call = np.array(["Unknown"] * len(M), dtype=object)
    for i in range(len(M)):
        if top_score[i] > 0 and margin[i] >= 0.05:
            call[i] = types[top[i]]
    return call


def load_olalekan_hgsoc():
    """Olalekan 2021 HGSOC (GSE147082). Raw counts. Mal = Epithelial
    (marker-derived); non-mal = T cells (marker-derived)."""
    print("[Olalekan HGSOC] loading ...")
    OLA = DATA / "GSE147082"
    files = sorted(OLA.glob("*.csv"))
    files = [f for f in files if not f.name.endswith(".csv.gz")]
    dfs, pids = [], []
    for f in files:
        pid = f.stem.split("_")[-1]
        df = pd.read_csv(f, index_col=0, low_memory=False)
        dfs.append(df)
        pids.append(pid)
    gene_sets = [set(d.index) for d in dfs]
    common = sorted(set.intersection(*gene_sets))
    print(f"  gene intersection: {len(common)}")
    parts, cell_pids = [], []
    for d, pid in zip(dfs, pids):
        sub = d.loc[common]
        parts.append(sub.values.astype(np.float32))
        cell_pids.extend([pid] * sub.shape[1])
    X_counts = np.concatenate(parts, axis=1)
    print(f"  pooled: {X_counts.shape[0]} genes x {X_counts.shape[1]} cells")

    libsize = X_counts.sum(axis=0)
    keep = libsize >= 500
    X_counts = X_counts[:, keep]
    cell_pids = np.array(cell_pids)[keep]
    detected = (X_counts > 0).sum(axis=1)
    keep_g = detected >= 10
    X_counts = X_counts[keep_g]
    gene_names = np.array(common)[keep_g]
    print(f"  after QC: {X_counts.shape[1]} cells, {X_counts.shape[0]} genes")

    libsize = X_counts.sum(axis=0)
    libsize[libsize == 0] = 1
    Xn = X_counts / libsize[None, :] * 1e4
    X_log = np.log1p(Xn).astype(np.float32)

    Xc = X_log.T  # cells x genes
    gene_index = {g: i for i, g in enumerate(gene_names)}
    rng = np.random.default_rng(SEED)
    scores = {ct: _hgsc_score_marker(Xc, gene_index, ml, n_ctrl=50, rng=rng)
              for ct, ml in _HGSOC_MARKERS.items()}
    cell_types = _hgsc_assign(scores)
    counts = pd.Series(cell_types).value_counts()
    print("  cell-type counts:")
    for k, v in counts.items():
        print(f"    {k:14s} {v}")

    mask_epi = (cell_types == "Epithelial")
    mask_t = (cell_types == "Tcell")
    rng = np.random.default_rng(SEED)
    sel, labels = simple_sample(mask_epi, mask_t, N_PER, rng)
    X_lin = X_counts[:, sel]
    # log1p(CP10K) for embedding (matches N5c pipeline)
    libsize_sel = X_lin.sum(axis=0)
    libsize_sel = np.where(libsize_sel == 0, 1, libsize_sel)
    X_log = np.log1p(X_lin / libsize_sel[None, :] * 1e4).astype(np.float32)
    return _cohort_common("Olalekan_HGSOC", X_log, X_lin, gene_names,
                          labels, "Epithelial", "Tcell")


COHORT_LOADERS = [
    ("Tirosh_melanoma", load_tirosh),
    ("Darmanis_GBM",    load_darmanis),
    ("Puram_HNSCC",     load_puram),
    ("Li_CRC",          load_li_crc),
    ("PDAC_peng",       load_pdac),
    ("Chen_prostate",   load_chen_prostate),
    ("Olalekan_HGSOC",  load_olalekan_hgsoc),
]


# ==============================================================
# Classifier helpers
# ==============================================================
def evaluate_single_feature(X_feat, y, seed, test_size=TEST_SIZE):
    """Train logistic regression on 80% of (X_feat, y), test on 20%.
    X_feat: (n, k) feature matrix. y: (n,) 0/1 labels.
    Returns dict(auc_roc, auc_pr, accuracy, f1)."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y, test_size=test_size, random_state=seed, stratify=y,
    )
    # scale features (matters when comparing kappa to UMI magnitudes)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="lbfgs"
    )
    clf.fit(X_tr_s, y_tr)
    p = clf.predict_proba(X_te_s)[:, 1]
    yhat = (p >= 0.5).astype(int)
    # guard against single-class test slices
    if len(np.unique(y_te)) < 2:
        auc_roc = float("nan")
    else:
        auc_roc = float(roc_auc_score(y_te, p))
    auc_pr = float(average_precision_score(y_te, p))
    return {
        "auc_roc": auc_roc,
        "auc_pr":  auc_pr,
        "accuracy": float(accuracy_score(y_te, yhat)),
        "f1":       float(f1_score(y_te, yhat, zero_division=0)),
    }


def aggregate_runs(X_feat, y, seeds):
    """Run evaluate_single_feature across `seeds`, return mean/std dict."""
    runs = [evaluate_single_feature(X_feat, y, s) for s in seeds]
    out = {}
    for key in runs[0]:
        vals = np.array([r[key] for r in runs if not np.isnan(r[key])])
        out[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        out[f"{key}_std"]  = float(np.std(vals))  if len(vals) else float("nan")
    return out


def _safe_log(x):
    return np.log1p(np.clip(x, 0, None))


# ==============================================================
# Main driver
# ==============================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("CLASSIFIER_AUC -- per-cell kappa as malignant / non-malignant")
    print("                  classifier: 7-cohort within-cohort + LOCO")
    print(f"  seed={SEED}  N_PER={N_PER}+{N_PER}  kNN={K_NN}  alpha={ALPHA}")
    print(f"  n_edges={N_EDGES}  classifier_seeds={N_CLASSIFIER_SEEDS}")
    print("=" * 72)

    seeds = list(range(SEED, SEED + N_CLASSIFIER_SEEDS))

    # ---------- (A) Within-cohort ----------
    cohorts = []
    per_cohort_rows = []
    for name, loader in COHORT_LOADERS:
        print("\n" + "-" * 72)
        print(f"COHORT: {name}")
        print("-" * 72)
        t_c = time.time()
        try:
            cd = loader()
        except Exception as e:
            print(f"  !! LOADER FAILED: {e}")
            continue
        # Drop cells with NaN kappa (insufficient incident edges)
        valid = ~np.isnan(cd["kappa"])
        if not valid.all():
            print(f"  dropping {(~valid).sum()} NaN-kappa cells")
        for k in ("kappa", "libsize", "n_genes", "pct_mito", "y"):
            cd[k] = cd[k][valid]
        n = len(cd["y"])
        n_mal = int(cd["y"].sum())
        print(f"  cells (post-NaN): {n}  mal={n_mal}  non={n - n_mal}  "
              f"({cd['label_a']} vs {cd['label_b']})")

        # kappa stats (sanity vs existing pipeline)
        a = cd["kappa"][cd["y"] == 1]
        b = cd["kappa"][cd["y"] == 0]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) and len(b):
            tt, pt = ttest_ind(a, b, equal_var=False)
            uu, pu = mannwhitneyu(a, b)
            d = (np.sum(a[:, None] > b[None, :]) -
                 np.sum(a[:, None] < b[None, :])) / (len(a) * len(b))
            print(f"  kappa check: mal={a.mean():+.4f} non={b.mean():+.4f}  "
                  f"Welch p={pt:.3g}  MWU p={pu:.3g}  Cliff d={d:+.3f}")

        # Feature matrices
        X_kappa = cd["kappa"].reshape(-1, 1)
        X_umi = _safe_log(cd["libsize"]).reshape(-1, 1)
        X_ng = _safe_log(cd["n_genes"]).reshape(-1, 1)
        X_pm = cd["pct_mito"].reshape(-1, 1)
        X_rand = RNG.standard_normal((n, 1))
        X_multi = np.column_stack([
            cd["kappa"],
            _safe_log(cd["libsize"]),
            _safe_log(cd["n_genes"]),
            cd["pct_mito"],
        ])

        y = cd["y"]
        res_kappa = aggregate_runs(X_kappa, y, seeds)
        res_umi   = aggregate_runs(X_umi,   y, seeds)
        res_ng    = aggregate_runs(X_ng,    y, seeds)
        res_pm    = aggregate_runs(X_pm,    y, seeds)
        res_rand  = aggregate_runs(X_rand,  y, seeds)
        res_multi = aggregate_runs(X_multi, y, seeds)

        def fmt(r):
            return (f"AUC-ROC={r['auc_roc_mean']:.3f}+-{r['auc_roc_std']:.3f}"
                    f"  AUC-PR={r['auc_pr_mean']:.3f}+-{r['auc_pr_std']:.3f}"
                    f"  acc={r['accuracy_mean']:.3f}  F1={r['f1_mean']:.3f}")

        print(f"  kappa-only : {fmt(res_kappa)}")
        print(f"  UMI-only   : {fmt(res_umi)}")
        print(f"  n_genes    : {fmt(res_ng)}")
        print(f"  pct_mito   : {fmt(res_pm)}")
        print(f"  random     : {fmt(res_rand)}  (sanity, ~0.5)")
        print(f"  multi-feat : {fmt(res_multi)}")

        per_cohort_rows.append({
            "cohort": name,
            "label_a": cd["label_a"],
            "label_b": cd["label_b"],
            "n_cells": n,
            "n_mal": n_mal,
            "n_non": n - n_mal,
            "cliff_delta_kappa": float(d) if len(a) and len(b) else float("nan"),
            "kappa_auc_roc_mean":  res_kappa["auc_roc_mean"],
            "kappa_auc_roc_std":   res_kappa["auc_roc_std"],
            "kappa_auc_pr_mean":   res_kappa["auc_pr_mean"],
            "kappa_auc_pr_std":    res_kappa["auc_pr_std"],
            "kappa_acc_mean":      res_kappa["accuracy_mean"],
            "kappa_f1_mean":       res_kappa["f1_mean"],
            "umi_auc_roc_mean":    res_umi["auc_roc_mean"],
            "umi_auc_roc_std":     res_umi["auc_roc_std"],
            "umi_auc_pr_mean":     res_umi["auc_pr_mean"],
            "n_genes_auc_roc_mean": res_ng["auc_roc_mean"],
            "pct_mito_auc_roc_mean": res_pm["auc_roc_mean"],
            "random_auc_roc_mean": res_rand["auc_roc_mean"],
            "random_auc_roc_std":  res_rand["auc_roc_std"],
            "multi_auc_roc_mean":  res_multi["auc_roc_mean"],
            "multi_auc_roc_std":   res_multi["auc_roc_std"],
            "multi_auc_pr_mean":   res_multi["auc_pr_mean"],
            "multi_acc_mean":      res_multi["accuracy_mean"],
            "multi_f1_mean":       res_multi["f1_mean"],
        })

        # Retain pooled features for LOCO
        cohorts.append({
            "name":      name,
            "kappa":     cd["kappa"].copy(),
            "libsize":   cd["libsize"].copy(),
            "n_genes":   cd["n_genes"].copy(),
            "pct_mito":  cd["pct_mito"].copy(),
            "y":         cd["y"].copy(),
        })
        del cd
        gc.collect()
        print(f"  ({time.time() - t_c:.1f}s)")

    per_cohort_df = pd.DataFrame(per_cohort_rows)
    per_cohort_csv = HERE / "per_cohort_auc.csv"
    per_cohort_df.to_csv(per_cohort_csv, index=False)
    print(f"\nWrote {per_cohort_csv}")

    # ---------- (B) Leave-one-cohort-out ----------
    print("\n" + "=" * 72)
    print("LEAVE-ONE-COHORT-OUT")
    print("=" * 72)
    loco_rows = []
    for held in cohorts:
        train_idx = [c for c in cohorts if c["name"] != held["name"]]
        X_tr_kappa = np.concatenate([c["kappa"] for c in train_idx]).reshape(-1, 1)
        X_te_kappa = held["kappa"].reshape(-1, 1)
        X_tr_multi = np.column_stack([
            np.concatenate([c["kappa"]    for c in train_idx]),
            _safe_log(np.concatenate([c["libsize"] for c in train_idx])),
            _safe_log(np.concatenate([c["n_genes"] for c in train_idx])),
            np.concatenate([c["pct_mito"] for c in train_idx]),
        ])
        X_te_multi = np.column_stack([
            held["kappa"],
            _safe_log(held["libsize"]),
            _safe_log(held["n_genes"]),
            held["pct_mito"],
        ])
        X_tr_umi = _safe_log(
            np.concatenate([c["libsize"] for c in train_idx])
        ).reshape(-1, 1)
        X_te_umi = _safe_log(held["libsize"]).reshape(-1, 1)
        y_tr = np.concatenate([c["y"] for c in train_idx])
        y_te = held["y"]

        def loco_eval(X_tr, X_te, label):
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="lbfgs"
            )
            clf.fit(X_tr_s, y_tr)
            p = clf.predict_proba(X_te_s)[:, 1]
            yhat = (p >= 0.5).astype(int)
            if len(np.unique(y_te)) < 2:
                auc_roc = float("nan")
            else:
                auc_roc = float(roc_auc_score(y_te, p))
            return {
                "label": label,
                "auc_roc": auc_roc,
                "auc_pr":  float(average_precision_score(y_te, p)),
                "accuracy": float(accuracy_score(y_te, yhat)),
                "f1":       float(f1_score(y_te, yhat, zero_division=0)),
            }

        r_kappa = loco_eval(X_tr_kappa, X_te_kappa, "kappa_only")
        r_umi   = loco_eval(X_tr_umi,   X_te_umi,   "umi_only")
        r_multi = loco_eval(X_tr_multi, X_te_multi, "multi")
        print(f"  held-out={held['name']:18s}  kappa AUC={r_kappa['auc_roc']:.3f}"
              f"  UMI AUC={r_umi['auc_roc']:.3f}"
              f"  multi AUC={r_multi['auc_roc']:.3f}"
              f"  (n_test={len(y_te)})")

        loco_rows.append({
            "held_out_cohort": held["name"],
            "n_train": int(len(y_tr)),
            "n_test":  int(len(y_te)),
            "kappa_auc_roc": r_kappa["auc_roc"],
            "kappa_auc_pr":  r_kappa["auc_pr"],
            "kappa_acc":     r_kappa["accuracy"],
            "kappa_f1":      r_kappa["f1"],
            "umi_auc_roc":   r_umi["auc_roc"],
            "umi_auc_pr":    r_umi["auc_pr"],
            "multi_auc_roc": r_multi["auc_roc"],
            "multi_auc_pr":  r_multi["auc_pr"],
            "multi_acc":     r_multi["accuracy"],
            "multi_f1":      r_multi["f1"],
        })

    loco_df = pd.DataFrame(loco_rows)
    loco_csv = HERE / "leave_one_out_auc.csv"
    loco_df.to_csv(loco_csv, index=False)
    print(f"\nWrote {loco_csv}")

    # ---------- summary ----------
    n_cohorts = len(per_cohort_rows)
    kappa_aucs = per_cohort_df["kappa_auc_roc_mean"].values
    multi_aucs = per_cohort_df["multi_auc_roc_mean"].values
    rand_aucs = per_cohort_df["random_auc_roc_mean"].values
    umi_aucs = per_cohort_df["umi_auc_roc_mean"].values
    kappa_loco = loco_df["kappa_auc_roc"].values
    multi_loco = loco_df["multi_auc_roc"].values
    umi_loco = loco_df["umi_auc_roc"].values

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Within-cohort kappa-only AUC-ROC:  "
          f"mean={np.nanmean(kappa_aucs):.3f}  "
          f"min={np.nanmin(kappa_aucs):.3f}  max={np.nanmax(kappa_aucs):.3f}  "
          f"#>0.7={int(np.sum(kappa_aucs > 0.7))}/{n_cohorts}")
    print(f"Within-cohort random-feature AUC:  "
          f"mean={np.nanmean(rand_aucs):.3f}  (expect ~0.5)")
    print(f"Within-cohort UMI-only AUC:        "
          f"mean={np.nanmean(umi_aucs):.3f}")
    print(f"Within-cohort multi-feature AUC:   "
          f"mean={np.nanmean(multi_aucs):.3f}")
    print(f"LOCO kappa-only AUC-ROC:           "
          f"mean={np.nanmean(kappa_loco):.3f}  "
          f"min={np.nanmin(kappa_loco):.3f}  max={np.nanmax(kappa_loco):.3f}  "
          f"#>0.7={int(np.sum(kappa_loco > 0.7))}/{n_cohorts}")
    print(f"LOCO UMI-only AUC:                 "
          f"mean={np.nanmean(umi_loco):.3f}")
    print(f"LOCO multi-feature AUC:            "
          f"mean={np.nanmean(multi_loco):.3f}")

    # Operational verdict per pre-registered threshold
    frac_within_gt07 = np.nanmean(kappa_aucs > 0.7)
    frac_loco_gt07   = np.nanmean(kappa_loco > 0.7)
    if frac_within_gt07 >= 0.7:
        verdict_within = "OPERATIONALLY VALIDATED (>=70% cohorts AUC>0.7)"
    elif np.nanmean(kappa_aucs) > 0.65:
        verdict_within = "MODERATE (mean AUC 0.65-0.70 or mixed)"
    else:
        verdict_within = "WEAK (most cohorts AUC<=0.65)"
    if frac_loco_gt07 >= 0.7:
        verdict_loco = "GENERALIZES (>=70% held-out AUC>0.7)"
    elif np.nanmean(kappa_loco) > 0.65:
        verdict_loco = "PARTIAL (mean held-out AUC 0.65-0.70)"
    else:
        verdict_loco = "DOES NOT GENERALIZE"

    print()
    print(f"  Within-cohort kappa-only: {verdict_within}")
    print(f"  LOCO kappa-only:          {verdict_loco}")

    results = {
        "config": {
            "seed": SEED,
            "n_per_arm": N_PER,
            "k_nn": K_NN,
            "alpha": ALPHA,
            "n_edges": N_EDGES,
            "n_classifier_seeds": N_CLASSIFIER_SEEDS,
            "test_size": TEST_SIZE,
        },
        "within_cohort": per_cohort_df.to_dict(orient="records"),
        "loco":          loco_df.to_dict(orient="records"),
        "summary": {
            "kappa_within_auc_mean":  float(np.nanmean(kappa_aucs)),
            "kappa_within_auc_min":   float(np.nanmin(kappa_aucs)),
            "kappa_within_auc_max":   float(np.nanmax(kappa_aucs)),
            "kappa_within_n_gt_0p7":  int(np.sum(kappa_aucs > 0.7)),
            "random_within_auc_mean": float(np.nanmean(rand_aucs)),
            "umi_within_auc_mean":    float(np.nanmean(umi_aucs)),
            "multi_within_auc_mean":  float(np.nanmean(multi_aucs)),
            "kappa_loco_auc_mean":    float(np.nanmean(kappa_loco)),
            "kappa_loco_auc_min":     float(np.nanmin(kappa_loco)),
            "kappa_loco_auc_max":     float(np.nanmax(kappa_loco)),
            "kappa_loco_n_gt_0p7":    int(np.sum(kappa_loco > 0.7)),
            "umi_loco_auc_mean":      float(np.nanmean(umi_loco)),
            "multi_loco_auc_mean":    float(np.nanmean(multi_loco)),
            "verdict_within_cohort":  verdict_within,
            "verdict_loco":           verdict_loco,
            "wallclock_s":            float(time.time() - t0),
        },
    }
    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_json}")
    print(f"\nTotal wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
