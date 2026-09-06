"""
META_patient_level -- Patient-level random-effects meta-analysis of per-cell
Ollivier-Ricci curvature (kappa) malignant-vs-non-malignant Cliff's delta
across seven solid-tumour scRNA-seq cohorts.

PROBLEM SOLVED: the published headline numbers are per-cell p-values
(e.g. p < 1e-106) which treat cells as independent. Cells from the same
patient share kNN edges and biological context (pseudoreplication). This
script makes the PATIENT the unit of inference.

Pipeline per cohort (kappa computation is method-identical to exp/E1):
  1. Load malignant + non-malignant cells with patient IDs.
  2. Normalise -> HVG-2000 (variance) -> PCA-50.
  3. Stratified sample (per patient per group, capped) -> ONE pooled kNN(k=15).
  4. Ollivier-Ricci kappa per edge (alpha=0.5 lazy walk, POT), per-cell mean.
  5. Per patient (>=10 mal AND >=10 non-mal with valid kappa): Cliff's delta
     + U-statistic (Hoeffding projection) standard error.
  6. DerSimonian-Laird random-effects meta across patients (per cohort + overall).
  7. Patient-clustered bootstrap (B=1000) per cohort: resample patients with
     replacement, recompute the pooled delta, 95% percentile CI.

Cohorts with recoverable patient IDs (6 -> primary patient-level meta):
  Tirosh melanoma (GSE72056), Darmanis GBM (GSE84465), Puram HNSCC (GSE103322),
  Peng/Moncada PDAC (GSE111672), Chen prostate (GSE176031), Olalekan ovarian
  HGSOC (GSE147082).

Li CRC (GSE81861) is EXCLUDED from the patient-level meta: the GEO processed
file (GSE81861_CRC_tumor_FPKM.csv) contains no patient-of-origin labels
(the RHCxxxx barcodes are sequential cell indices, not patient IDs). It is
reported separately as a single-cohort estimate with an explicit caveat.

Outputs (in this directory):
  results.json, per_patient_delta.csv, forest_plot_data.csv, run.log
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ot
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
SEED = 20260507

# stratified-sampling caps (per patient per group, and total per cohort)
CAP_PER_PATIENT_GROUP = 60
TOTAL_CAP = 3500
MIN_CELLS_PER_GROUP = 10     # patient included if >= this many in BOTH groups
B_BOOT = 1000                # patient-clustered bootstrap replicates
N_EDGES_TARGET = 6000        # Ollivier-Ricci sampled edges (coverage-biased)


# ===========================================================================
# Ollivier-Ricci primitives (copied verbatim from exp/E1_within_patient/run.py)
# ===========================================================================
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


def ollivier_ricci_edges(G, alpha=0.5, n_edges=None, rng=None, verbose=False):
    if rng is None:
        rng = np.random.default_rng(0)
    edges = list(G.edges())
    if n_edges is not None and n_edges < len(edges):
        idx = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in idx]
    out = {}
    for ei, (u, v) in enumerate(edges):
        if verbose and (ei + 1) % 1000 == 0:
            print(f"    OR edge {ei+1}/{len(edges)}", flush=True)
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


# ===========================================================================
# Effect sizes
# ===========================================================================
def cliffs_delta_fast(a, b):
    """Sort-based Cliff's delta = P(a>b) - P(a<b). Point estimate only."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    B = np.sort(b)
    n_less = np.searchsorted(B, a, side="left")       # b < a
    n_leq = np.searchsorted(B, a, side="right")        # b <= a
    n_greater = len(B) - n_leq                         # b > a
    gt = n_less.sum()
    lt = n_greater.sum()
    return float((gt - lt) / (len(a) * len(b)))


def cliff_delta_se(a, b):
    """Cliff's delta and its U-statistic (Hoeffding projection) SE.

    delta is a two-sample U-statistic of degree (1,1) with kernel sign(a-b).
    Var(delta) ~ zeta10/n1 + zeta01/n2 where zeta10 = Var(g1(X)),
    g1(x)=E[sign(x-Y)], estimated by the variance of the row means of the
    dominance matrix; analogously for columns.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    D = np.sign(a[:, None] - b[None, :])      # n1 x n2
    delta = float(D.mean())
    row = D.mean(axis=1)                        # g1(X_i)
    col = D.mean(axis=0)                        # g2(Y_j)
    var = float(row.var(ddof=1) / n1 + col.var(ddof=1) / n2)
    se = float(np.sqrt(max(var, 1e-12)))
    return delta, se


# ===========================================================================
# DerSimonian-Laird random-effects meta-analysis
# ===========================================================================
def dersimonian_laird(deltas, ses, label=""):
    """Random-effects meta with DerSimonian-Laird tau^2 estimator.

    Returns dict with pooled delta, SE, 95% CI, z, p, Q, df, tau2, I2, pQ, k.
    Handles k<2 gracefully (no heterogeneity stats).
    """
    d = np.asarray(deltas, float)
    s = np.asarray(ses, float)
    k = len(d)
    mask = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s = d[mask], s[mask]
    k = len(d)
    if k == 0:
        return dict(label=label, k=0, delta=np.nan, se=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, z=np.nan, p=np.nan, Q=np.nan, df=0,
                    tau2=np.nan, I2=np.nan, pQ=np.nan, delta_FE=np.nan)
    w = 1.0 / s ** 2                      # fixed-effect (inverse-variance) weights
    sw = w.sum()
    d_FE = float((w * d).sum() / sw)
    if k < 2:
        return dict(label=label, k=k, delta=d_FE, se=float(s[0]),
                    ci_lo=float(d_FE - 1.96 * s[0]),
                    ci_hi=float(d_FE + 1.96 * s[0]),
                    z=float(d_FE / s[0]), p=float(2 * norm.sf(abs(d_FE / s[0]))),
                    Q=0.0, df=0, tau2=0.0, I2=0.0, pQ=np.nan, delta_FE=d_FE)
    Q = float((w * (d - d_FE) ** 2).sum())
    df = k - 1
    c = sw - (w ** 2).sum() / sw
    tau2 = float(max(0.0, (Q - df) / c)) if c > 0 else 0.0
    wstar = 1.0 / (s ** 2 + tau2)         # random-effects weights
    swstar = wstar.sum()
    d_DL = float((wstar * d).sum() / swstar)
    se_DL = float(np.sqrt(1.0 / swstar))
    z = d_DL / se_DL
    p = float(2 * norm.sf(abs(z)))
    I2 = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0
    pQ = float(chi2.sf(Q, df))
    return dict(label=label, k=k, delta=d_DL, se=se_DL,
                ci_lo=float(d_DL - 1.96 * se_DL),
                ci_hi=float(d_DL + 1.96 * se_DL),
                z=float(z), p=p, Q=Q, df=df, tau2=tau2, I2=I2, pQ=pQ,
                delta_FE=d_FE)


def patient_clustered_bootstrap(patient_deltas, patient_ses, B=B_BOOT,
                                seed=SEED):
    """Resample patients with replacement; recompute DL pooled delta each time.
    Returns the bootstrap mean delta and 95% percentile CI."""
    d = np.asarray(patient_deltas, float)
    s = np.asarray(patient_ses, float)
    k = len(d)
    if k < 2:
        return dict(B=0, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, k, size=k)
        res = dersimonian_laird(d[idx], s[idx])
        boot[b] = res["delta"]
    boot = boot[np.isfinite(boot)]
    return dict(B=int(len(boot)), mean=float(np.mean(boot)),
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)))


# ===========================================================================
# Common preprocessing: HVG-2000 + PCA-50 on a normalised log matrix
# ===========================================================================
def hvg_pca(X_log, n_hvg=2000, n_pc=50, seed=SEED):
    """X_log: genes x cells (already log-normalised). Returns cells x n_pc."""
    var = X_log.var(axis=1)
    hvg = np.argsort(var)[::-1][:n_hvg]
    Xs = X_log[hvg].T.astype(np.float32)        # cells x genes
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    n_comp = min(n_pc, Xs.shape[0] - 1, Xs.shape[1])
    return PCA(n_components=n_comp, random_state=seed).fit_transform(Xs)


def stratified_indices(patient, is_g1, is_g2, cap_group=CAP_PER_PATIENT_GROUP,
                       total_cap=TOTAL_CAP, seed=SEED):
    """For each patient take up to cap_group cells from group1 (malignant) and
    group2 (comparator); cap the total. Returns indices into the input arrays."""
    rng = np.random.default_rng(seed)
    chosen = []
    for pid in np.unique(patient):
        for mask in (is_g1, is_g2):
            idx = np.where((patient == pid) & mask)[0]
            if len(idx) == 0:
                continue
            take = min(len(idx), cap_group)
            chosen.append(rng.choice(idx, size=take, replace=False))
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if len(chosen) > total_cap:                  # downsample keeping proportions
        chosen = rng.choice(chosen, size=total_cap, replace=False)
    return np.sort(chosen)


def compute_kappa_for_cohort(cohort, X_log, patient, is_mal, is_comp,
                             force=False, cache_tag=""):
    """Restrict to malignant + comparator cells (cell-type-matched, matching
    each cohort's published contrast), build a pooled kNN graph on a patient-
    stratified sample, compute per-cell Ollivier-Ricci kappa.
    Returns a DataFrame[patient, is_mal, kappa]. Results are cached to disk."""
    keep = is_mal | is_comp
    X_log = X_log[:, keep]
    patient = patient[keep]
    is_mal = is_mal[keep]
    is_comp = is_comp[keep]
    cache_path = CACHE / f"{cohort}{cache_tag}_kappa.npz"
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=True)
        df = pd.DataFrame({"patient": z["patient"], "is_mal": z["is_mal"],
                           "kappa": z["kappa"]})
        print(f"  [cache hit] {cohort}: {len(df)} cells loaded from {cache_path.name}")
        return df

    sel = stratified_indices(patient, is_mal, is_comp)
    pat_s, mal_s = patient[sel], is_mal[sel]
    print(f"  stratified sample: {len(sel)} cells "
          f"(mal={int(mal_s.sum())}, comparator={int((~mal_s).sum())}), "
          f"{len(np.unique(pat_s))} patients")
    Xpca = hvg_pca(X_log[:, sel])
    print(f"  PCA: {Xpca.shape}  building kNN(k=15) ...")
    G = build_knn_graph(Xpca, k=15)
    n_edges = min(N_EDGES_TARGET, G.number_of_edges())
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"computing OR on {n_edges} edges ...")
    t0 = time.time()
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges,
                                  rng=np.random.default_rng(SEED), verbose=True)
    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa = np.array([per_cell.get(i, np.nan) for i in range(len(sel))],
                     dtype=float)
    n_valid = int(np.sum(np.isfinite(kappa)))
    print(f"  OR done in {time.time()-t0:.1f}s; valid-kappa cells: "
          f"{n_valid}/{len(sel)} ({100*n_valid/len(sel):.1f}%)")

    df = pd.DataFrame({"patient": pat_s, "is_mal": mal_s, "kappa": kappa})
    np.savez(cache_path, patient=pat_s, is_mal=mal_s, kappa=kappa)
    print(f"  cached -> {cache_path.name}")
    return df


def per_patient_deltas(df, cohort):
    """Cliff's delta (mal vs non-mal kappa) per patient with U-statistic SE.
    Requires >= MIN_CELLS_PER_GROUP in BOTH groups (with valid kappa)."""
    rows = []
    df = df.dropna(subset=["kappa"])
    for pid, sub in df.groupby("patient"):
        a = sub.loc[sub.is_mal, "kappa"].values
        b = sub.loc[~sub.is_mal, "kappa"].values
        if len(a) < MIN_CELLS_PER_GROUP or len(b) < MIN_CELLS_PER_GROUP:
            continue
        delta, se = cliff_delta_se(a, b)
        tt = ttest_ind(a, b, equal_var=False)
        rows.append(dict(cohort=cohort, patient_id=str(pid),
                         n_malignant=int(len(a)), n_comparator=int(len(b)),
                         mean_kappa_mal=float(a.mean()),
                         mean_kappa_comp=float(b.mean()),
                         delta=float(delta), se=float(se),
                         ci_lo=float(delta - 1.96 * se),
                         ci_hi=float(delta + 1.96 * se),
                         welch_p=float(tt.pvalue)))
    return pd.DataFrame(rows)


# ===========================================================================
# Cohort loaders -> (X_log genes x cells, patient array, is_mal array)
# ===========================================================================
def load_tirosh():
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("[Tirosh] loading ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    patient = pd.to_numeric(header.iloc[1, 1:], errors="coerce").values
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)      # genes x cells (log2 TPM/10+1)
    is_mal = (malignant == 2)
    is_comp = (malignant == 1) & (nonmal_type == 1)     # T cells (published comparator)
    keep = is_mal | is_comp
    X = X[:, keep]
    patient = patient[keep].astype(str)
    is_mal = is_mal[keep]
    is_comp = is_comp[keep]
    keepc = np.array([p != "nan" and p.strip() != "" for p in patient])
    return X[:, keepc], patient[keepc], is_mal[keepc], is_comp[keepc]


def load_darmanis():
    EXPR = DATA / "GSE84465_GBM.csv"
    META = DATA / "GSE84465_meta.txt"
    print("[Darmanis] loading ...")
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
    patient = meta_idx["patient id"].values.astype(str)
    Xlog = np.log1p(df_expr.values.astype(np.float32))  # raw counts -> log1p
    is_mal = (cell_type == "Neoplastic")
    is_comp = (cell_type == "Immune cell")              # published comparator
    keep = is_mal | is_comp
    return Xlog[:, keep], patient[keep], is_mal[keep], is_comp[keep]


def load_puram():
    EXPR = DATA / "GSE103322_HNSCC.txt"
    print("[Puram] loading ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=6, header=None, low_memory=False)
    cell_ids = header.iloc[0, 1:].astype(str).values
    cancer = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    noncancer = pd.to_numeric(header.iloc[4, 1:], errors="coerce").values
    nonmal_type = header.iloc[5, 1:].astype(str).values
    nonmal_type = np.array([t.lstrip("-").strip() for t in nonmal_type])
    patient = []
    for cid in cell_ids:
        m = re.match(r"^HN(?:SCC)?(\d+)", cid)
        patient.append(m.group(1) if m else "-1")
    patient = np.array(patient)
    expr = pd.read_csv(EXPR, sep="\t", skiprows=6, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)      # log2(TPM/10+1)
    is_mal = (cancer == 1)
    is_comp = (noncancer == 1) & (nonmal_type == "Fibroblast")  # published comparator
    keep = (is_mal | is_comp) & (patient != "-1")
    X = X[:, keep]
    patient = patient[keep]
    is_mal = is_mal[keep]
    is_comp = is_comp[keep]
    return X, patient, is_mal, is_comp


def load_pdac():
    import gzip
    PDAC_A = DATA / "GSE111672_PDAC-A-indrop-filtered-expMat.txt.gz"
    PDAC_B = DATA / "GSE111672_PDAC-B-indrop-filtered-expMat.txt.gz"
    print("[Peng PDAC] loading ...")
    parts = []
    for path, tag in [(PDAC_A, "PDAC-A"), (PDAC_B, "PDAC-B")]:
        with gzip.open(path, "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
        labels = np.array([h.strip() for h in header[1:]], dtype=object)
        df = pd.read_csv(path, sep="\t", compression="gzip", header=0,
                         low_memory=False)
        X = df.iloc[:, 1:].values.astype(np.float32)   # genes x cells (counts)
        parts.append((tag, labels, X))
    # gene intersection
    ga = parts[0][1]  # placeholder; use df index
    # reload to grab gene names
    dfs = []
    for path, tag, _ in [(PDAC_A, "PDAC-A", None), (PDAC_B, "PDAC-B", None)]:
        df = pd.read_csv(path, sep="\t", compression="gzip", header=0,
                         index_col=0, low_memory=False)
        dfs.append((tag, df))
    common = sorted(set(dfs[0][1].index) & set(dfs[1][1].index))
    mats, labels_all, patient_all = [], [], []
    for tag, df in dfs:
        df = df.loc[common]
        mats.append(df.values.astype(np.float32))
        with gzip.open((DATA / f"GSE111672_{tag}-indrop-filtered-expMat.txt.gz"),
                       "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
        labs = [h.strip() for h in header[1:]]
        labels_all.extend(labs)
        patient_all.extend([tag] * len(labs))
    X = np.concatenate(mats, axis=1)                    # genes x cells (counts)
    labels = np.array(labels_all, dtype=object)
    patient = np.array(patient_all, dtype=object)
    mal_set = {"Cancer clone A", "Cancer clone B"}
    ductal_set = {
        "Ductal - terminal ductal like",
        "Ductal - CRISP3 high/centroacinar like",
        "Ductal - MHC Class II",
        "Ductal - APOL1 high/hypoxic",
    }
    is_mal = np.array([l in mal_set for l in labels])
    is_comp = np.array([l in ductal_set for l in labels])  # non-mal ductal (published)
    keep = is_mal | is_comp
    Xlog = np.log1p(X[:, keep])
    return Xlog, patient[keep], is_mal[keep], is_comp[keep]


# ---- Chen prostate: marker-based cell typing (ported from N5b) -------------
CHEN_MARKERS = {
    "luminal": ["KLK3", "KLK2", "AR", "MSMB", "ACPP", "NKX3-1", "KRT8", "KRT18", "EPCAM"],
    "basal":   ["KRT5", "KRT14", "TP63", "KRT15", "DST"],
    "tcell":   ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "PTPRC", "TRAC"],
    "myeloid": ["CD14", "CD68", "LYZ", "C1QA", "C1QB", "AIF1", "CSF1R"],
    "fibro":   ["DCN", "COL1A1", "COL1A2", "LUM", "PDGFRA", "ACTA2"],
    "endo":    ["VWF", "PECAM1", "CDH5", "ENG", "CLDN5"],
}


def load_chen():
    DATA_DIR = DATA / "GSE176031_chen"
    import os
    print("[Chen prostate] loading ...")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt.gz"))
    dfs, meta_rows = [], []
    for f in files:
        m = re.search(r"PR(\d+)_([TN])(\d)?", f)
        if not m:
            continue
        patient, tissue = f"PR{m.group(1)}", m.group(2)
        df = pd.read_csv(DATA_DIR / f, sep="\t", index_col=0)
        df.columns = [f"{f.split('_dge')[0]}__{c}" for c in df.columns]
        dfs.append(df)
        for c in df.columns:
            meta_rows.append({"cell_id": c, "patient": patient, "tissue": tissue})
    full = pd.concat(dfs, axis=1, join="outer").fillna(0).astype(np.float32)
    meta = pd.DataFrame(meta_rows).set_index("cell_id").loc[full.columns]

    # QC
    n_genes = (full.values > 0).sum(axis=0)
    n_umi = full.values.sum(axis=0)
    keep = (n_genes >= 200) & (n_umi >= 500)
    full = full.iloc[:, keep]
    meta = meta.iloc[keep]

    # CP10K + log1p
    X = full.values
    libsize = X.sum(axis=0)
    libsize = np.where(libsize == 0, 1, libsize)
    Xn = X / libsize[None, :] * 1e4
    Xlog_df = pd.DataFrame(np.log1p(Xn).astype(np.float32),
                           index=full.index, columns=full.columns)

    # marker scores -> argmax cell type
    scores = {}
    for ct, markers in CHEN_MARKERS.items():
        avail = [g for g in markers if g in Xlog_df.index]
        if not avail:
            scores[ct] = np.full(Xlog_df.shape[1], -np.inf)
            continue
        scores[ct] = Xlog_df.loc[avail].mean(axis=0).values
    score_df = pd.DataFrame(scores, index=Xlog_df.columns)
    zs = (score_df - score_df.mean(axis=0)) / (score_df.std(axis=0) + 1e-9)
    call = zs.idxmax(axis=1).values
    top_z = zs.max(axis=1).values
    sorted_z = np.sort(zs.values, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]
    confident = (top_z >= 0.5) & (margin >= 0.25)
    call = np.where(confident, call, "Unknown")
    meta = meta.copy()
    meta["celltype"] = call

    # malignant = luminal in tumor tissue; comparator = tcell (published +0.21)
    is_lum = (meta["celltype"] == "luminal").values
    is_tcell = (meta["celltype"] == "tcell").values
    is_t = (meta["tissue"] == "T").values
    is_unknown = (meta["celltype"] == "Unknown").values
    is_mal = is_lum & is_t
    is_comp = is_tcell                                  # Immune T (published)
    keep = (~is_unknown) & (is_mal | is_comp)
    Xlog = Xlog_df.values[:, keep]
    patient = meta["patient"].values[keep].astype(str)
    is_mal = is_mal[keep]
    is_comp = is_comp[keep]
    return Xlog, patient, is_mal, is_comp


# ---- Olalekan ovarian: Tirosh-style marker scoring (ported from N5c) -------
OV_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1",
                   "MUC16", "PAX8", "WT1", "FOLR1", "MSLN", "CLDN3", "CLDN4"],
    "Tcell": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "TRAC", "TRBC1", "TRBC2"],
    "Bcell": ["CD79A", "CD79B", "MS4A1", "CD19", "IGKC", "IGHG1"],
    "Macrophage": ["CD68", "CD163", "AIF1", "LYZ", "C1QA", "C1QB", "C1QC", "MARCO"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "PDGFRA", "PDGFRB",
                   "ACTA2", "FAP", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "ENG", "KDR"],
}


def _score_marker_set(X_cells_genes, gene_index, markers, n_ctrl=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    present = [g for g in markers if g in gene_index]
    if not present:
        return np.full(X_cells_genes.shape[0], np.nan, dtype=np.float32)
    sig_idx = np.array([gene_index[g] for g in present], dtype=int)
    means = X_cells_genes.mean(axis=0)
    quantiles = np.quantile(means, np.linspace(0, 1, 26))
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
    sig_score = X_cells_genes[:, sig_idx].mean(axis=1)
    ctrl_score = (X_cells_genes[:, ctrl_idx].mean(axis=1)
                  if len(ctrl_idx) else np.zeros_like(sig_score))
    return sig_score - ctrl_score


def load_olalekan():
    ODIR = DATA / "GSE147082"
    print("[Olalekan ovarian] loading ...")
    files = sorted(ODIR.glob("*.csv"))
    files = [f for f in files if not f.name.endswith(".gz")]
    dfs, pids = [], []
    for f in files:
        pid = f.stem.split("_")[-1]
        df = pd.read_csv(f, index_col=0, low_memory=False)
        dfs.append(df)
        pids.append(pid)
    common = sorted(set.intersection(*[set(d.index) for d in dfs]))
    parts, cell_pids = [], []
    for d, pid in zip(dfs, pids):
        sub = d.loc[common]
        parts.append(sub.values.astype(np.float32))
        cell_pids.extend([pid] * sub.shape[1])
    X_counts = np.concatenate(parts, axis=1)             # genes x cells
    patient = np.array(cell_pids, dtype=object)

    libsize = X_counts.sum(axis=0)
    keep = libsize >= 500
    X_counts = X_counts[:, keep]
    patient = patient[keep]
    detected = (X_counts > 0).sum(axis=1)
    keepg = detected >= 10
    X_counts = X_counts[keepg]
    gene_names = np.array(common)[keepg]
    libsize = X_counts.sum(axis=0)
    libsize[libsize == 0] = 1
    Xn = X_counts / libsize[None, :] * 1e4
    Xlog = np.log1p(Xn).astype(np.float32)              # genes x cells
    gene_index = {g: i for i, g in enumerate(gene_names)}

    Xc = Xlog.T                                         # cells x genes
    rng = np.random.default_rng(SEED)
    scores = {ct: _score_marker_set(Xc, gene_index, ml, n_ctrl=50, rng=rng)
              for ct, ml in OV_MARKERS.items()}
    types = list(scores.keys())
    M = np.stack([scores[t] for t in types], axis=1)
    M = np.where(np.isfinite(M), M, -np.inf)
    order = np.argsort(M, axis=1)
    top = order[:, -1]
    top_score = M[np.arange(len(M)), top]
    second_score = M[np.arange(len(M)), order[:, -2]]
    margin = top_score - second_score
    call = np.array(["Unknown"] * len(M), dtype=object)
    confident = (top_score > 0) & (margin >= 0.05)
    call[confident] = np.array(types)[top[confident]]

    is_mal = (call == "Epithelial")
    is_comp = (call == "Tcell")                          # T cells (published +0.48)
    keep = is_mal | is_comp
    return Xlog[:, keep], patient[keep], is_mal[keep], is_comp[keep]


def load_li_crc_single():
    """Li CRC has NO patient IDs in the GEO processed file. Loaded only for a
    single-cohort (non-patient-level) reference estimate. Comparator = T cells
    (matches the published E4 headline)."""
    csv_path = DATA / "GSE81861_CRC_tumor_FPKM.csv"
    print("[Li CRC] loading (single-cohort, no patient IDs) ...")
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    cell_types = []
    for c in df.columns:
        parts = c.split("__")
        cell_types.append(parts[1] if len(parts) >= 2 else "NA")
    cell_types = np.array(cell_types)
    is_mal = (cell_types == "Epithelial")
    is_comp = (cell_types == "Tcell")
    keep = is_mal | is_comp
    df = df.loc[:, keep]
    Xlog = np.log1p(df.values.astype(np.float32))       # FPKM -> log1p
    is_mal = is_mal[keep]
    is_comp = is_comp[keep]
    patient = np.array(["LiCRC_single"] * len(is_mal), dtype=object)
    return Xlog, patient, is_mal, is_comp


COHORT_LOADERS = {
    "Tirosh_melanoma": load_tirosh,
    "Darmanis_GBM": load_darmanis,
    "Puram_HNSCC": load_puram,
    "Peng_PDAC": load_pdac,
    "Chen_prostate": load_chen,
    "Olalekan_ovarian": load_olalekan,
}

# cell-type-matched comparator per cohort (matches each published headline; see
# RESEARCH_NOTE.md Sec 3.1. Using pooled "all non-malignant" is INVALID: it
# mixes benign epithelial cells into the comparator and flips the sign in
# prostate / PDAC -- documented in RESULTS.md Sec 3.3.)
COMPARATORS = {
    "Tirosh_melanoma": "T cells (nonmal_type==1)",
    "Darmanis_GBM": "Immune cell",
    "Puram_HNSCC": "Fibroblast",
    "Peng_PDAC": "non-malignant Ductal (4 subtypes)",
    "Chen_prostate": "Immune T (marker-derived tcell)",
    "Olalekan_ovarian": "T cells (marker-derived Tcell)",
}


COHORT_ORDER = ["Tirosh_melanoma", "Darmanis_GBM", "Puram_HNSCC", "Peng_PDAC",
                "Chen_prostate", "Olalekan_ovarian"]


def make_forest_plot(all_pp, cohort_meta, overall, out_png):
    """Per-patient Cliff's delta with 95% CI, grouped by cohort; per-cohort
    and overall DL pooled estimates as diamonds."""
    fig, ax = plt.subplots(figsize=(9, 0.34 * (len(all_pp) + 12) + 2))
    palette = dict(zip(COHORT_ORDER,
                       ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b"]))
    y = 0
    yticks, ylabels = [], []
    for c in COHORT_ORDER:
        sub = all_pp[all_pp.cohort == c].copy()
        if not len(sub):
            continue
        # cohort header
        yticks.append(y)
        ylabels.append(f"{c}  (n={len(sub)})")
        ax.axhline(y, color="#cccccc", lw=0.5)
        y -= 1
        sub = sub.sort_values("delta")
        for _, r in sub.iterrows():
            col = palette.get(c, "#333333")
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.0)
            ax.scatter(r.delta, y, color=col, s=14, zorder=3)
            yticks.append(y)
            ylabels.append(f"   {r.patient_id}  "
                           f"(m={r.n_malignant}/c={r.n_comparator})")
            y -= 1
        # cohort pooled diamond
        m = cohort_meta.get(c, {})
        if np.isfinite(m.get("delta", np.nan)):
            d, lo, hi = m["delta"], m["ci_lo"], m["ci_hi"]
            ax.plot([lo, hi], [y, y], color=palette.get(c, "#333"), lw=2.5)
            ax.scatter(d, y, marker="D", color=palette.get(c, "#333"),
                       s=70, zorder=4, edgecolor="k", lw=0.5)
            yticks.append(y)
            ylabels.append(f"   SUBTOTAL delta={d:+.2f} "
                           f"[{lo:+.2f},{hi:+.2f}] p={m['p']:.1e}")
            y -= 1.5
    # overall
    yticks.append(y)
    ylabels.append("OVERALL (random-effects)")
    y -= 1
    d, lo, hi = overall["delta"], overall["ci_lo"], overall["ci_hi"]
    ax.plot([lo, hi], [y, y], color="k", lw=3)
    ax.scatter(d, y, marker="D", color="k", s=160, zorder=5, edgecolor="k")
    yticks.append(y)
    ylabels.append(f"   delta={d:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                   f"I2={overall['I2']:.1f}%  p={overall['p']:.1e}")
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Cliff's delta  (malignant vs comparator kappa, within patient)\n"
                  "positive => malignant cells more positively curved")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title("Patient-level meta-analysis of per-cell Ollivier-Ricci kappa\n"
                 "35 patients across 6 solid-tumour cohorts "
                 "(DerSimonian-Laird random effects)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


# ===========================================================================
# Driver
# ===========================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print("META_patient_level -- patient-level random-effects meta-analysis")
    print(f"  seed={SEED}  cap/group={CAP_PER_PATIENT_GROUP}  "
          f"total_cap={TOTAL_CAP}  min/group={MIN_CELLS_PER_GROUP}")
    print(f"  bootstrap B={B_BOOT}  OR edges target={N_EDGES_TARGET}")
    print("=" * 72)

    all_patient_rows = []
    cohort_meta = {}

    for cohort, loader in COHORT_LOADERS.items():
        print("\n" + "#" * 72)
        print(f"# COHORT: {cohort}")
        print("#" * 72)
        try:
            X_log, patient, is_mal, is_comp = loader()
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            cohort_meta[cohort] = {"error": str(e)}
            continue
        # normalise types
        patient = np.asarray(patient, dtype=object).astype(str)
        is_mal = np.asarray(is_mal, dtype=bool)
        is_comp = np.asarray(is_comp, dtype=bool)
        n_cells = X_log.shape[1]
        n_pat = len(np.unique(patient))
        print(f"  loaded (mal+comparator): {n_cells} cells, {n_pat} patients; "
              f"mal={int(is_mal.sum())} comparator={int(is_comp.sum())}")

        # per-patient cell counts (mal vs comparator, pre-sampling)
        pc = pd.DataFrame({"patient": patient, "is_mal": is_mal, "is_comp": is_comp})
        qual = 0
        for pid in np.unique(patient):
            m = ((pc.patient == pid) & pc.is_mal).sum()
            c = ((pc.patient == pid) & pc.is_comp).sum()
            if m >= MIN_CELLS_PER_GROUP and c >= MIN_CELLS_PER_GROUP:
                qual += 1
        print(f"  patients with >={MIN_CELLS_PER_GROUP}/{MIN_CELLS_PER_GROUP} "
              f"(mal/comparator) cells: {qual}/{n_pat}")

        df = compute_kappa_for_cohort(cohort, X_log, patient, is_mal, is_comp,
                                      cache_tag="_v2")
        pp = per_patient_deltas(df, cohort)
        print(f"  patients passing delta threshold: {len(pp)}")
        if len(pp):
            print(pp[["patient_id", "n_malignant", "n_comparator",
                      "delta", "se", "ci_lo", "ci_hi"]].to_string(index=False))

        # naive cell-level delta (pooled, ignoring patient) for comparison
        valid = df.dropna(subset=["kappa"])
        a_all = valid.loc[valid.is_mal, "kappa"].values
        b_all = valid.loc[~valid.is_mal, "kappa"].values
        naive_delta = cliffs_delta_fast(a_all, b_all)
        naive_t = ttest_ind(a_all, b_all, equal_var=False)
        print(f"  NAIVE cell-level (pseudorep): delta={naive_delta:+.4f} "
              f"Welch p={naive_t.pvalue:.3g}  n={len(a_all)}/{len(b_all)}")

        # per-cohort DL meta
        if len(pp):
            cm = dersimonian_laird(pp["delta"].values, pp["se"].values,
                                    label=cohort)
            boot = patient_clustered_bootstrap(pp["delta"].values,
                                               pp["se"].values)
        else:
            cm = dersimonian_laird([], [], label=cohort)
            boot = dict(B=0, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan)
        cm["n_patients_in_meta"] = int(len(pp))
        cm["comparator"] = COMPARATORS.get(cohort, "")
        cm["naive_cell_delta"] = float(naive_delta) if np.isfinite(naive_delta) else None
        cm["naive_cell_p"] = float(naive_t.pvalue)
        cm["bootstrap"] = boot
        cm["n_patients_qualifying"] = int(qual)
        cm["n_patients_total"] = int(n_pat)
        print(f"  DL pooled delta={cm['delta']:+.4f} "
              f"[{cm['ci_lo']:+.4f},{cm['ci_hi']:+.4f}]  "
              f"tau2={cm['tau2']:.5f}  I2={cm['I2']:.1f}%  "
              f"Q={cm['Q']:.2f}(df={cm['df']}) pQ={cm['pQ']:.3g}  "
              f"p={cm['p']:.3g}")
        print(f"  bootstrap (B={boot['B']}): mean={boot['mean']:+.4f} "
              f"CI=[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}]")
        cohort_meta[cohort] = cm
        all_patient_rows.append(pp)

    # ----- Li CRC single-cohort reference (excluded from patient-level meta) -
    print("\n" + "#" * 72)
    print("# COHORT: Li_CRC (single-cohort reference, NO patient IDs)")
    print("#" * 72)
    try:
        X_log, patient, is_mal, is_comp = load_li_crc_single()
        df = compute_kappa_for_cohort("Li_CRC_single", X_log, patient, is_mal,
                                      is_comp, force=True, cache_tag="_v2")
        valid = df.dropna(subset=["kappa"])
        a_all = valid.loc[valid.is_mal, "kappa"].values
        b_all = valid.loc[~valid.is_mal, "kappa"].values
        naive_delta = cliffs_delta_fast(a_all, b_all)
        naive_t = ttest_ind(a_all, b_all, equal_var=False)
        print(f"  Li CRC single-cohort delta={naive_delta:+.4f} "
              f"Welch p={naive_t.pvalue:.3g}  n={len(a_all)}/{len(b_all)}")
        print("  NOTE: GSE81861 processed file has no patient-of-origin labels;")
        print("        excluded from patient-level meta-analysis.")
        cohort_meta["Li_CRC_single"] = {
            "delta": float(naive_delta), "p": float(naive_t.pvalue),
            "n_mal": int(len(a_all)), "n_comparator": int(len(b_all)),
            "comparator": "T cells",
            "excluded_from_patient_meta": True,
            "reason": "patient IDs not in GEO processed file "
                      "(GSE81861_CRC_tumor_FPKM.csv; RHCxxxx are sequential "
                      "cell indices)"}
    except Exception as e:
        print(f"  Li CRC load failed: {e}")
        cohort_meta["Li_CRC_single"] = {"error": str(e)}

    # ----- overall patient-level meta (all patients across 6 cohorts) -------
    all_pp = pd.concat([r for r in all_patient_rows if len(r)], ignore_index=True)
    print("\n" + "=" * 72)
    print(f"OVERALL patient-level meta: {len(all_pp)} patients across "
          f"{all_pp['cohort'].nunique()} cohorts")
    print("=" * 72)
    overall = dersimonian_laird(all_pp["delta"].values, all_pp["se"].values,
                                label="OVERALL_patient_level")
    boot_overall = patient_clustered_bootstrap(all_pp["delta"].values,
                                               all_pp["se"].values, B=B_BOOT,
                                               seed=SEED + 1)
    overall["bootstrap"] = boot_overall
    print(f"  pooled delta = {overall['delta']:+.4f} "
          f"[{overall['ci_lo']:+.4f}, {overall['ci_hi']:+.4f}]")
    print(f"  z = {overall['z']:.2f}  p = {overall['p']:.3e}")
    print(f"  tau2 = {overall['tau2']:.5f}  I2 = {overall['I2']:.1f}%")
    print(f"  Q = {overall['Q']:.2f} (df={overall['df']})  p(Q) = {overall['pQ']:.3e}")
    print(f"  bootstrap (B={boot_overall['B']}): mean={boot_overall['mean']:+.4f} "
          f"CI=[{boot_overall['ci_lo']:+.4f},{boot_overall['ci_hi']:+.4f}]")

    # ----- cohort-as-study meta (6 cohorts, each cohort's DL pooled delta) ---
    cohort_deltas, cohort_ses, cohort_names = [], [], []
    for c, m in cohort_meta.items():
        if c == "Li_CRC_single":
            continue
        if "delta" in m and np.isfinite(m["delta"]) and np.isfinite(m["se"]):
            cohort_deltas.append(m["delta"])
            cohort_ses.append(m["se"])
            cohort_names.append(c)
    cohort_as_study = dersimonian_laird(cohort_deltas, cohort_ses,
                                        label="OVERALL_cohort_level")
    print(f"\nCOHORT-AS-STUDY meta: {len(cohort_deltas)} cohorts")
    print(f"  pooled delta = {cohort_as_study['delta']:+.4f} "
          f"[{cohort_as_study['ci_lo']:+.4f}, {cohort_as_study['ci_hi']:+.4f}]  "
          f"I2 = {cohort_as_study['I2']:.1f}%  p = {cohort_as_study['p']:.3e}")

    # ----- write outputs ----------------------------------------------------
    all_pp.to_csv(HERE / "per_patient_delta.csv", index=False)
    forest = all_pp.copy()
    forest.to_csv(HERE / "forest_plot_data.csv", index=False)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    results = {
        "seed": SEED,
        "design": {
            "unit_of_analysis": "patient",
            "effect_size": "Cliff's delta (malignant vs non-malignant kappa, within patient)",
            "delta_variance": "U-statistic Hoeffding projection (row/col mean variance)",
            "meta_method": "DerSimonian-Laird random-effects (inverse-variance)",
            "bootstrap": f"patient-clustered, B={B_BOOT}",
            "min_cells_per_group": MIN_CELLS_PER_GROUP,
            "pipeline": "HVG-2000, PCA-50, kNN k=15, Ollivier-Ricci alpha=0.5 (POT)",
            "cap_per_patient_group": CAP_PER_PATIENT_GROUP,
            "total_cap_per_cohort": TOTAL_CAP,
        },
        "overall_patient_level": _clean(overall),
        "overall_cohort_level": _clean(cohort_as_study),
        "per_cohort": {k: _clean(v) for k, v in cohort_meta.items()},
        "n_patients_total": int(len(all_pp)),
        "n_cohorts_in_patient_meta": int(all_pp["cohort"].nunique()),
        "wallclock_seconds": float(time.time() - t_start),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ----- clean summary table ---------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY TABLE -- patient-level Cliff's delta (malignant > non-mal)")
    print("=" * 72)
    print(f"{'cohort':<20} {'k_pat':>6} {'pooled_delta':>13} {'95% CI':>22} "
          f"{'I2%':>6} {'p(Q)':>9} {'p(meta)':>10}")
    print("-" * 92)
    for c in ["Tirosh_melanoma", "Darmanis_GBM", "Puram_HNSCC", "Peng_PDAC",
              "Chen_prostate", "Olalekan_ovarian"]:
        m = cohort_meta.get(c, {})
        if "delta" not in m or not np.isfinite(m.get("delta", np.nan)):
            print(f"{c:<20} {'--':>6} {'--':>13} {'--':>22} {'--':>6} "
                  f"{'--':>9} {'--':>10}")
            continue
        ci = f"[{m['ci_lo']:+.3f},{m['ci_hi']:+.3f}]"
        print(f"{c:<20} {m.get('n_patients_in_meta','?'):>6} "
              f"{m['delta']:>+13.4f} {ci:>22} {m['I2']:>6.1f} "
              f"{m['pQ']:>9.2e} {m['p']:>10.2e}")
    print("-" * 92)
    print(f"{'OVERALL (patient)':<20} {len(all_pp):>6} "
          f"{overall['delta']:>+13.4f} "
          f"[{overall['ci_lo']:+.3f},{overall['ci_hi']:+.3f}]"
          f"   {overall['I2']:>6.1f} {overall['pQ']:>9.2e} "
          f"{overall['p']:>10.2e}")
    li = cohort_meta.get("Li_CRC_single", {})
    if "delta" in li:
        print(f"\nLi_CRC (no patient IDs, single-cohort ref): delta={li['delta']:+.4f} "
              f"p={li['p']:.2e}  [EXCLUDED from patient-level meta]")

    print(f"\nWrote: {HERE/'results.json'}")
    print(f"Wrote: {HERE/'per_patient_delta.csv'}  ({len(all_pp)} patients)")
    print(f"Wrote: {HERE/'forest_plot_data.csv'}")
    forest_png = HERE / "forest_plot.png"
    make_forest_plot(all_pp, cohort_meta, overall, forest_png)
    print(f"Wrote: {forest_png}")
    print(f"\nWallclock: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
