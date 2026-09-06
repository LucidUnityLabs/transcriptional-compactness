"""
HELDOUT_GSE161529 -- POST-HOC SENSITIVITY analysis of dual-label handling.

NOT part of PREREGISTRATION.md. The baseline variant ('excluded') is the
pre-registered run recorded in results.json. This script re-runs the
PRIMARY arm (malignant vs immune) ONLY, holding everything else identical
to exp/HELDOUT_GSE161529/run.py (seed 20260507, caps 60/patient/group and
total 3500, HVG-2000, PCA-50, kNN k=15, Ollivier-Ricci alpha=0.5 on 4000
edges, >=10 cells/group filter, DerSimonian-Laird meta, patient-clustered
bootstrap B=1000), varying ONLY the rule applied to the 26,439 dual-labeled
barcodes (listed both 'malignant' and 'immune' in cells_primary.tsv.gz):

  excluded   -- baseline rule: dual-labeled barcodes dropped entirely
                (must reproduce results.json pooled delta exactly:
                +0.16801092468924114; the run ABORTS otherwise)
  dual_mal   -- dual-labeled barcodes kept, counted as malignant
  dual_immune-- dual-labeled barcodes kept, counted as immune

Does NOT modify run.py, results.json, or the baseline cache files; uses
its own cache files cache/HELDOUT_GSE161529_primary_sens_<variant>_kappa.npz.

Output: results_sensitivity.json (numbers only; no interpretation here).

Copied from exp/HELDOUT_GSE161529/run.py (which copied exp/HELDOUT_GSE131907,
exp/META_patient_level, exp/E1_within_patient, exp/V2_metastasis verbatim).
"""

import gzip
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import ot
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.stats import chi2, norm, ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
GEO = DATA / "GSE161529"
LAB_DIR = GEO / "labels"
SAMPLES_DIR = GEO / "samples"
FEATURES = SAMPLES_DIR / "GSE161529_features.tsv.gz"
MANIFEST = SAMPLES_DIR / "MANIFEST.json"

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

SEED = 20260507

# locked parameters (identical to run.py / PREREGISTRATION.md)
N_HVG = 2000            # or all genes if fewer
N_PCA = 50
K_NN = 15
ALPHA = 0.5
N_EDGES_TARGET = 4000   # or all edges if fewer
MIN_CELLS_PER_GROUP = 10
MIN_PATIENTS = 5

# stratified-sampling caps (exp/META_patient_level convention)
CAP_PER_PATIENT_GROUP = 60
TOTAL_CAP = 3500
B_BOOT = 1000

COHORT = "HELDOUT_GSE161529"
ARM = "primary"
COMP_LABEL = "immune"
LABELS_FILE = "cells_primary.tsv.gz"

VARIANT_ORDER = ("excluded", "dual_mal", "dual_immune")

# baseline (pre-registered) primary-arm numbers from results.json; the
# 'excluded' variant must reproduce these exactly before anything else runs
EXPECTED_BASELINE = {
    "pooled_delta": 0.16801092468924114,
    "se": 0.12188877196441941,
    "ci_lo": -0.07089106836102091,
    "ci_hi": 0.4069129177395032,
    "one_sided_p": 0.08404061877133445,
    "i2": 96.14265770302804,
    "k": 18,
    "n_delta_positive": 12,
    "n_delta_negative": 6,
    "bootstrap_mean": 0.17293171428411425,
    "bootstrap_ci_lo": -0.026810050510547197,
    "bootstrap_ci_hi": 0.38217938145881786,
}


# ===========================================================================
# Ollivier-Ricci primitives (copied verbatim from run.py)
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
# Effect sizes (copied verbatim from run.py)
# ===========================================================================
def cliffs_delta_fast(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    B = np.sort(b)
    n_less = np.searchsorted(B, a, side="left")
    n_leq = np.searchsorted(B, a, side="right")
    n_greater = len(B) - n_leq
    gt = n_less.sum()
    lt = n_greater.sum()
    return float((gt - lt) / (len(a) * len(b)))


def cliff_delta_se(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    D = np.sign(a[:, None] - b[None, :])
    delta = float(D.mean())
    row = D.mean(axis=1)
    col = D.mean(axis=0)
    var = float(row.var(ddof=1) / n1 + col.var(ddof=1) / n2)
    se = float(np.sqrt(max(var, 1e-12)))
    return delta, se


# ===========================================================================
# DerSimonian-Laird random-effects meta (copied verbatim from run.py)
# ===========================================================================
def dersimonian_laird(deltas, ses, label=""):
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
    w = 1.0 / s ** 2
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
    wstar = 1.0 / (s ** 2 + tau2)
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
# Stratified sampling (copied verbatim from run.py)
# ===========================================================================
def stratified_indices(patient, is_g1, is_g2, cap_group=CAP_PER_PATIENT_GROUP,
                       total_cap=TOTAL_CAP, seed=SEED):
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
    if len(chosen) > total_cap:
        chosen = rng.choice(chosen, size=total_cap, replace=False)
    return np.sort(chosen)


# ===========================================================================
# Labels + per-sample 10x loading (copied verbatim from run.py)
# ===========================================================================
def load_labels(name):
    df = pd.read_csv(LAB_DIR / name, sep="\t", dtype={"patient": str})
    return df


def load_mapping():
    with open(MANIFEST) as f:
        return json.load(f)["mapping"]   # sample -> {gsm, stem}


def load_features():
    genes = []
    with gzip.open(FEATURES, "rt") as f:
        for line in f:
            genes.append(line.split("\t")[1].rstrip("\n"))
    return np.array(genes, dtype=object)


def read_barcodes(stem):
    with gzip.open(SAMPLES_DIR / f"{stem}-barcodes.tsv.gz", "rt") as f:
        return [line.strip() for line in f]


def mtx_dims(stem):
    with gzip.open(SAMPLES_DIR / f"{stem}-matrix.mtx.gz", "rt") as f:
        for line in f:
            if line.startswith("%"):
                continue
            a = line.split()
            return int(a[0]), int(a[1])          # (n_features, n_cells)


def load_counts_for_cells(mapping, wanted):
    """wanted: dict sample -> set of full-label barcodes to keep.
    Returns (counts CSR cells x genes, gene_names, cell_barcodes) for the
    union of wanted cells that are actually present in the mtx files.
    Union order: sorted(sample), then barcode order in mtx."""
    gene_names = load_features()
    n_genes = len(gene_names)
    rows, cols, vals = [], [], []
    cell_barcodes = []
    g_row = 0
    per_sample_loaded = {}
    for sample in sorted(wanted):
        stem = mapping[sample]["stem"]
        bcs = read_barcodes(stem)
        pos = {b: i for i, b in enumerate(bcs)}
        local_wanted = {b[len(sample) + 1:]: b for b in wanted[sample]}
        keep_local = {loc: pos[loc] for loc in local_wanted if loc in pos}
        n_missing = len(wanted[sample]) - len(keep_local)
        per_sample_loaded[sample] = {
            "n_wanted": len(wanted[sample]), "n_in_mtx": len(keep_local),
            "n_wanted_absent_from_mtx": n_missing}
        if not keep_local:
            continue
        keep_cols = np.array(sorted(keep_local.values()), dtype=np.int64)
        col_to_row = {c: g_row + r for r, c in enumerate(keep_cols)}
        n_feat, n_cells = mtx_dims(stem)
        t0 = time.time()
        tri = pd.read_csv(SAMPLES_DIR / f"{stem}-matrix.mtx.gz", sep=" ",
                          comment="%", header=None,
                          dtype=np.int32).to_numpy()
        if tri.shape[0] and tri[0, 0] == n_feat and tri[0, 1] == n_cells:
            tri = tri[1:]  # MatrixMarket dims line parses as a data row
        tri = tri[(tri[:, 0] <= n_feat) & (tri[:, 1] <= n_cells)]
        m = np.isin(tri[:, 1] - 1, keep_cols)
        tri = tri[m]
        rows.append(np.array([col_to_row[c - 1] for c in tri[:, 1]],
                             dtype=np.int64))
        cols.append(tri[:, 0].astype(np.int64) - 1)
        vals.append(tri[:, 2].astype(np.float32))
        cell_barcodes.extend(local_wanted[bcs[c]] for c in keep_cols)
        g_row += len(keep_cols)
        print(f"    {sample}: kept {len(keep_local)} cells "
              f"({tri.shape[0]:,} nnz, {time.time()-t0:.1f}s)", flush=True)
        del tri
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    counts = coo_matrix((vals, (rows, cols)),
                        shape=(g_row, n_genes)).tocsr()
    counts.sum_duplicates()
    return counts, gene_names, np.array(cell_barcodes, dtype=object), \
        per_sample_loaded


# ===========================================================================
# kappa computation per variant with disk cache
# (copy of run.py compute_arm; ONLY the dual-label resolution at the top
#  differs -- everything from stratified_indices onward is verbatim)
# ===========================================================================
def compute_arm(variant, mapping, force=False):
    cache_path = CACHE / f"{COHORT}_{ARM}_sens_{variant}_kappa.npz"
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=True)
        df = pd.DataFrame({"patient": z["patient"], "sample": z["sample"],
                           "is_mal": z["is_mal"], "kappa": z["kappa"]})
        print(f"  [cache hit] {len(df)} cells loaded from {cache_path.name}")
        return df, None

    lab = load_labels(LABELS_FILE)
    lab = lab[lab["label"].isin(["malignant", COMP_LABEL])].copy()
    n_rows_raw = len(lab)
    nlab = lab.groupby("barcode")["label"].nunique()
    conflicted = set(nlab[nlab > 1].index)
    conflict_by_sample = (lab[lab["barcode"].isin(conflicted)]
                          .groupby("sample").size().to_dict())

    # ---- THE ONLY DIVERGENCE FROM run.py: dual-label resolution rule ----
    is_dual = lab["barcode"].isin(conflicted)
    if variant == "excluded":
        lab = lab[~is_dual].drop_duplicates("barcode", keep="first")
        n_dual_excl, n_dual_mal, n_dual_comp = len(conflicted), 0, 0
    elif variant == "dual_mal":
        lab = lab[~(is_dual & (lab["label"] != "malignant"))].drop_duplicates(
            "barcode", keep="first")
        n_dual_excl, n_dual_mal, n_dual_comp = 0, len(conflicted), 0
    elif variant == "dual_immune":
        lab = lab[~(is_dual & (lab["label"] == "malignant"))].drop_duplicates(
            "barcode", keep="first")
        n_dual_excl, n_dual_mal, n_dual_comp = 0, 0, len(conflicted)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    # ---------------------------------------------------------------------

    patient = lab["patient"].astype(str).values
    is_mal = (lab["label"] == "malignant").values
    is_comp = ~is_mal
    print(f"  [{variant}] labels: {n_rows_raw} rows -> {len(lab)} unique "
          f"cells ({len(conflicted)} dual-labeled barcodes: "
          f"excluded={n_dual_excl}, as-malignant={n_dual_mal}, "
          f"as-{COMP_LABEL}={n_dual_comp}), "
          f"{len(np.unique(patient))} patients "
          f"(malignant={int(is_mal.sum())}, {COMP_LABEL}={int(is_comp.sum())})")

    qual = {}
    for pid in np.unique(patient):
        m = int(((patient == pid) & is_mal).sum())
        c = int(((patient == pid) & is_comp).sum())
        qual[pid] = {"n_malignant": m, f"n_{COMP_LABEL}": c,
                     "qualifies": bool(m >= MIN_CELLS_PER_GROUP
                                       and c >= MIN_CELLS_PER_GROUP)}
    n_qual = sum(v["qualifies"] for v in qual.values())
    print(f"  [{variant}] patients with >= {MIN_CELLS_PER_GROUP}/"
          f"{MIN_CELLS_PER_GROUP} (mal/{COMP_LABEL}) cells pre-sampling: "
          f"{n_qual}/{len(qual)}")

    sel = stratified_indices(patient, is_mal, is_comp)
    sel_lab = lab.iloc[sel]
    pat_s = patient[sel]
    mal_s = is_mal[sel]
    print(f"  [{variant}] stratified sample: {len(sel)} cells "
          f"(mal={int(mal_s.sum())}, comp={int((~mal_s).sum())}), "
          f"{len(np.unique(pat_s))} patients")

    wanted = {}
    for sample, sub in sel_lab.groupby("sample"):
        wanted[sample] = set(sub["barcode"])
    counts, gene_names, cell_order, per_sample_loaded = \
        load_counts_for_cells(mapping, wanted)

    lab_by_bc = lab.set_index("barcode")
    kept = lab_by_bc.loc[cell_order]
    assert len(kept) == len(cell_order) == counts.shape[0], \
        "barcode alignment failed (duplicate labels?)"
    pat_m = kept["patient"].astype(str).values
    mal_m = (kept["label"] == "malignant").values
    smp_m = kept["sample"].values

    # log1p-CPM(1e6), gene filter >=5 cells (V2/R5 loading convention)
    print("  [Normalize] log1p-CPM ...", flush=True)
    expressed = (counts > 0).sum(axis=0).A1
    gene_keep = np.where(expressed >= 5)[0]
    print(f"  genes detected in >=5 cells: {len(gene_keep)} of "
          f"{counts.shape[1]}", flush=True)
    counts = counts[:, gene_keep]
    gene_names = gene_names[gene_keep]

    libsize = counts.sum(axis=1).A1.astype(np.float32)
    libsize[libsize == 0] = 1.0
    norm = diags(1e6 / libsize) @ counts
    norm = norm.astype(np.float32)
    norm.data = np.log1p(norm.data)
    norm = norm.tocsr()

    # HVG-2000 by variance over sampled cells (META hvg_pca convention)
    nrow = norm.shape[0]
    sums = np.array(norm.sum(axis=0)).ravel()
    sq = norm.copy()
    sq.data = sq.data ** 2
    sums_sq = np.array(sq.sum(axis=0)).ravel()
    means = sums / nrow
    var = sums_sq / nrow - means ** 2
    n_hvg = min(N_HVG, len(gene_names))
    hvg = np.argsort(var)[::-1][:n_hvg]
    Xm = norm[:, hvg].toarray().astype(np.float32)
    Xm = Xm - Xm.mean(axis=0, keepdims=True)
    print(f"  HVG matrix: {Xm.shape}")
    n_comp = min(N_PCA, Xm.shape[0] - 1, Xm.shape[1])
    Xpca = PCA(n_components=n_comp, random_state=SEED).fit_transform(Xm)
    print(f"  PCA: {Xpca.shape}")
    del Xm, sq, norm, counts

    print(f"  building kNN(k={K_NN}) ...")
    G = build_knn_graph(Xpca, k=K_NN)
    n_edges = min(N_EDGES_TARGET, G.number_of_edges())
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"computing OR on {n_edges} edges ...", flush=True)
    t0 = time.time()
    edge_k = ollivier_ricci_edges(G, alpha=ALPHA, n_edges=n_edges,
                                  rng=np.random.default_rng(SEED), verbose=True)
    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa = np.array([per_cell.get(i, np.nan) for i in range(len(pat_m))],
                     dtype=float)
    n_valid = int(np.sum(np.isfinite(kappa)))
    print(f"  OR done in {time.time()-t0:.1f}s; valid-kappa cells: "
          f"{n_valid}/{len(pat_m)} ({100*n_valid/len(pat_m):.1f}%)")

    df = pd.DataFrame({"patient": pat_m, "sample": smp_m,
                       "is_mal": mal_m, "kappa": kappa})
    np.savez(cache_path, patient=pat_m, sample=smp_m, is_mal=mal_m,
             kappa=kappa)
    print(f"  cached -> {cache_path.name}")

    extras = {
        "n_label_rows_raw": int(n_rows_raw),
        "n_dual_labeled_barcodes": int(len(conflicted)),
        "n_dual_labeled_cells_excluded": int(n_dual_excl),
        "n_dual_labeled_cells_assigned_malignant": int(n_dual_mal),
        "n_dual_labeled_cells_assigned_comparator": int(n_dual_comp),
        "dual_label_conflict_rows_by_sample": conflict_by_sample,
        "n_cells_in_labels_arm": int(len(lab)),
        "n_malignant_pool": int(is_mal.sum()),
        "n_comparator_pool": int(is_comp.sum()),
        "n_patients_in_labels_arm": int(len(np.unique(patient))),
        "qualifying_patients_pre_sampling": qual,
        "n_patients_qualifying_pre_sampling": int(n_qual),
        "n_cells_sampled": int(len(sel)),
        "n_malignant_sampled": int(mal_s.sum()),
        "n_comparator_sampled": int((~mal_s).sum()),
        "per_sample_load_audit": per_sample_loaded,
        "patients_lost_to_matrix_join": sorted({
            p for p, sub in lab.iloc[sel].groupby("patient")
            if sum(per_sample_loaded.get(s, {}).get("n_in_mtx", 0)
                   for s in sub["sample"].unique()) == 0}),
        "genes_detected_ge5_cells": int(len(gene_keep)),
        "n_hvg_used": int(n_hvg),
        "n_pca_components": int(n_comp),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "or_edges_computed": int(n_edges),
        "or_seconds": float(time.time() - t0),
    }
    return df, extras


def per_patient_deltas(df, comp_label):
    rows = []
    df = df.dropna(subset=["kappa"])
    for pid, sub in df.groupby("patient"):
        a = sub.loc[sub.is_mal, "kappa"].values
        b = sub.loc[~sub.is_mal, "kappa"].values
        if len(a) < MIN_CELLS_PER_GROUP or len(b) < MIN_CELLS_PER_GROUP:
            continue
        delta, se = cliff_delta_se(a, b)
        tt = ttest_ind(a, b, equal_var=False)
        rows.append(dict(patient_id=str(pid),
                         n_malignant=int(len(a)), n_comparator=int(len(b)),
                         mean_kappa_mal=float(a.mean()),
                         mean_kappa_comp=float(b.mean()),
                         delta=float(delta), se=float(se),
                         ci_lo=float(delta - 1.96 * se),
                         ci_hi=float(delta + 1.96 * se),
                         welch_p=float(tt.pvalue)))
    return pd.DataFrame(rows)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


VARIANT_RULE_TEXT = {
    "excluded": ("baseline (pre-registered) rule: dual-labeled barcodes "
                 "dropped from the primary arm entirely"),
    "dual_mal": ("dual-labeled barcodes kept and counted as malignant "
                 "(immune row dropped)"),
    "dual_immune": ("dual-labeled barcodes kept and counted as immune "
                    "(malignant row dropped)"),
}


def summarize_variant(variant, df, extras):
    pp = per_patient_deltas(df, COMP_LABEL)
    print(f"  patients passing >={MIN_CELLS_PER_GROUP}/"
          f"{MIN_CELLS_PER_GROUP} filter (valid kappa): {len(pp)}")
    if len(pp):
        print(pp[["patient_id", "n_malignant", "n_comparator",
                  "delta", "se", "ci_lo", "ci_hi"]].to_string(index=False))

    valid = df.dropna(subset=["kappa"])
    a_all = valid.loc[valid.is_mal, "kappa"].values
    b_all = valid.loc[~valid.is_mal, "kappa"].values
    naive_delta = cliffs_delta_fast(a_all, b_all)
    naive_t = ttest_ind(a_all, b_all, equal_var=False)
    print(f"  NAIVE cell-level (pseudorep): delta={naive_delta:+.4f} "
          f"Welch p={naive_t.pvalue:.3g}  n={len(a_all)}/{len(b_all)}")

    meta = dersimonian_laird(pp["delta"].values, pp["se"].values,
                             label=f"{COHORT}_primary_sens_{variant}")
    meta["one_sided_p_delta_gt_0"] = float(norm.sf(meta["z"]))
    boot = patient_clustered_bootstrap(pp["delta"].values, pp["se"].values)
    print(f"  DL pooled delta={meta['delta']:+.6f} "
          f"[{meta['ci_lo']:+.6f},{meta['ci_hi']:+.6f}]  "
          f"tau2={meta['tau2']:.5f}  I2={meta['I2']:.1f}%  "
          f"Q={meta['Q']:.2f}(df={meta['df']}) pQ={meta['pQ']:.3g}")
    print(f"  z={meta['z']:.3f}  two-sided p={meta['p']:.3g}  "
          f"one-sided p(delta>0)={meta['one_sided_p_delta_gt_0']:.3g}")
    print(f"  bootstrap (B={boot['B']}): mean={boot['mean']:+.4f} "
          f"CI=[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}]")

    n_pos = int((pp["delta"] > 0).sum()) if len(pp) else 0
    summary = {
        "dual_label_rule": VARIANT_RULE_TEXT[variant],
        "pipeline_facts": _clean(extras),
        "n_cells_sampled_with_valid_kappa": int(len(valid)),
        "n_malignant_cells_valid": int(len(a_all)),
        "n_comparator_cells_valid": int(len(b_all)),
        "n_patients_passing_filter": int(len(pp)),
        "min_patients_criterion_met": bool(len(pp) >= MIN_PATIENTS),
        "dl_meta": _clean(meta),
        "pooled_delta": _clean(meta["delta"]),
        "pooled_se": _clean(meta["se"]),
        "pooled_ci_95": [_clean(meta["ci_lo"]), _clean(meta["ci_hi"])],
        "i_squared_percent": _clean(meta["I2"]),
        "one_sided_p_delta_gt_0": _clean(meta["one_sided_p_delta_gt_0"]),
        "two_sided_p": _clean(meta["p"]),
        "per_patient_direction": {
            "n_delta_positive": n_pos,
            "n_delta_negative": int(len(pp) - n_pos),
            "n_total": int(len(pp)),
            "fraction_positive": float(n_pos / len(pp)) if len(pp) else None,
        },
        "per_patient_delta_compact": {
            str(r.patient_id): {"delta": _clean(r.delta),
                                "n_malignant": int(r.n_malignant),
                                "n_comparator": int(r.n_comparator)}
            for r in pp.itertuples()
        },
        "naive_cell_level": {
            "delta": _clean(naive_delta),
            "welch_p": float(naive_t.pvalue),
        },
        "bootstrap": _clean(boot),
    }
    return summary


def main():
    t_start = time.time()
    print("=" * 72)
    print(f"{COHORT} PRIMARY arm -- POST-HOC sensitivity of dual-label handling")
    print(f"  variants: {VARIANT_ORDER}")
    print(f"  seed={SEED}  cap/group={CAP_PER_PATIENT_GROUP}  "
          f"total_cap={TOTAL_CAP}  min/group={MIN_CELLS_PER_GROUP}")
    print(f"  OR alpha={ALPHA}  edges target={N_EDGES_TARGET}  "
          f"bootstrap B={B_BOOT}")
    print("  'excluded' MUST reproduce results.json pooled delta "
          f"{EXPECTED_BASELINE['pooled_delta']!r} exactly or the run ABORTS")
    print("=" * 72)

    mapping = load_mapping()

    variant_results = {}
    gate = None
    for variant in VARIANT_ORDER:
        print("-" * 72)
        print(f"VARIANT: {variant} (malignant vs {COMP_LABEL})")
        print("-" * 72, flush=True)
        df, extras = compute_arm(variant, mapping)
        summary = summarize_variant(variant, df, extras)

        if variant == "excluded":
            obs = summary
            gate = {
                "expected_pooled_delta": EXPECTED_BASELINE["pooled_delta"],
                "observed_pooled_delta": obs["pooled_delta"],
                "pooled_delta_exact_match":
                    obs["pooled_delta"] == EXPECTED_BASELINE["pooled_delta"],
                "se_exact_match":
                    obs["pooled_se"] == EXPECTED_BASELINE["se"],
                "ci_exact_match":
                    obs["pooled_ci_95"] == [EXPECTED_BASELINE["ci_lo"],
                                            EXPECTED_BASELINE["ci_hi"]],
                "one_sided_p_exact_match":
                    obs["one_sided_p_delta_gt_0"]
                    == EXPECTED_BASELINE["one_sided_p"],
                "i2_exact_match":
                    obs["i_squared_percent"] == EXPECTED_BASELINE["i2"],
                "k_exact_match":
                    obs["n_patients_passing_filter"] == EXPECTED_BASELINE["k"],
                "direction_counts_exact_match": (
                    obs["per_patient_direction"]["n_delta_positive"]
                    == EXPECTED_BASELINE["n_delta_positive"]
                    and obs["per_patient_direction"]["n_delta_negative"]
                    == EXPECTED_BASELINE["n_delta_negative"]),
                "bootstrap_exact_match": (
                    obs["bootstrap"]["mean"]
                    == EXPECTED_BASELINE["bootstrap_mean"]
                    and obs["bootstrap"]["ci_lo"]
                    == EXPECTED_BASELINE["bootstrap_ci_lo"]
                    and obs["bootstrap"]["ci_hi"]
                    == EXPECTED_BASELINE["bootstrap_ci_hi"]),
            }
            if not gate["pooled_delta_exact_match"]:
                print("=" * 72)
                print("BASELINE REPRODUCTION FAILED -- STOPPING (no "
                      "results_sensitivity.json written)")
                print(f"  expected pooled delta: "
                      f"{EXPECTED_BASELINE['pooled_delta']!r}")
                print(f"  observed pooled delta: "
                      f"{obs['pooled_delta']!r}")
                print(f"  difference: "
                      f"{obs['pooled_delta'] - EXPECTED_BASELINE['pooled_delta']!r}")
                print(json.dumps(gate, indent=2))
                sys.exit(1)
            print("  [gate] baseline reproduced exactly "
                  f"(pooled delta {obs['pooled_delta']!r}) -- proceeding")

        variant_results[variant] = summary

    results = {
        "header": {
            "analysis": ("post-hoc sensitivity analysis of dual-label "
                         "handling; NOT part of PREREGISTRATION.md; baseline "
                         "variant is the pre-registered run"),
            "cohort": "GSE161529 (Pal et al. 2021 breast cancer atlas)",
            "arm": "primary only (malignant vs immune)",
            "question": ("does the pre-registered primary-arm pooled delta "
                         "depend on excluding the 26,439 dual-labeled "
                         "barcodes (labeled both 'malignant' under "
                         "Total-object inferCNV clusters and 'immune' under "
                         "Sub-object clusters in cells_primary.tsv.gz)?"),
            "variants": {
                "excluded": VARIANT_RULE_TEXT["excluded"],
                "dual_mal": VARIANT_RULE_TEXT["dual_mal"],
                "dual_immune": VARIANT_RULE_TEXT["dual_immune"],
            },
            "held_identical_to_preregistered_run": (
                "seed 20260507; stratified caps 60/patient/group and total "
                "3500; HVG-2000, PCA-50, kNN k=15, Ollivier-Ricci alpha=0.5 "
                "(POT) on 4000 edges; >=10 cells/group filter; "
                "DerSimonian-Laird meta; patient-clustered bootstrap B=1000; "
                "only the dual-label assignment rule differs"),
            "baseline_source": ("results.json comparators.primary "
                                "(pre-registered run, PREREGISTRATION.md "
                                "2026-06-29)"),
            "seed": SEED,
        },
        "baseline_reproduction": gate,
        "variants": variant_results,
        "wallclock_seconds": float(time.time() - t_start),
    }
    with open(HERE / "results_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {HERE / 'results_sensitivity.json'}")
    print(f"Wallclock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
