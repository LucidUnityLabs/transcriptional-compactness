"""
HELDOUT_GSE131907 -- Pre-registered held-out cohort validation (SECONDARY).

Per PREREGISTRATION.md (2026-06-29, LOCKED):
  Secondary cohort GSE131907 (Kim et al. 2020 metastatic LUAD atlas).
  Annotation: original-author cell_annotation.txt.
  Comparator: immune cells.  Prediction: delta > 0.

  Pipeline (locked): HVG-2000 -> PCA-50 -> kNN k=15 (Euclidean) ->
  Ollivier-Ricci alpha=0.5 (POT ot.emd2, 4000 sampled edges or all if fewer)
  -> per-patient Cliff's delta (>=10 malignant AND >=10 comparator cells) ->
  DerSimonian-Laird random-effects meta -> pooled delta, 95% CI, I2,
  one-sided p (delta > 0).

  Decision rule (applied OUTSIDE this script; raw numbers only reported):
  SUPPORTED if pooled delta > 0 and one-sided p < 0.05; NOT SUPPORTED otherwise.

Conventions replicated from exp/META_patient_level/run.py and exp/E1_within_patient/run.py:
  seed 20260507; stratified sample cap 60/patient/group, total cap 3500;
  OR primitives verbatim from E1; Cliff's delta + U-statistic (Hoeffding
  projection) SE; DL meta identical; patient-clustered bootstrap B=1000.

Data loading replicated from exp/V2_metastasis/run.py (streaming reader for
the 208k-column genes-x-cells UMI TSV; log1p-CPM(1e6); gene filter >=5 cells).

Outputs (in this directory):
  results.json, per_patient_delta.csv, run.log
  cache/HELDOUT_GSE131907_kappa.npz
"""

import gzip
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import ot
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.stats import chi2, norm, ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
KIM_DIR = DATA / "GSE131907_kim_nsclc"
KIM_MTX = KIM_DIR / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
KIM_ANN = KIM_DIR / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

SEED = 20260507

# locked parameters (PREREGISTRATION.md)
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

# original-author annotation vocabulary:
#   malignant = Epithelial cells with Cell_subtype in {tS1,tS2,tS3} (primary
#   tLung tumour annotation) or 'Malignant cells' (metastatic-site annotation);
#   union is the original-author malignant call (cf. exp/R5 and exp/V2 usage).
MAL_SUBTYPES = {"tS1", "tS2", "tS3", "Malignant cells"}
IMMUNE_REFINED = {"T/NK cells", "Myeloid cells", "B lymphocytes", "MAST cells"}


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
# Effect sizes (copied verbatim from exp/META_patient_level/run.py)
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
# DerSimonian-Laird random-effects meta (copied verbatim from META)
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
# Stratified sampling (copied verbatim from exp/META_patient_level/run.py)
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
# GSE131907 loaders (annotation + streaming matrix, per exp/V2_metastasis)
# ===========================================================================
def load_kim_annotation():
    print(f"  reading {KIM_ANN.name} ...", flush=True)
    df = pd.read_csv(KIM_ANN, sep="\t", index_col=0, dtype=str)
    print(f"  annotation: {df.shape}, columns={list(df.columns)}", flush=True)
    return df


def load_kim_matrix(keep_cells_set):
    print(f"  reading {KIM_MTX.name} (streaming) ...", flush=True)
    t0 = time.time()
    with gzip.open(KIM_MTX, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        cell_cols = header[1:]
        print(f"  matrix header: {len(cell_cols)} cell columns", flush=True)
        keep_idx = np.array([i for i, c in enumerate(cell_cols)
                             if c in keep_cells_set], dtype=np.int64)
        print(f"  cells to keep from matrix: {len(keep_idx)}", flush=True)
        cell_order = [cell_cols[i] for i in keep_idx]
        n_total_cols = len(cell_cols)

        gene_names = []
        data = []
        idxs = []
        indptr = [0]
        n_lines = 0
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tab_pos = line.find("\t")
            if tab_pos < 0:
                continue
            gene_names.append(line[:tab_pos])
            vals_str = line[tab_pos + 1:]
            row_full = np.fromstring(vals_str, dtype=np.float32, sep="\t")
            if row_full.shape[0] != n_total_cols:
                if row_full.shape[0] < n_total_cols:
                    pad = np.zeros(n_total_cols - row_full.shape[0],
                                   dtype=np.float32)
                    row_full = np.concatenate([row_full, pad])
                else:
                    row_full = row_full[:n_total_cols]
            row_vals = row_full[keep_idx]
            nz = np.where(row_vals > 0)[0]
            if len(nz) > 0:
                data.append(row_vals[nz])
                idxs.append(nz.astype(np.int32))
            indptr.append(indptr[-1] + len(nz))
            n_lines += 1
            if n_lines % 4000 == 0:
                print(f"    {n_lines} genes parsed ({time.time()-t0:.1f}s)",
                      flush=True)

    print(f"  parsed {n_lines} genes in {time.time()-t0:.1f}s", flush=True)
    n_genes = len(gene_names)
    n_cells = len(cell_order)
    if data:
        data_arr = np.concatenate(data)
        idx_arr = np.concatenate(idxs)
    else:
        data_arr = np.zeros(0, dtype=np.float32)
        idx_arr = np.zeros(0, dtype=np.int32)
    indptr_arr = np.array(indptr, dtype=np.int64)
    counts_gxc = csr_matrix((data_arr, idx_arr, indptr_arr),
                            shape=(n_genes, n_cells))
    counts = counts_gxc.T.tocsr()      # cells x genes
    print(f"  counts (cells x genes): {counts.shape}, nnz={counts.nnz}",
          flush=True)
    return counts, np.array(gene_names), np.array(cell_order)


# ===========================================================================
# kappa computation with disk cache (naming per META convention)
# ===========================================================================
COHORT = "HELDOUT_GSE131907"


def compute_kappa(force=False):
    cache_path = CACHE / f"{COHORT}_kappa.npz"
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=True)
        df = pd.DataFrame({"patient": z["patient"], "is_mal": z["is_mal"],
                           "kappa": z["kappa"]})
        print(f"  [cache hit] {len(df)} cells loaded from {cache_path.name}")
        return df

    ann = load_kim_annotation()
    is_mal = ((ann["Cell_type.refined"] == "Epithelial cells")
              & ann["Cell_subtype"].isin(MAL_SUBTYPES)).values
    is_comp = ann["Cell_type.refined"].isin(IMMUNE_REFINED).values
    patient = ann["Sample"].astype(str).values
    print(f"  annotation totals: malignant={int(is_mal.sum())}, "
          f"immune={int(is_comp.sum())}, "
          f"samples={len(np.unique(patient))}")

    # per-patient qualifying counts (pre-sampling)
    pc = pd.DataFrame({"patient": patient, "is_mal": is_mal,
                       "is_comp": is_comp})
    qual = 0
    for pid in np.unique(patient):
        m = ((pc.patient == pid) & pc.is_mal).sum()
        c = ((pc.patient == pid) & pc.is_comp).sum()
        if m >= MIN_CELLS_PER_GROUP and c >= MIN_CELLS_PER_GROUP:
            qual += 1
    print(f"  samples with >={MIN_CELLS_PER_GROUP}/{MIN_CELLS_PER_GROUP} "
          f"(mal/immune) cells pre-sampling: {qual}/{len(np.unique(patient))}")

    sel = stratified_indices(patient, is_mal, is_comp)
    sel_barcodes = ann.index[sel]
    pat_s = patient[sel]
    mal_s = is_mal[sel]
    print(f"  stratified sample: {len(sel)} cells "
          f"(mal={int(mal_s.sum())}, comparator={int((~mal_s).sum())}), "
          f"{len(np.unique(pat_s))} patients")

    counts, gene_names, cell_order = load_kim_matrix(set(sel_barcodes))

    # align annotation to matrix cell order
    ann_kept = ann.loc[cell_order]
    pat_m = ann_kept["Sample"].astype(str).values
    mal_m = ((ann_kept["Cell_type.refined"] == "Epithelial cells")
             & ann_kept["Cell_subtype"].isin(MAL_SUBTYPES)).values

    # log1p-CPM(1e6), gene filter >=5 cells (exp/V2 + R5 convention)
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
    del Xm, sq, norm

    print(f"  building kNN(k={K_NN}) ...")
    G = build_knn_graph(Xpca, k=K_NN)
    n_edges = min(N_EDGES_TARGET, G.number_of_edges())
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"computing OR on {n_edges} edges ...")
    t0 = time.time()
    edge_k = ollivier_ricci_edges(G, alpha=ALPHA, n_edges=n_edges,
                                  rng=np.random.default_rng(SEED), verbose=True)
    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa = np.array([per_cell.get(i, np.nan) for i in range(len(pat_m))],
                     dtype=float)
    n_valid = int(np.sum(np.isfinite(kappa)))
    print(f"  OR done in {time.time()-t0:.1f}s; valid-kappa cells: "
          f"{n_valid}/{len(pat_m)} ({100*n_valid/len(pat_m):.1f}%)")

    df = pd.DataFrame({"patient": pat_m, "is_mal": mal_m, "kappa": kappa})
    np.savez(cache_path, patient=pat_m, is_mal=mal_m, kappa=kappa)
    print(f"  cached -> {cache_path.name}")
    return df


def per_patient_deltas(df):
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


def main():
    t_start = time.time()
    print("=" * 72)
    print("HELDOUT_GSE131907 -- pre-registered secondary held-out validation")
    print(f"  seed={SEED}  cap/group={CAP_PER_PATIENT_GROUP}  "
          f"total_cap={TOTAL_CAP}  min/group={MIN_CELLS_PER_GROUP}")
    print(f"  OR alpha={ALPHA}  edges target={N_EDGES_TARGET}  "
          f"bootstrap B={B_BOOT}")
    print("=" * 72)

    df = compute_kappa()
    pp = per_patient_deltas(df)
    print(f"  patients passing >={MIN_CELLS_PER_GROUP}/{MIN_CELLS_PER_GROUP} "
         f"filter (valid kappa): {len(pp)}")
    if len(pp):
        print(pp[["patient_id", "n_malignant", "n_comparator",
                  "delta", "se", "ci_lo", "ci_hi"]].to_string(index=False))

    # naive cell-level delta (pseudorep, for reference; same as META reports)
    valid = df.dropna(subset=["kappa"])
    a_all = valid.loc[valid.is_mal, "kappa"].values
    b_all = valid.loc[~valid.is_mal, "kappa"].values
    naive_delta = cliffs_delta_fast(a_all, b_all)
    naive_t = ttest_ind(a_all, b_all, equal_var=False)
    print(f"  NAIVE cell-level (pseudorep): delta={naive_delta:+.4f} "
          f"Welch p={naive_t.pvalue:.3g}  n={len(a_all)}/{len(b_all)}")

    meta = dersimonian_laird(pp["delta"].values, pp["se"].values,
                             label=COHORT)
    meta["one_sided_p_delta_gt_0"] = float(norm.sf(meta["z"]))
    boot = patient_clustered_bootstrap(pp["delta"].values, pp["se"].values)
    print(f"  DL pooled delta={meta['delta']:+.4f} "
          f"[{meta['ci_lo']:+.4f},{meta['ci_hi']:+.4f}]  "
          f"tau2={meta['tau2']:.5f}  I2={meta['I2']:.1f}%  "
          f"Q={meta['Q']:.2f}(df={meta['df']}) pQ={meta['pQ']:.3g}")
    print(f"  z={meta['z']:.3f}  two-sided p={meta['p']:.3g}  "
          f"one-sided p(delta>0)={meta['one_sided_p_delta_gt_0']:.3g}")
    print(f"  bootstrap (B={boot['B']}): mean={boot['mean']:+.4f} "
          f"CI=[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}]")

    pp.to_csv(HERE / "per_patient_delta.csv", index=False)

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

    n_pos = int((pp["delta"] > 0).sum()) if len(pp) else 0
    results = {
        "cohort": "GSE131907 (Kim et al. 2020 metastatic LUAD atlas)",
        "analysis": "pre-registered secondary held-out validation "
                    "(PREREGISTRATION.md 2026-06-29)",
        "prereg_locked_params": {
            "pca_dims": N_PCA, "hvg": N_HVG, "knn_k": K_NN,
            "or_alpha": ALPHA, "edge_sampling": N_EDGES_TARGET,
            "min_cells_per_patient_per_group": MIN_CELLS_PER_GROUP,
            "min_patients_per_cohort": MIN_PATIENTS,
            "meta_method": "DerSimonian-Laird random-effects",
            "significance": "one-sided p < 0.05",
        },
        "labels": {
            "annotation": "original-author cell_annotation.txt",
            "malignant": "Cell_type.refined=='Epithelial cells' & "
                         "Cell_subtype in {tS1,tS2,tS3,Malignant cells}",
            "comparator": "immune cells: Cell_type.refined in "
                          "{T/NK cells, Myeloid cells, B lymphocytes, "
                          "MAST cells}",
            "patient_unit": "Sample (annotation's only patient-of-origin "
                            "proxy; 58 samples / 44 patients in paper)",
        },
        "seed": SEED,
        "design": {
            "unit_of_analysis": "patient (sample)",
            "effect_size": "Cliff's delta (malignant vs immune kappa, "
                           "within patient)",
            "delta_variance": "U-statistic Hoeffding projection "
                              "(row/col mean variance)",
            "meta_method": "DerSimonian-Laird random-effects "
                           "(inverse-variance)",
            "bootstrap": f"patient-clustered, B={B_BOOT}",
            "min_cells_per_group": MIN_CELLS_PER_GROUP,
            "pipeline": f"HVG-{N_HVG}, PCA-{N_PCA}, kNN k={K_NN}, "
                        f"Ollivier-Ricci alpha={ALPHA} (POT), "
                        f"{N_EDGES_TARGET} edges",
            "cap_per_patient_group": CAP_PER_PATIENT_GROUP,
            "total_cap_per_cohort": TOTAL_CAP,
            "normalization": "log1p-CPM(1e6); genes detected in >=5 cells "
                             "(dataset loading convention from exp/V2, R5)",
        },
        "n_cells_sampled_with_valid_kappa": int(len(valid)),
        "n_malignant_cells_valid": int(len(a_all)),
        "n_comparator_cells_valid": int(len(b_all)),
        "n_patients_passing_filter": int(len(pp)),
        "min_patients_criterion_met": bool(len(pp) >= MIN_PATIENTS),
        "per_patient_cliffs_delta": {
            str(r.patient_id): _clean({
                "n_malignant": int(r.n_malignant),
                "n_comparator": int(r.n_comparator),
                "mean_kappa_mal": r.mean_kappa_mal,
                "mean_kappa_comp": r.mean_kappa_comp,
                "delta": r.delta, "se": r.se,
                "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                "welch_p": r.welch_p,
            }) for r in pp.itertuples()
        },
        "per_patient_direction": {
            "n_delta_positive": n_pos,
            "n_delta_negative": int(len(pp) - n_pos),
            "n_total": int(len(pp)),
            "fraction_positive": float(n_pos / len(pp)) if len(pp) else None,
        },
        "dl_meta": _clean(meta),
        "pooled_delta": _clean(meta["delta"]),
        "pooled_ci_95": [_clean(meta["ci_lo"]), _clean(meta["ci_hi"])],
        "i_squared_percent": _clean(meta["I2"]),
        "one_sided_p_delta_gt_0": _clean(meta["one_sided_p_delta_gt_0"]),
        "two_sided_p": _clean(meta["p"]),
        "naive_cell_level": {
            "delta": _clean(naive_delta),
            "welch_p": float(naive_t.pvalue),
        },
        "bootstrap": _clean(boot),
        "decision_rule_inputs": {
            "pooled_delta": _clean(meta["delta"]),
            "one_sided_p": _clean(meta["one_sided_p_delta_gt_0"]),
            "rule": "SUPPORTED if pooled delta > 0 and one-sided p < 0.05; "
                    "NOT SUPPORTED otherwise (applied outside this file)",
        },
        "wallclock_seconds": float(time.time() - t_start),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {HERE / 'results.json'}")
    print(f"Wrote: {HERE / 'per_patient_delta.csv'}  ({len(pp)} patients)")
    print(f"Wrote: {CACHE / (COHORT + '_kappa.npz')}")
    print(f"Wallclock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
