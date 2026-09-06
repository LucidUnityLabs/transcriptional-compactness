"""
N1 — Per-cell Ollivier-Ricci kappa as ICB-response biomarker
============================================================
Sade-Feldman et al. 2018 (Cell 175:998), GSE120575.

Hypothesis: Per-cell kappa at *baseline* (Pre-treatment) on tumour-infiltrating
CD8+ T cells differs between cells from patients who later become Responders vs
Non-responders.

Procedure:
  1. Load TPM matrix + per-cell metadata. Subset to Pre samples.
  2. Score CD8+ T cells via CD8A / CD8B / CD3D / CD3E expression.
  3. HVG-2000 -> log -> centre -> PCA-50 jointly across all baseline CD8 cells.
  4. kNN k=15, Ollivier-Ricci alpha=0.5, ~4000 sampled edges.
  5. Per-cell mean kappa: Cliff's delta R vs NR + Welch t + MWU.
  6. Per-patient mean kappa: MWU + AUC (single-feature classifier).
  7. Logistic regression responder ~ mean_kappa + n_cells.
"""
import sys, gzip, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data"
SEED = 20260508
RNG = np.random.default_rng(SEED)

EXPR_GZ = DATA / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
META_GZ = DATA / "GSE120575_patient_ID_single_cells.txt.gz"


# ---------------- metadata ----------------
def load_metadata():
    """Per-cell metadata table indexed by cell title (column header in TPM)."""
    rows = []
    with gzip.open(META_GZ, "rt", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            if not parts[0].startswith("Sample "):
                continue
            # parts[1] = cell title (e.g. A10_P3_M11)
            # parts[4] = patient_timepoint (e.g. Pre_P1)
            # parts[5] = response  (Responder / Non-responder)
            # parts[6] = therapy   (anti-PD1 / anti-CTLA4 / anti-CTLA4+PD1)
            cell_id = parts[1]
            tag = parts[4]
            if "_P" not in tag:
                continue
            timepoint, patient = tag.split("_", 1)
            rows.append(dict(cell_id=cell_id, patient_id=patient,
                             timepoint=timepoint,
                             response=parts[5], therapy=parts[6]))
    df = pd.DataFrame(rows)
    return df


# ---------------- TPM streaming ----------------
def load_tpm_for_cells(cell_ids, marker_genes):
    """
    Stream the TPM file once. For each gene row, store the values restricted
    to selected cells. Also collect per-cell statistics (variance, mean log)
    needed for HVG selection without materialising the full matrix.
    Returns (gene_names_kept, X[g,n], cell_idx_map, marker_expr).
    Strategy:
      pass 1: read header, find column indices for cell_ids,
              compute log1p mean & sumsq per gene -> variance
              accumulate marker rows separately
      then: pick HVG = top-2000 variance,
              pass 2 across same cached file -> only HVG rows
    Implementation note: for compactness we just stream once and keep all
    selected-cell rows in memory as float32 (rows ~55k, n cells ~few-thousand
    after CD8 filter is the goal but we don't have CD8 calls yet, so first
    pass keeps all 5928 baseline cells; ~55000*5928*4 = 1.3 GB -- too big).

    Better: pass 1 -> compute variance + collect marker rows,
            pass 2 -> only HVG rows. This script is single-pass: collect
            log1p, variance via Welford as we go, BUT only keep the
            marker-gene rows + a rolling log buffer for HVG candidates.
    """
    pass  # not used; see streaming pipeline below


def stream_tpm_matrix(cell_ids_set, marker_set, n_hvg=2000):
    """
    Two-pass streaming reader for the gzipped TPM file.

    Pass 1: read header -> column indices of selected cells.
            Stream all gene rows, for each gene:
                - compute variance (after log1p) over selected cells
                - if gene in marker_set, store the row (log1p)
            -> top-n_hvg gene names by variance.
    Pass 2: re-stream gene rows, store only HVG rows (log1p) into matrix.

    Returns:
      hvg_genes: list[str]
      Xlog:      ndarray (n_hvg, n_cells) float32, log1p(TPM)
      cell_cols: list[str] in order they appear in the file (cell IDs of
                 selected cells, ordered by file column index)
      marker_expr: dict gene -> ndarray of log1p over selected cells (in
                   same column order as cell_cols)
    """
    print(f"  TPM file: {EXPR_GZ}")
    # ---- header pass ----
    with gzip.open(EXPR_GZ, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    # header[0] is empty (gene-name column), header[1:] are cell IDs.
    # The gene rows have one *extra* trailing empty field. So if a gene row
    # has F fields, the cell values are parts[1:1+n_cells].
    # We strip trailing empty header tokens too, just in case.
    while header and header[-1] == "":
        header.pop()
    all_cells = header[1:]
    n_all = len(all_cells)
    sel_idx = [i for i, c in enumerate(all_cells) if c in cell_ids_set]
    cell_cols = [all_cells[i] for i in sel_idx]
    sel_idx_arr = np.array(sel_idx, dtype=np.int64)
    print(f"  selected {len(cell_cols)} of {n_all} cells from header")

    # ---- pass 1: variance per gene + marker rows ----
    print("  pass 1: computing variances + marker rows ...")
    gene_var = []
    gene_names = []
    marker_expr = {}
    n_cells = len(cell_cols)
    t0 = time.time()
    with gzip.open(EXPR_GZ, "rt") as f:
        f.readline()  # header (cells)
        f.readline()  # second row: patient/timepoint string per cell - skip
        for ln, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene = parts[0]
            # take only the n_all expected cell columns; ignore trailing empties
            cell_strs = parts[1:1 + n_all]
            if len(cell_strs) != n_all:
                continue
            try:
                vals = np.array(cell_strs, dtype=np.float32)
            except ValueError:
                continue
            sub = vals[sel_idx_arr]
            sub = np.log1p(sub)
            v = float(sub.var())
            gene_var.append(v)
            gene_names.append(gene)
            if gene in marker_set:
                marker_expr[gene] = sub.astype(np.float32)
            if ln % 5000 == 0:
                print(f"    pass1 row {ln}  elapsed={time.time()-t0:.1f}s")
    gene_var = np.array(gene_var)
    print(f"  pass 1 done: {len(gene_names)} genes, t={time.time()-t0:.1f}s")

    order = np.argsort(gene_var)[::-1]
    hvg_idx = order[:n_hvg]
    hvg_set = set(gene_names[i] for i in hvg_idx)
    hvg_genes_ordered = []
    print(f"  HVG-{n_hvg}: top variance range "
          f"{gene_var[hvg_idx[0]]:.3f} ... {gene_var[hvg_idx[-1]]:.3f}")

    # ---- pass 2: collect HVG rows ----
    print("  pass 2: collecting HVG rows ...")
    Xlog = np.zeros((n_hvg, n_cells), dtype=np.float32)
    g_pos = 0
    t0 = time.time()
    with gzip.open(EXPR_GZ, "rt") as f:
        f.readline(); f.readline()
        for ln, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene = parts[0]
            if gene not in hvg_set:
                continue
            cell_strs = parts[1:1 + n_all]
            if len(cell_strs) != n_all:
                continue
            try:
                vals = np.array(cell_strs, dtype=np.float32)
            except ValueError:
                continue
            sub = np.log1p(vals[sel_idx_arr])
            Xlog[g_pos] = sub
            hvg_genes_ordered.append(gene)
            g_pos += 1
            if g_pos % 500 == 0:
                print(f"    pass2 hvg {g_pos}/{n_hvg}  t={time.time()-t0:.1f}s")
    Xlog = Xlog[:g_pos]
    print(f"  pass 2 done: {g_pos} HVGs, t={time.time()-t0:.1f}s")
    return hvg_genes_ordered, Xlog, cell_cols, marker_expr


# ---------------- Ollivier-Ricci ----------------
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
    for k_edge, (u, v) in enumerate(edges):
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
        ok = True
        for i, a in enumerate(sup_u):
            for j, b in enumerate(sup_v):
                d = dists[a].get(b, np.inf)
                if not np.isfinite(d):
                    ok = False
                    break
                C[i, j] = d
            if not ok:
                break
        if not ok:
            out[(u, v)] = np.nan
            continue
        W = ot.emd2(w_u, w_v, C)
        d_uv = G[u][v]["weight"]
        out[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan
        if (k_edge + 1) % 500 == 0:
            print(f"      OR edge {k_edge+1}/{len(edges)}")
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


# ---------------- statistics ----------------
def cliffs_delta(a, b):
    """Cliff's delta = (#a>b - #a<b) / (na*nb).  Vectorised O(na*nb)."""
    a = np.asarray(a); b = np.asarray(b)
    na = len(a); nb = len(b)
    # chunked to keep memory bounded
    pos = 0; neg = 0
    chunk = 2000
    for i in range(0, na, chunk):
        ai = a[i:i+chunk][:, None]
        pos += int(np.sum(ai > b[None, :]))
        neg += int(np.sum(ai < b[None, :]))
    return (pos - neg) / (na * nb)


# ---------------- main ----------------
def main():
    print("=" * 64)
    print("N1 Sade-Feldman 2018 GSE120575 -- per-cell kappa as ICB biomarker")
    print(f"seed={SEED}")
    print("=" * 64)

    # 1. metadata
    print("\n[1] loading per-cell metadata ...")
    meta = load_metadata()
    print(f"  metadata rows: {len(meta)}")
    print(f"  unique patients: {meta['patient_id'].nunique()}")
    print(f"  timepoints: {meta['timepoint'].value_counts().to_dict()}")

    pre = meta[meta["timepoint"] == "Pre"].copy()
    pre_cells = set(pre["cell_id"])
    print(f"  Pre-treatment cells in metadata: {len(pre_cells)}")
    print(f"  Pre patients: {pre['patient_id'].nunique()}")

    # 2. stream TPM, restrict to Pre cells, collect HVG and marker rows
    print("\n[2] streaming TPM for Pre cells ...")
    markers = {"CD8A", "CD8B", "CD4", "CD3D", "CD3E", "CD3G",
               "MS4A1", "CD19", "FOXP3", "NCAM1", "FCGR3A"}
    hvg, Xlog, cell_cols, marker_expr = stream_tpm_matrix(
        pre_cells, markers, n_hvg=2000)
    print(f"  HVG matrix: {Xlog.shape}, cells: {len(cell_cols)}")
    print(f"  markers found: {sorted(marker_expr.keys())}")

    # 3. CD8 T-cell scoring
    print("\n[3] scoring CD8+ T cells via CD8A/CD8B vs CD3 panel ...")
    def get(g):
        return marker_expr.get(g, np.zeros(len(cell_cols), dtype=np.float32))
    cd8a = get("CD8A"); cd8b = get("CD8B")
    cd3d = get("CD3D"); cd3e = get("CD3E"); cd3g = get("CD3G")
    cd4  = get("CD4");  foxp3= get("FOXP3")
    ms4a1= get("MS4A1"); cd19 = get("CD19")

    cd8_score = np.maximum(cd8a, cd8b)
    cd3_score = np.maximum.reduce([cd3d, cd3e, cd3g])
    # CD8 T cell call: CD8 > thr, CD3 expressed, low B-cell / CD4-Treg
    thr = 1.0  # log1p(TPM); roughly TPM>1.7
    is_cd8 = (cd8_score > thr) & (cd3_score > thr) & (ms4a1 < 1.0) & (cd19 < 1.0)
    # exclude obvious Tregs
    is_cd8 = is_cd8 & ~((foxp3 > 1.0) & (cd4 > cd8_score))
    n_cd8 = int(is_cd8.sum())
    print(f"  CD8 T cells: {n_cd8} / {len(cell_cols)} "
          f"({100*n_cd8/len(cell_cols):.1f}%)")

    cd8_idx = np.where(is_cd8)[0]
    cell_cols_cd8 = [cell_cols[i] for i in cd8_idx]
    Xlog_cd8 = Xlog[:, cd8_idx]
    print(f"  CD8 expression matrix: {Xlog_cd8.shape}")

    # join per-cell metadata
    meta_pre = pre.set_index("cell_id")
    rows = []
    missing = 0
    for c in cell_cols_cd8:
        if c not in meta_pre.index:
            missing += 1
            continue
        r = meta_pre.loc[c]
        rows.append(dict(cell_id=c, patient_id=r["patient_id"],
                         response=r["response"], therapy=r["therapy"]))
    cell_meta = pd.DataFrame(rows).set_index("cell_id")
    print(f"  joined metadata for {len(cell_meta)} CD8 cells (missing: {missing})")

    # 4. PCA + kNN + Ollivier-Ricci
    print("\n[4] PCA-50 + kNN k=15 ...")
    Xc = Xlog_cd8.T.astype(np.float32)            # (n_cells, n_genes)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n_pcs = min(50, min(Xc.shape) - 1)
    Xpca = PCA(n_components=n_pcs, random_state=SEED).fit_transform(Xc)
    print(f"  PCA shape: {Xpca.shape}")

    G = build_knn_graph(Xpca, k=15)
    print(f"  kNN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    n_edges = min(4000, G.number_of_edges())
    print(f"\n[5] Ollivier-Ricci alpha=0.5 on {n_edges} sampled edges ...")
    t0 = time.time()
    ek = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=RNG)
    n_valid = sum(1 for v in ek.values() if not np.isnan(v))
    print(f"  valid edges: {n_valid}/{len(ek)}  ({time.time()-t0:.1f}s)")

    pcm = per_cell_mean_curvature(G, ek)
    kappa_arr = np.array([pcm[i] for i in range(Xpca.shape[0])])

    # build per-cell DataFrame
    per_cell = cell_meta.reset_index().copy()
    per_cell["kappa"] = kappa_arr
    per_cell = per_cell.dropna(subset=["kappa"]).copy()
    print(f"\n  per-cell rows with kappa: {len(per_cell)}")
    print(per_cell["response"].value_counts().to_dict())

    # 6. analyses
    print("\n[6] per-cell statistics ...")
    R = per_cell.loc[per_cell["response"] == "Responder", "kappa"].values
    NR = per_cell.loc[per_cell["response"] == "Non-responder", "kappa"].values
    print(f"  R cells:  n={len(R)}  mean kappa={R.mean():.4f} +- {R.std():.4f}")
    print(f"  NR cells: n={len(NR)} mean kappa={NR.mean():.4f} +- {NR.std():.4f}")
    delta = cliffs_delta(R, NR)
    t_stat, p_t = ttest_ind(R, NR, equal_var=False)
    u_stat, p_u_cell = mannwhitneyu(R, NR, alternative="two-sided")
    print(f"  Cliff's delta (R - NR) = {delta:+.4f}")
    print(f"  Welch t = {t_stat:.3f}  p = {p_t:.4g}")
    print(f"  MWU U = {u_stat:.1f}  p = {p_u_cell:.4g}")

    # per-patient
    print("\n[7] per-patient summary ...")
    per_pt = (per_cell.groupby("patient_id")
              .agg(n_cells=("kappa", "size"),
                   mean_kappa=("kappa", "mean"),
                   std_kappa=("kappa", "std"),
                   response=("response", "first"),
                   therapy=("therapy", "first"))
              .reset_index())
    print(per_pt.to_string(index=False))

    R_means = per_pt.loc[per_pt["response"] == "Responder", "mean_kappa"].values
    NR_means = per_pt.loc[per_pt["response"] == "Non-responder", "mean_kappa"].values
    if len(R_means) >= 2 and len(NR_means) >= 2:
        u_pt, p_u_pt = mannwhitneyu(R_means, NR_means, alternative="two-sided")
    else:
        u_pt, p_u_pt = (np.nan, np.nan)
    # AUC: label R = 1, mean_kappa as score
    y = (per_pt["response"] == "Responder").astype(int).values
    score = per_pt["mean_kappa"].values
    if y.sum() > 0 and y.sum() < len(y):
        auc_high = roc_auc_score(y, score)            # high kappa -> R
        auc_low = roc_auc_score(y, -score)            # high kappa -> NR
        auc_directional = max(auc_high, auc_low)
        auc_dir_label = "high-kappa = Responder" if auc_high >= auc_low else "high-kappa = Non-responder"
    else:
        auc_high = auc_low = auc_directional = np.nan
        auc_dir_label = "n/a"

    print(f"\n  per-patient: n_R={len(R_means)} n_NR={len(NR_means)}")
    print(f"  R   mean_kappa = {np.nanmean(R_means):.4f} +- {np.nanstd(R_means):.4f}")
    print(f"  NR  mean_kappa = {np.nanmean(NR_means):.4f} +- {np.nanstd(NR_means):.4f}")
    print(f"  MWU on patient means: U={u_pt}  p={p_u_pt}")
    print(f"  AUC (high kappa = R)  = {auc_high:.4f}")
    print(f"  AUC (high kappa = NR) = {auc_low:.4f}")
    print(f"  best directional AUC  = {auc_directional:.4f}  ({auc_dir_label})")

    # 8. logistic regression
    print("\n[8] logistic regression responder ~ mean_kappa + n_cells ...")
    X_lr = sm.add_constant(per_pt[["mean_kappa", "n_cells"]].values.astype(float))
    yy = (per_pt["response"] == "Responder").astype(int).values
    try:
        lr = sm.Logit(yy, X_lr).fit(disp=False, maxiter=200)
        lr_summary = lr.summary().as_text()
    except Exception as e:
        lr = None
        lr_summary = f"logit failed: {e}"
    print(lr_summary)

    # 9. outputs
    print("\n[9] writing outputs ...")
    per_cell_out = per_cell[["cell_id", "patient_id", "response", "kappa"]]
    per_cell_out.to_csv(OUT / "n1_sadefeldman_per_cell.csv", index=False)
    per_pt[["patient_id", "response", "n_cells",
            "mean_kappa", "std_kappa", "therapy"]].to_csv(
        OUT / "n1_sadefeldman_per_patient.csv", index=False)

    # plot: per-patient mean kappa by response, jitter + box
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    groups = ["Non-responder", "Responder"]
    data_box = [per_pt.loc[per_pt["response"] == g, "mean_kappa"].values
                for g in groups]
    bp = ax.boxplot(data_box, positions=[0, 1], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#d75555", "#5588cc"]):
        patch.set_facecolor(color); patch.set_alpha(0.4)
    rng = np.random.default_rng(SEED)
    for i, g in enumerate(groups):
        sub = per_pt[per_pt["response"] == g]
        x = i + (rng.random(len(sub)) - 0.5) * 0.18
        ax.scatter(x, sub["mean_kappa"].values, s=70, alpha=0.85,
                   color=("#d75555" if g == "Non-responder" else "#5588cc"),
                   edgecolor="k", linewidth=0.6)
        for _, row in sub.iterrows():
            ax.text(i + 0.12, row["mean_kappa"], row["patient_id"],
                    fontsize=7, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(groups)
    ax.set_ylabel("per-patient mean per-cell Ollivier-Ricci kappa")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    title = (f"Sade-Feldman 2018  baseline CD8 T cells\n"
             f"per-patient MWU p={p_u_pt:.3g}  AUC={auc_directional:.3f}  "
             f"({auc_dir_label})")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "n1_sadefeldman_summary.png", dpi=130)
    plt.close()

    # text summary
    lines = []
    lines.append("Sade-Feldman 2018 GSE120575 -- per-cell kappa as ICB biomarker")
    lines.append("=" * 64)
    lines.append(f"seed={SEED}")
    lines.append(f"baseline CD8 T cells: {len(per_cell)} from {per_pt.shape[0]} patients "
                 f"(R={len(R_means)}, NR={len(NR_means)})")
    lines.append("")
    lines.append("[per-cell] (treats cells iid, ignores patient correlation)")
    lines.append(f"  R cells: n={len(R)}  kappa = {R.mean():.4f} +- {R.std():.4f}")
    lines.append(f"  NR cells:n={len(NR)} kappa = {NR.mean():.4f} +- {NR.std():.4f}")
    lines.append(f"  Cliff's delta (R - NR) = {delta:+.4f}")
    lines.append(f"  Welch t = {t_stat:.3f}  p = {p_t:.4g}")
    lines.append(f"  MWU U  = {u_stat:.1f} p = {p_u_cell:.4g}")
    lines.append("")
    lines.append("[per-patient] (proper clinically informative test)")
    lines.append(f"  R  mean_kappa = {np.nanmean(R_means):.4f} +- {np.nanstd(R_means):.4f}")
    lines.append(f"  NR mean_kappa = {np.nanmean(NR_means):.4f} +- {np.nanstd(NR_means):.4f}")
    lines.append(f"  MWU U = {u_pt}  p = {p_u_pt}")
    lines.append(f"  AUC high-kappa-=-R   = {auc_high:.4f}")
    lines.append(f"  AUC high-kappa-=-NR  = {auc_low:.4f}")
    lines.append(f"  best directional AUC = {auc_directional:.4f}  ({auc_dir_label})")
    lines.append("")
    lines.append("[logistic regression responder ~ mean_kappa + n_cells]")
    lines.append(lr_summary)
    lines.append("")
    lines.append("Per-patient table:")
    lines.append(per_pt.to_string(index=False))
    (OUT / "n1_sadefeldman_summary.txt").write_text("\n".join(lines))

    print("\nWrote:")
    for f in ["n1_sadefeldman_per_cell.csv",
              "n1_sadefeldman_per_patient.csv",
              "n1_sadefeldman_summary.png",
              "n1_sadefeldman_summary.txt"]:
        print(f"  {OUT / f}")


if __name__ == "__main__":
    main()
