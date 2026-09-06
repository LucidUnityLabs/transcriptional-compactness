"""
E7 — Clonality vs Malignancy specificity test for per-cell Ollivier-Ricci kappa.

Question: does kappa detect MALIGNANCY specifically, or any CLONAL EXPANSION?
If TCR-expanded T-cell clones (non-malignant) show kappa as elevated as
malignant cells, the malignancy framing dies — kappa is just a clonal-
compactness sensor.

Dataset: Yost et al. 2019, GSE123813 (BCC scRNA-seq + paired TCR).
- BCC scRNA counts (~53k cells, all cell types incl. tumor + T cells)
- BCC all metadata (cluster labels incl. Tumor_1/Tumor_2 + T-cell clusters)
- BCC TCR (cdr3s_aa per T cell)

Pipeline (matches path1_hyperbolic/run_ricci.py pipeline):
  HVG-2000, log1p, mean-center, PCA-50, kNN k=15, alpha=0.5, sample 4000 edges,
  per-cell mean kappa = mean kappa over incident edges.

Three groups for comparison (within the BCC dataset):
  1. T cells with NO TCR or singleton CDR3 group (n_cdr3 == 1)        -> "non_expanded_T"
  2. T cells in CDR3 groups with >= 10 cells                          -> "expanded_T"
  3. Malignant cells (cluster == Tumor_1 or Tumor_2)                  -> "malignant"

Each group is sub-sampled to a target size (default 600) before building a
SINGLE kNN graph over the union, so curvature comparison is consistent.

Outputs (this dir):
  e7_clonality_summary.csv     per-group means/std/n + univariate stats
  e7_three_way_table.csv       pairwise Welch + MWU + Cliff's delta
  e7_clonality_distributions.png   kappa overlays
  run.log                       stdout

Random seed 20260507.
"""

import sys, time, gzip, io
from pathlib import Path
import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE   = Path(__file__).parent
COUNTS = HERE / "GSE123813_bcc_scRNA_counts.txt.gz"
META   = HERE / "GSE123813_bcc_all_metadata.txt.gz"
TCR    = HERE / "GSE123813_bcc_tcr.txt.gz"
LOG    = HERE / "run.log"

SEED       = 20260507
N_PER      = 600
HVG_K      = 2000
N_PCS      = 50
KNN_K      = 15
ALPHA      = 0.5
N_EDGES    = 4000
EXP_THRESH = 10   # CDR3 group >= this -> "expanded"

T_CELL_CLUSTERS = {
    "CD8_mem_T_cells", "CD4_T_cells", "Tregs", "CD8_act_T_cells",
    "CD8_ex_T_cells", "Tcell_prolif",
}
MAL_CLUSTERS = {"Tumor_1", "Tumor_2"}

RNG = np.random.default_rng(SEED)


# ------------------------- logging -------------------------
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()

_log_fh = open(LOG, "w")
sys.stdout = Tee(sys.__stdout__, _log_fh)
sys.stderr = Tee(sys.__stderr__, _log_fh)
def log(*a): print(*a)


# ------------------------- Ollivier-Ricci -------------------------
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


def ollivier_ricci_edges(G, alpha=0.5, n_edges=None, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    edges = list(G.edges())
    if n_edges is not None and n_edges < len(edges):
        idx = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in idx]
    out = {}
    for (u, v) in edges:
        nu = list(G.neighbors(u)); nv = list(G.neighbors(v))
        sup_u = [u] + nu;          sup_v = [v] + nv
        w_u = np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu)) if nu else np.array([1.0])
        w_v = np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv)) if nv else np.array([1.0])
        sub_nodes = list(set(sup_u + sup_v))
        sub = G.subgraph(sub_nodes)
        try:
            dists = {n_: nx.single_source_dijkstra_path_length(sub, n_) for n_ in sup_u}
        except Exception:
            out[(u, v)] = np.nan; continue
        C = np.zeros((len(sup_u), len(sup_v)))
        for i, a in enumerate(sup_u):
            for j, b in enumerate(sup_v):
                C[i, j] = dists[a].get(b, np.inf)
        if not np.all(np.isfinite(C)):
            out[(u, v)] = np.nan; continue
        W = ot.emd2(w_u, w_v, C)
        d_uv = G[u][v]["weight"]
        out[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan
    return out


def per_cell_mean_curvature(G, edge_kappa):
    by_cell = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k): continue
        by_cell[u].append(k); by_cell[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by_cell.items()}


# ------------------------- effect size -------------------------
def cliffs_delta(a, b):
    """Cliff's delta: P(a>b) - P(a<b). a,b 1D arrays."""
    a = np.asarray(a); b = np.asarray(b)
    # rank-based formulation; O(n log n)
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    # average ranks for ties
    sorted_vals = combined[order]
    i = 0; n = len(combined)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    r_a = ranks[: len(a)].sum()
    U = r_a - len(a) * (len(a) + 1) / 2.0
    return 2.0 * U / (len(a) * len(b)) - 1.0


# ------------------------- group assignment -------------------------
def assign_groups(meta, tcr):
    """Return DataFrame with cell.id and 'group' in {non_expanded_T, expanded_T, malignant, other}."""
    md = meta.copy()
    md = md.set_index("cell.id")
    # join TCR. TCR has 2-name header but 3 data fields per row; pandas treats
    # the first field as the index, so tcr.index = cell ids.
    tcr_idx = tcr[~tcr.index.duplicated(keep="first")]
    md["cdr3"] = tcr_idx["cdr3s_aa"].reindex(md.index)
    # within T-cell clusters only consider TCR
    is_T = md["cluster"].isin(T_CELL_CLUSTERS)
    is_mal = md["cluster"].isin(MAL_CLUSTERS)

    md["group"] = "other"
    md.loc[is_mal, "group"] = "malignant"

    # CDR3 group sizes among T cells (only cells with non-null CDR3)
    t_meta = md[is_T].copy()
    has_tcr = t_meta["cdr3"].notna()
    cdr3_counts = t_meta.loc[has_tcr, "cdr3"].value_counts()
    log(f"  T cells with TCR: {has_tcr.sum()} / {len(t_meta)}")
    log(f"  unique CDR3 groups: {len(cdr3_counts)}")
    log(f"  CDR3 group sizes: max={cdr3_counts.max()}, "
        f">=10: {(cdr3_counts >= EXP_THRESH).sum()} groups containing "
        f"{cdr3_counts[cdr3_counts >= EXP_THRESH].sum()} cells")

    expanded_cdr3 = set(cdr3_counts[cdr3_counts >= EXP_THRESH].index)
    singleton_cdr3 = set(cdr3_counts[cdr3_counts == 1].index)

    # expanded T = T cell with TCR in expanded set
    is_exp = is_T & md["cdr3"].isin(expanded_cdr3)
    # non-expanded T = T cell with NO TCR or with singleton CDR3
    is_singleton = is_T & md["cdr3"].isin(singleton_cdr3)
    is_no_tcr   = is_T & md["cdr3"].isna()
    is_nonexp = is_singleton | is_no_tcr

    md.loc[is_exp, "group"] = "expanded_T"
    md.loc[is_nonexp, "group"] = "non_expanded_T"

    log("  group counts:")
    for g, n in md["group"].value_counts().items():
        log(f"    {g:18s} {n}")
    return md.reset_index()


# ------------------------- counts loader -------------------------
def load_counts_for_cells(cells_wanted):
    """
    Stream the gzipped tab-separated counts file, keep only the columns
    corresponding to cells_wanted (a set of cell ids). Returns
    (gene_names, X) where X has shape (n_genes, n_cells_kept) float32.
    """
    log(f"  streaming counts file (target {len(cells_wanted)} cells) ...")
    with gzip.open(COUNTS, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        # data rows have 1 extra column at the front (gene name).
        # so 'header' has 53030 cell IDs that align to data columns 2..53031.
        keep_mask = np.array([c in cells_wanted for c in header])
        keep_idx = np.where(keep_mask)[0]
        cell_ids_kept = [header[i] for i in keep_idx]
        log(f"  matched {len(keep_idx)} of {len(header)} cell columns")

        gene_names = []
        rows = []
        # data rows: column 0 is gene, columns 1.. correspond to header 0..
        # so a data column at index (k+1) corresponds to header[k].
        target_data_cols = keep_idx + 1
        t0 = time.time()
        for ln, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            gene_names.append(parts[0])
            arr = np.array([parts[c] for c in target_data_cols], dtype=np.float32)
            rows.append(arr)
            if (ln + 1) % 5000 == 0:
                log(f"    {ln+1} genes streamed ({time.time()-t0:.1f}s)")
    X = np.vstack(rows).astype(np.float32)
    log(f"  counts matrix: {X.shape}  (genes x cells)  in {time.time()-t0:.1f}s")
    return gene_names, np.array(cell_ids_kept), X


# ------------------------- main -------------------------
def main():
    t_start = time.time()
    log("="*72)
    log("E7 — Ollivier-Ricci kappa: clonality vs malignancy specificity test")
    log(f"seed={SEED} n_per_group={N_PER} hvg={HVG_K} pcs={N_PCS} "
        f"k={KNN_K} alpha={ALPHA} n_edges={N_EDGES} exp_threshold={EXP_THRESH}")
    log("="*72)

    # 1) metadata + TCR ---------------------------------------------------
    log("\n[1/5] loading metadata + TCR")
    meta = pd.read_csv(META, sep="\t")
    tcr  = pd.read_csv(TCR,  sep="\t")
    log(f"  meta rows={len(meta)} cols={list(meta.columns)}")
    log(f"  tcr  rows={len(tcr)}  cols={list(tcr.columns)}")
    md = assign_groups(meta, tcr)

    # 2) sample groups ----------------------------------------------------
    log("\n[2/5] sampling cells per group")
    sample_idxs = {}
    for grp in ["malignant", "expanded_T", "non_expanded_T"]:
        pool = md[md["group"] == grp]["cell.id"].values
        n_take = min(N_PER, len(pool))
        chosen = RNG.choice(pool, size=n_take, replace=False)
        sample_idxs[grp] = chosen
        log(f"  {grp:18s} sampled {n_take} of {len(pool)}")
    cells_wanted_list = np.concatenate([sample_idxs[g] for g in
                                        ["malignant", "expanded_T", "non_expanded_T"]])
    cells_wanted = set(cells_wanted_list)

    # 3) stream counts ----------------------------------------------------
    log("\n[3/5] streaming counts and building expression matrix")
    gene_names, cells_kept, X = load_counts_for_cells(cells_wanted)
    # build label vector aligned to cells_kept order
    cell_to_group = dict(zip(md["cell.id"], md["group"]))
    labels = np.array([cell_to_group.get(c, "other") for c in cells_kept])
    log(f"  per-group cells in matrix:")
    for g, n in pd.Series(labels).value_counts().items():
        log(f"    {g:18s} {n}")

    # log1p + HVG selection on the joint expression
    Xlog = np.log1p(X)
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:HVG_K]
    Xlog = Xlog[hvg]
    log(f"  HVG-{HVG_K} selected; matrix now {Xlog.shape}")

    # cells x genes for PCA
    Xc = Xlog.T.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n_pcs = min(N_PCS, min(Xc.shape) - 1)
    Xpca = PCA(n_components=n_pcs, random_state=SEED).fit_transform(Xc)
    log(f"  PCA -> {Xpca.shape}")

    # 4) kNN + Ollivier-Ricci --------------------------------------------
    log("\n[4/5] kNN + Ollivier-Ricci")
    G = build_knn_graph(Xpca, k=KNN_K)
    log(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    edge_kappa = ollivier_ricci_edges(G, alpha=ALPHA, n_edges=N_EDGES, rng=RNG)
    n_valid = sum(1 for v in edge_kappa.values() if not np.isnan(v))
    log(f"  valid edges: {n_valid}/{len(edge_kappa)}")
    per_cell = per_cell_mean_curvature(G, edge_kappa)

    kappa_by_group = {}
    for g in ["malignant", "expanded_T", "non_expanded_T"]:
        idxs = np.where(labels == g)[0]
        vals = np.array([per_cell[i] for i in idxs if not np.isnan(per_cell[i])])
        kappa_by_group[g] = vals
        log(f"  {g:18s} n_with_kappa={len(vals)}  mean={vals.mean():.4f}  "
            f"std={vals.std():.4f}")

    # 5) stats ------------------------------------------------------------
    log("\n[5/5] statistics + outputs")
    rows_summary = []
    for g, vals in kappa_by_group.items():
        rows_summary.append(dict(
            group=g, n=len(vals),
            kappa_mean=float(vals.mean()),
            kappa_std=float(vals.std()),
            kappa_median=float(np.median(vals)),
        ))

    pairs = [("malignant", "non_expanded_T"),
             ("expanded_T", "non_expanded_T"),
             ("malignant", "expanded_T")]
    pair_rows = []
    for a, b in pairs:
        va, vb = kappa_by_group[a], kappa_by_group[b]
        if len(va) < 5 or len(vb) < 5:
            log(f"  pair {a} vs {b}: not enough data")
            continue
        t_stat, p_t = ttest_ind(va, vb, equal_var=False)
        u_stat, p_u = mannwhitneyu(va, vb)
        delta = cliffs_delta(va, vb)
        log(f"  {a} vs {b}: mean_a={va.mean():.4f} mean_b={vb.mean():.4f} "
            f"Welch t={t_stat:.3f} p={p_t:.3g}  MWU U={u_stat:.0f} p={p_u:.3g}  "
            f"Cliff_delta={delta:.3f}")
        pair_rows.append(dict(
            group_a=a, group_b=b,
            mean_a=float(va.mean()), mean_b=float(vb.mean()),
            n_a=len(va), n_b=len(vb),
            welch_t=float(t_stat), welch_p=float(p_t),
            mwu_U=float(u_stat), mwu_p=float(p_u),
            cliffs_delta=float(delta),
        ))

    df_sum = pd.DataFrame(rows_summary)
    df_pair = pd.DataFrame(pair_rows)
    df_sum.to_csv(HERE / "e7_clonality_summary.csv", index=False)
    df_pair.to_csv(HERE / "e7_three_way_table.csv", index=False)
    log(f"  wrote e7_clonality_summary.csv ({len(df_sum)} rows)")
    log(f"  wrote e7_three_way_table.csv ({len(df_pair)} rows)")

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"non_expanded_T": "#4477aa",
              "expanded_T":     "#ee7733",
              "malignant":      "#cc3311"}
    for g in ["non_expanded_T", "expanded_T", "malignant"]:
        v = kappa_by_group[g]
        if len(v) == 0: continue
        ax.hist(v, bins=40, density=True, alpha=0.55,
                label=f"{g} (n={len(v)}, mean={v.mean():.3f})",
                color=colors[g])
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
    ax.set_ylabel("density")
    ax.set_title("E7 — kappa: malignancy vs clonal expansion (Yost BCC)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_png = HERE / "e7_clonality_distributions.png"
    plt.savefig(out_png, dpi=120); plt.close()
    log(f"  wrote {out_png.name}")

    # verdict
    log("\n" + "="*72)
    log("VERDICT")
    log("="*72)
    if not df_pair.empty:
        d_exp_vs_non = df_pair.loc[(df_pair.group_a=="expanded_T") &
                                   (df_pair.group_b=="non_expanded_T"),
                                   "cliffs_delta"].values
        d_mal_vs_non = df_pair.loc[(df_pair.group_a=="malignant") &
                                   (df_pair.group_b=="non_expanded_T"),
                                   "cliffs_delta"].values
        if len(d_exp_vs_non) and len(d_mal_vs_non):
            d_e = float(d_exp_vs_non[0]); d_m = float(d_mal_vs_non[0])
            log(f"  Cliff delta  malignant vs non_expanded_T = {d_m:+.3f}")
            log(f"  Cliff delta  expanded_T vs non_expanded_T = {d_e:+.3f}")
            sign = lambda x: "same direction" if (d_e * d_m) > 0 else "opposite direction"
            log(f"  ({sign(d_e)} of effect)")
            if abs(d_e) >= 0.3 and (d_e * d_m) > 0:
                verdict = ("MALIGNANCY FRAMING DIES — TCR-expanded T cells show "
                           "kappa shift comparable to malignant cells. The signal "
                           "is clonal/transcriptional compactness, not malignancy.")
            elif abs(d_e) < 0.1:
                verdict = ("MALIGNANCY FRAMING SURVIVES — clonal expansion alone "
                           "does NOT elevate kappa; tumor-specific transcriptional "
                           "programs are needed to produce the kappa shift.")
            else:
                verdict = ("PARTIAL — clonality contributes but does not fully "
                           "explain the malignant kappa signal. Headline becomes "
                           "'kappa tracks transcriptional compactness, of which "
                           "clonal expansion is one major contributor'.")
            log("  " + verdict)
    log(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
