"""
T2 — Wu et al. 2020 (GSE139555) clonality replication of Yost BCC E7.

Question: Is the Yost-BCC clonality finding (TCR-expanded T cells show
intermediate per-cell Ollivier-Ricci kappa between non-expanded T and
malignant, ~46% of the full malignant-vs-T shift) general to other tumor
TCR-paired scRNA-seq datasets, or BCC-specific?

Dataset: Wu et al. 2020 "Single-cell analyses inform mechanisms of
T-cell receptor (TCR) repertoire utilization in human cancer". GSE139555 —
NSCLC + colorectal + renal + endometrial T cells with paired TCR.
141k T cells across 14 patients, clonotypes already assigned in
GSE139555_tcell_metadata.txt.gz.

The Wu T-cell deposit is T-cells-only (no malignant cells in the same
matrix), so this is a *two-way* comparison: expanded_T vs non_expanded_T,
matching the second arm of the Yost test. We do not have a malignant
arm here, so we cannot compute the clonality share of a malignant shift,
only the absolute size of the expanded-vs-non-expanded effect.

Pipeline (matches E7):
  HVG-2000, log1p, mean-center, PCA-50, kNN k=15, alpha=0.5,
  sample 4000 edges, per-cell mean kappa = mean kappa over incident edges.

Two groups:
  expanded_T     — clonotype with >= 10 cells in this dataset
  non_expanded_T — singleton clonotype (1 cell) or no TCR

To keep download tractable we use two samples that together carry the
densest TCR + matching expression:
  LT2 (Lung tumor, 8486 T cells)
  LN3 (Lung NAT,   11594 T cells)
This gives ~14k TCR'd T cells across two tissues and is enough to
sample 600 expanded + 600 non-expanded.

Random seed 20260507.
"""

import sys, time, gzip
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE   = Path(__file__).parent
TCELL_META = HERE / "GSE139555_tcell_metadata.txt.gz"

# (sample_prefix, GSM_ID, full_name)
SAMPLES = [
    ("LT2", "GSM4143657", "GSM4143657_SAM24348188-lt2"),
    ("LN3", "GSM4143660", "GSM4143660_SAM24349906-ln3"),
]
LOG    = HERE / "run.log"

SEED       = 20260507
N_PER      = 600
HVG_K      = 2000
N_PCS      = 50
KNN_K      = 15
ALPHA      = 0.5
N_EDGES    = 4000
EXP_THRESH = 10

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
    a = np.asarray(a); b = np.asarray(b)
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
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
def assign_groups(meta):
    """Add a 'group' column in {expanded_T, non_expanded_T, other}."""
    md = meta.copy()
    has_tcr = md["clonotype"].notna() & (md["clonotype"] != "NA") & (md["clonotype"] != "")
    sizes = md.loc[has_tcr, "clonotype"].value_counts()
    log(f"  T cells with TCR clonotype: {has_tcr.sum()} / {len(md)}")
    log(f"  unique clonotypes: {len(sizes)}")
    log(f"  max clonotype size: {sizes.max()}")
    log(f"  clonotypes >= {EXP_THRESH}: {(sizes >= EXP_THRESH).sum()} groups containing "
        f"{sizes[sizes >= EXP_THRESH].sum()} cells")
    log(f"  singleton clonotypes: {(sizes == 1).sum()}")

    expanded_set  = set(sizes[sizes >= EXP_THRESH].index)
    singleton_set = set(sizes[sizes == 1].index)

    md["group"] = "other"
    is_exp = has_tcr & md["clonotype"].isin(expanded_set)
    is_sng = has_tcr & md["clonotype"].isin(singleton_set)
    is_no  = ~has_tcr
    md.loc[is_exp, "group"] = "expanded_T"
    md.loc[is_sng | is_no, "group"] = "non_expanded_T"
    log("  group counts (full T-cell meta):")
    for g, n in md["group"].value_counts().items():
        log(f"    {g:18s} {n}")
    return md


# ------------------------- counts loader -------------------------
def load_sample_matrix(sample_prefix, full_name, cells_wanted_in_sample):
    """
    Read a 10x mtx triple. 'genes.tsv' is gene-id then symbol; we use symbol.
    Return (gene_symbols, kept_cell_ids_with_prefix, X_genes_x_cells_dense_kept).
    """
    barcode_path = HERE / f"{full_name}.barcodes.tsv.gz"
    genes_path   = HERE / f"{full_name}.genes.tsv.gz"
    mtx_path     = HERE / f"{full_name}.matrix.mtx.gz"

    with gzip.open(barcode_path, "rt") as fh:
        barcodes = [ln.strip() for ln in fh if ln.strip()]
    with gzip.open(genes_path, "rt") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    # Some 10x genes.tsv files use only one col (the GeneID:xxx) — handle both.
    if len(rows[0]) >= 2:
        gene_symbols = [r[1] for r in rows]
    else:
        gene_symbols = [r[0] for r in rows]
    log(f"  {sample_prefix}: barcodes={len(barcodes)} genes={len(gene_symbols)}")

    # cells in this sample wanted
    prefixed = [f"{sample_prefix}_{b}" for b in barcodes]
    keep_mask = np.array([c in cells_wanted_in_sample for c in prefixed])
    keep_idx  = np.where(keep_mask)[0]
    log(f"  {sample_prefix}: keeping {len(keep_idx)} cells of {len(barcodes)}")

    # load mtx
    with gzip.open(mtx_path, "rt") as fh:
        # skip comments
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        nrows, ncols, nnz = [int(x) for x in line.split()]
        log(f"  {sample_prefix}: mtx {nrows}x{ncols} nnz={nnz}")
        # sparse build
        I = np.empty(nnz, dtype=np.int32)
        J = np.empty(nnz, dtype=np.int32)
        V = np.empty(nnz, dtype=np.float32)
        for k, ln in enumerate(fh):
            r, c, v = ln.split()
            I[k] = int(r) - 1
            J[k] = int(c) - 1
            V[k] = float(v)
        # genes x cells sparse
        M = sp.csc_matrix((V, (I, J)), shape=(nrows, ncols))
    # column-subset to kept cells
    Xk = M[:, keep_idx].toarray().astype(np.float32)
    kept_cell_ids = [prefixed[i] for i in keep_idx]
    log(f"  {sample_prefix}: kept matrix {Xk.shape}")
    return gene_symbols, kept_cell_ids, Xk


def load_counts_for_cells(cells_wanted):
    """Return (gene_symbols, cells_kept_in_order, X_genes_x_cells)."""
    # split wanted cells by sample prefix
    by_sample = {s: set() for s, *_ in SAMPLES}
    for c in cells_wanted:
        pref = c.split("_", 1)[0]
        if pref in by_sample:
            by_sample[pref].add(c)
    blocks = []
    cells_kept_all = []
    gene_ref = None
    for sample_prefix, _gsm, full_name in SAMPLES:
        wanted = by_sample[sample_prefix]
        log(f"  loading sample {sample_prefix}: {len(wanted)} wanted cells")
        if not wanted:
            log(f"  -> no wanted cells in {sample_prefix}, skipping")
            continue
        genes, kept_ids, Xk = load_sample_matrix(sample_prefix, full_name, wanted)
        if gene_ref is None:
            gene_ref = genes
        else:
            assert genes == gene_ref, f"gene order mismatch for {sample_prefix}"
        blocks.append(Xk)
        cells_kept_all.extend(kept_ids)
    X = np.hstack(blocks).astype(np.float32) if blocks else np.zeros((0, 0), dtype=np.float32)
    log(f"  combined counts: {X.shape}")
    return gene_ref, np.array(cells_kept_all), X


# ------------------------- main -------------------------
def main():
    t_start = time.time()
    log("="*72)
    log("T2 — Wu et al. 2020 (GSE139555) clonality replication of Yost BCC E7")
    log(f"seed={SEED} n_per_group={N_PER} hvg={HVG_K} pcs={N_PCS} "
        f"k={KNN_K} alpha={ALPHA} n_edges={N_EDGES} exp_threshold={EXP_THRESH}")
    log(f"samples used: {[s[0] for s in SAMPLES]}")
    log("="*72)

    # 1) metadata + clonotype --------------------------------------------
    log("\n[1/5] loading T cell metadata + clonotype")
    meta = pd.read_csv(TCELL_META, sep="\t", index_col=0)
    meta.index.name = "cell.id"
    meta = meta.reset_index()
    log(f"  meta rows={len(meta)} cols={list(meta.columns)}")
    # restrict to cells in the chosen samples (have expression on disk)
    sample_prefixes = {s[0] for s in SAMPLES}
    meta["sample_prefix"] = meta["cell.id"].str.split("_").str[0]
    meta = meta[meta["sample_prefix"].isin(sample_prefixes)].copy()
    log(f"  after sample filter: {len(meta)}")
    md = assign_groups(meta)

    # 2) sample groups ---------------------------------------------------
    log("\n[2/5] sampling cells per group")
    sample_idxs = {}
    for grp in ["expanded_T", "non_expanded_T"]:
        pool = md[md["group"] == grp]["cell.id"].values
        n_take = min(N_PER, len(pool))
        chosen = RNG.choice(pool, size=n_take, replace=False)
        sample_idxs[grp] = chosen
        log(f"  {grp:18s} sampled {n_take} of {len(pool)}")
    cells_wanted_list = np.concatenate([sample_idxs[g] for g in
                                        ["expanded_T", "non_expanded_T"]])
    cells_wanted = set(cells_wanted_list)

    # 3) load counts -----------------------------------------------------
    log("\n[3/5] loading counts and building expression matrix")
    gene_names, cells_kept, X = load_counts_for_cells(cells_wanted)
    cell_to_group = dict(zip(md["cell.id"], md["group"]))
    labels = np.array([cell_to_group.get(c, "other") for c in cells_kept])
    log(f"  per-group cells in matrix:")
    for g, n in pd.Series(labels).value_counts().items():
        log(f"    {g:18s} {n}")

    # log1p + HVG
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

    # 4) kNN + Ollivier-Ricci -------------------------------------------
    log("\n[4/5] kNN + Ollivier-Ricci")
    G = build_knn_graph(Xpca, k=KNN_K)
    log(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    edge_kappa = ollivier_ricci_edges(G, alpha=ALPHA, n_edges=N_EDGES, rng=RNG)
    n_valid = sum(1 for v in edge_kappa.values() if not np.isnan(v))
    log(f"  valid edges: {n_valid}/{len(edge_kappa)}")
    per_cell = per_cell_mean_curvature(G, edge_kappa)

    kappa_by_group = {}
    for g in ["expanded_T", "non_expanded_T"]:
        idxs = np.where(labels == g)[0]
        vals = np.array([per_cell[i] for i in idxs if not np.isnan(per_cell[i])])
        kappa_by_group[g] = vals
        log(f"  {g:18s} n_with_kappa={len(vals)}  mean={vals.mean():.4f}  "
            f"std={vals.std():.4f}")

    # 5) stats ----------------------------------------------------------
    log("\n[5/5] statistics + outputs")
    rows_summary = []
    for g, vals in kappa_by_group.items():
        rows_summary.append(dict(
            group=g, n=len(vals),
            kappa_mean=float(vals.mean()),
            kappa_std=float(vals.std()),
            kappa_median=float(np.median(vals)),
        ))

    pairs = [("expanded_T", "non_expanded_T")]
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
            f"Cliff_delta={delta:+.3f}")
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
    df_sum.to_csv(HERE / "t2_wu_clonality_summary.csv", index=False)
    df_pair.to_csv(HERE / "t2_wu_pairs.csv", index=False)
    log(f"  wrote t2_wu_clonality_summary.csv ({len(df_sum)} rows)")
    log(f"  wrote t2_wu_pairs.csv ({len(df_pair)} rows)")

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"non_expanded_T": "#4477aa",
              "expanded_T":     "#ee7733"}
    for g in ["non_expanded_T", "expanded_T"]:
        v = kappa_by_group[g]
        if len(v) == 0: continue
        ax.hist(v, bins=40, density=True, alpha=0.55,
                label=f"{g} (n={len(v)}, mean={v.mean():.3f})",
                color=colors[g])
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
    ax.set_ylabel("density")
    ax.set_title("T2 — Wu 2020 GSE139555: kappa, expanded vs non-expanded T")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_png = HERE / "t2_wu_distributions.png"
    plt.savefig(out_png, dpi=120); plt.close()
    log(f"  wrote {out_png.name}")

    # verdict
    log("\n" + "="*72)
    log("VERDICT")
    log("="*72)
    if not df_pair.empty:
        d = float(df_pair.loc[0, "cliffs_delta"])
        log(f"  Cliff delta  expanded_T vs non_expanded_T = {d:+.3f}")
        log(f"  Yost BCC reference value                  = +0.175")
        if d > 0.30:
            verdict = ("CLONALITY DOMINATES — expanded T cells in Wu 2020 show "
                       "an even larger kappa shift than in Yost BCC; clonality "
                       "appears to be a major driver of the kappa signal in "
                       "this dataset.")
        elif 0.10 <= d <= 0.30:
            verdict = ("CLONALITY GENERAL — expanded vs non-expanded T cell "
                       "kappa shift is in the same range as Yost BCC, "
                       "supporting the BCC finding as a general property of "
                       "TCR-clonal expansion across tumor scRNA datasets.")
        elif 0.05 <= d < 0.10:
            verdict = ("WEAK / PARTIAL — clonality contributes but the effect "
                       "is smaller than in Yost BCC; tumor-specific programs "
                       "may dominate in BCC.")
        elif d < 0.05 and d > -0.05:
            verdict = ("CLONALITY BCC-SPECIFIC — no detectable kappa shift "
                       "between expanded and non-expanded T cells in Wu 2020; "
                       "the BCC finding does not generalize.")
        else:
            verdict = ("INVERTED — expanded T cells show LOWER kappa than "
                       "non-expanded; opposite to Yost BCC.")
        log("  " + verdict)
    log(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
