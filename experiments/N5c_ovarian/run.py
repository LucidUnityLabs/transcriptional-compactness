"""
N5c -- Ollivier-Ricci per-cell curvature on HGSOC scRNA-seq.

Seventh tumour type test. Replicates the per-cell OR pipeline of
path1_hyperbolic/run_ricci.py (and N5a/N5b style) on:

  Olalekan et al. 2021, GSE147082 -- 6 primary HGSOC patients,
  Smart-seq2-like read-count CSVs (genes x cells, no provided cell labels).

Because GSE147082 ships only counts (no cell-type calls), we assign coarse
cell types via canonical marker-gene scoring (per cell, log1p-normalised CP10K
score relative to a random gene panel; standard Tirosh-style scoring rule).
The malignant call combines a positive Epithelial score with a negative
Immune+Stromal score (so doublets and ambiguous cells are excluded).

Protocol (matches N5a/N5b):
  - log1p-CP10K, HVG-2000, PCA-50, kNN k=15
  - Ollivier-Ricci alpha=0.5, sampled edges = min(4000, |E|)
  - 600+600 stratified subsample where possible (else min(n_a, n_b))
  - Cliff's delta + Welch t + Mann-Whitney U
  - contrasts: malignant epithelial vs T cells; vs fibroblasts; vs all others
"""
import sys
import time
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

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data" / "GSE147082"
SEED = 20260508
RNG = np.random.default_rng(SEED)


# ---------------- Ollivier-Ricci on a kNN graph ----------------
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
        w_u = (np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu)) if nu
               else np.array([1.0]))
        w_v = (np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv)) if nv
               else np.array([1.0]))
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
        kappa = 1.0 - W / d_uv if d_uv > 0 else np.nan
        out[(u, v)] = float(kappa)
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
    a = np.asarray(a)
    b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return np.nan
    gt = 0
    lt = 0
    bsz = 500
    for i in range(0, n_a, bsz):
        chunk = a[i:i + bsz][:, None]
        gt += np.sum(chunk > b[None, :])
        lt += np.sum(chunk < b[None, :])
    return float(gt - lt) / (n_a * n_b)


# ---------------- Marker-gene cell typing ----------------
EPITHELIAL_MARKERS = [
    "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1",
    "MUC16", "PAX8", "WT1", "FOLR1", "MSLN", "CLDN3", "CLDN4",
]
TCELL_MARKERS = ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "TRAC", "TRBC1", "TRBC2"]
BCELL_MARKERS = ["CD79A", "CD79B", "MS4A1", "CD19", "IGKC", "IGHG1"]
MAC_MARKERS = ["CD68", "CD163", "AIF1", "LYZ", "C1QA", "C1QB", "C1QC", "MARCO"]
FIB_MARKERS = ["COL1A1", "COL1A2", "COL3A1", "DCN", "PDGFRA", "PDGFRB", "ACTA2", "FAP", "LUM"]
END_MARKERS = ["PECAM1", "VWF", "CDH5", "CLDN5", "ENG", "KDR"]

MARKER_SETS = {
    "Epithelial": EPITHELIAL_MARKERS,
    "Tcell": TCELL_MARKERS,
    "Bcell": BCELL_MARKERS,
    "Macrophage": MAC_MARKERS,
    "Fibroblast": FIB_MARKERS,
    "Endothelial": END_MARKERS,
}


def score_marker_set(X_log_cp10k, gene_index, marker_genes, n_ctrl=50,
                     rng=None):
    """
    Tirosh-style signature score: mean log expression of marker genes
    minus mean log expression of n_ctrl random control genes (binned by mean
    expression to remove length / depth bias). Returns one score per cell.

    X_log_cp10k: cells x genes float32 (log1p-CP10K)
    gene_index:  dict gene_name -> column index
    """
    if rng is None:
        rng = np.random.default_rng(0)
    present = [g for g in marker_genes if g in gene_index]
    if not present:
        return np.full(X_log_cp10k.shape[0], np.nan, dtype=np.float32)
    sig_idx = np.array([gene_index[g] for g in present], dtype=int)

    # bin all genes by mean expression for control panel
    means = X_log_cp10k.mean(axis=0)
    n_bins = 25
    quantiles = np.quantile(means, np.linspace(0, 1, n_bins + 1))
    quantiles[-1] += 1e-9
    bin_id = np.digitize(means, quantiles[1:-1])
    sig_bins = bin_id[sig_idx]
    ctrl_pool = []
    for b in np.unique(sig_bins):
        cands = np.where((bin_id == b))[0]
        cands = np.setdiff1d(cands, sig_idx, assume_unique=False)
        if len(cands) == 0:
            continue
        n_pick = min(n_ctrl, len(cands))
        pick = rng.choice(cands, size=n_pick, replace=False)
        ctrl_pool.append(pick)
    ctrl_idx = np.concatenate(ctrl_pool) if ctrl_pool else np.array([], dtype=int)

    sig_score = X_log_cp10k[:, sig_idx].mean(axis=1)
    if len(ctrl_idx):
        ctrl_score = X_log_cp10k[:, ctrl_idx].mean(axis=1)
    else:
        ctrl_score = np.zeros_like(sig_score)
    return sig_score - ctrl_score


def assign_cell_types(scores):
    """
    scores: dict celltype -> per-cell score array
    Rule:
      - argmax over signature scores assigns the call,
      - require that score > 0 (above control panel),
      - require max margin >= 0.05 over the second-best to discard ambiguous,
      - cells failing any of the above get "Unknown".
    """
    types = list(scores.keys())
    M = np.stack([scores[t] for t in types], axis=1)  # cells x types
    # robust to NaN
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


# ---------------- dataset loader ----------------
def load_olalekan_pooled():
    """
    Read all 6 patient CSVs, take gene intersection, concatenate cells, return
    (X_log_cp10k_genes_x_cells, gene_names, patient_ids, barcodes).
    """
    print("  loading GSE147082 (Olalekan 2021) ...")
    files = sorted(DATA.glob("*.csv"))
    if not files:
        print("  ERROR: no CSV files found in data/GSE147082")
        sys.exit(1)
    print(f"  found {len(files)} patient files")

    dfs = []
    pids = []
    for f in files:
        pid = f.stem.split("_")[-1]  # e.g. PT-3232
        # genes x cells, gene names in first col
        df = pd.read_csv(f, index_col=0, low_memory=False)
        print(f"    {f.name}: {df.shape[0]} genes, {df.shape[1]} cells")
        dfs.append(df)
        pids.append(pid)

    # gene intersection
    gene_sets = [set(d.index) for d in dfs]
    common = sorted(set.intersection(*gene_sets))
    print(f"  intersection: {len(common)} genes")

    parts = []
    cell_pids = []
    barcodes = []
    for d, pid in zip(dfs, pids):
        sub = d.loc[common]
        parts.append(sub.values.astype(np.float32))
        cell_pids.extend([pid] * sub.shape[1])
        barcodes.extend([f"{pid}__{c}" for c in sub.columns])

    X_counts = np.concatenate(parts, axis=1)  # genes x cells, raw counts
    print(f"  pooled matrix: {X_counts.shape[0]} genes x {X_counts.shape[1]} cells")

    # filter low-count cells (< 500 total UMI/reads)
    libsize = X_counts.sum(axis=0)
    keep = libsize >= 500
    X_counts = X_counts[:, keep]
    cell_pids = np.array(cell_pids)[keep]
    barcodes = np.array(barcodes)[keep]
    print(f"  after libsize>=500 cell filter: {X_counts.shape[1]} cells")

    # filter genes: keep genes detected in >=10 cells
    detected = (X_counts > 0).sum(axis=1)
    keep_g = detected >= 10
    X_counts = X_counts[keep_g]
    gene_names = np.array(common)[keep_g]
    print(f"  after gene>=10 detection filter: {X_counts.shape[0]} genes")

    # log1p CP10K normalisation
    libsize = X_counts.sum(axis=0)
    libsize[libsize == 0] = 1
    X_norm = X_counts / libsize[None, :] * 1e4
    X_log = np.log1p(X_norm).astype(np.float32)

    return X_log, gene_names, cell_pids, barcodes, X_counts


def main():
    t0 = time.time()
    print("=" * 60)
    print("N5c -- HGSOC (Olalekan 2021, GSE147082) Ollivier-Ricci")
    print("=" * 60)

    X_log, genes, patients, barcodes, X_counts = load_olalekan_pooled()
    gene_index = {g: i for i, g in enumerate(genes)}

    # ---- score marker sets and assign cell types (cells x genes oriented) ----
    Xc = X_log.T  # cells x genes
    scores = {}
    for ct, mlist in MARKER_SETS.items():
        present = [g for g in mlist if g in gene_index]
        print(f"  marker {ct}: {len(present)}/{len(mlist)} present")
        scores[ct] = score_marker_set(Xc, gene_index, mlist,
                                      n_ctrl=50, rng=np.random.default_rng(SEED))

    cell_types = assign_cell_types(scores)
    counts = pd.Series(cell_types).value_counts()
    print("\n  cell-type assignments:")
    for k, v in counts.items():
        print(f"    {k:14s} {v}")

    # save cell-type calls
    pd.DataFrame({
        "barcode": barcodes,
        "patient": patients,
        "cell_type": cell_types,
        **{f"score_{k}": v for k, v in scores.items()},
    }).to_csv(OUT / "n5c_cell_types.csv", index=False)

    # ---- HVG-2000 + PCA-50 on filtered cells (drop Unknown) ----
    keep_cells = cell_types != "Unknown"
    Xc_f = Xc[keep_cells]
    types_f = cell_types[keep_cells]
    pids_f = patients[keep_cells]
    print(f"\n  cells after dropping Unknown: {len(types_f)}")

    var = Xc_f.var(axis=0)
    hvg = np.argsort(var)[::-1][:2000]
    Xc_hvg = Xc_f[:, hvg]
    Xc_hvg = Xc_hvg - Xc_hvg.mean(axis=0, keepdims=True)
    n_comp = min(50, Xc_hvg.shape[0] - 1, Xc_hvg.shape[1])
    Xpca = PCA(n_components=n_comp, random_state=SEED).fit_transform(Xc_hvg)
    print(f"  PCA: {Xpca.shape}")

    # ---- contrast helper ----
    def run_contrast(name, label_a, mask_b_fn, b_label,
                     n_per_cap=600, k=15, n_edges_target=4000):
        print(f"\n--- contrast: {name} ---")
        mask_a = types_f == label_a
        mask_b = mask_b_fn(types_f)
        n_a_total = int(mask_a.sum())
        n_b_total = int(mask_b.sum())
        print(f"  total: {label_a} n={n_a_total}   {b_label} n={n_b_total}")
        if n_a_total < 30 or n_b_total < 30:
            print("  insufficient cells for contrast; skipping")
            return dict(name=name, label_a=label_a, label_b=b_label,
                        kappa_a=np.array([]), kappa_b=np.array([]),
                        p_t=np.nan, p_u=np.nan, delta=np.nan)
        n_per = min(n_a_total, n_b_total, n_per_cap)
        print(f"  per-group sample = {n_per}")
        rng = np.random.default_rng(SEED)
        idx_a_pool = np.where(mask_a)[0]
        idx_b_pool = np.where(mask_b)[0]
        sub_a = rng.choice(idx_a_pool, size=n_per, replace=False)
        sub_b = rng.choice(idx_b_pool, size=n_per, replace=False)
        sel = np.concatenate([sub_a, sub_b])
        Xs = Xpca[sel]
        G = build_knn_graph(Xs, k=k)
        n_edges = min(n_edges_target, G.number_of_edges())
        edge_kappa = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=rng)
        per_cell = per_cell_mean_curvature(G, edge_kappa)
        n_valid = sum(1 for x in edge_kappa.values() if not np.isnan(x))
        print(f"  |V|={G.number_of_nodes()} |E|={G.number_of_edges()}"
              f"  edges sampled={n_edges}  valid={n_valid}")

        kappa_a = np.array([per_cell[i] for i in range(len(sub_a))
                            if not np.isnan(per_cell[i])])
        kappa_b = np.array([per_cell[i] for i in range(len(sub_a),
                                                       len(sub_a) + len(sub_b))
                            if not np.isnan(per_cell[i])])
        if len(kappa_a) < 2 or len(kappa_b) < 2:
            return dict(name=name, label_a=label_a, label_b=b_label,
                        kappa_a=kappa_a, kappa_b=kappa_b,
                        p_t=np.nan, p_u=np.nan, delta=np.nan)
        t_stat, p_t = ttest_ind(kappa_a, kappa_b, equal_var=False)
        u_stat, p_u = mannwhitneyu(kappa_a, kappa_b)
        delta = cliffs_delta(kappa_a, kappa_b)
        print(f"  {label_a:18s} mean kappa = {kappa_a.mean():.4f}"
              f" +- {kappa_a.std():.4f}   n={len(kappa_a)}")
        print(f"  {b_label:18s} mean kappa = {kappa_b.mean():.4f}"
              f" +- {kappa_b.std():.4f}   n={len(kappa_b)}")
        print(f"  Welch t={t_stat:.3f} p={p_t:.4g}   "
              f"MWU U={u_stat:.1f} p={p_u:.4g}   Cliff's delta={delta:.3f}")
        return dict(name=name, label_a=label_a, label_b=b_label,
                    kappa_a=kappa_a, kappa_b=kappa_b,
                    p_t=float(p_t), p_u=float(p_u), delta=float(delta))

    results = []
    # contrast 1: malignant Epithelial vs T cells
    results.append(run_contrast(
        name="Epithelial vs Tcell",
        label_a="Epithelial",
        mask_b_fn=lambda ct: ct == "Tcell",
        b_label="Tcell",
    ))
    # contrast 2: vs Fibroblasts
    results.append(run_contrast(
        name="Epithelial vs Fibroblast",
        label_a="Epithelial",
        mask_b_fn=lambda ct: ct == "Fibroblast",
        b_label="Fibroblast",
    ))
    # contrast 3: vs Macrophages
    results.append(run_contrast(
        name="Epithelial vs Macrophage",
        label_a="Epithelial",
        mask_b_fn=lambda ct: ct == "Macrophage",
        b_label="Macrophage",
    ))
    # contrast 4: vs all non-epithelial (pooled non-malignant comparator)
    results.append(run_contrast(
        name="Epithelial vs Other",
        label_a="Epithelial",
        mask_b_fn=lambda ct: (ct != "Epithelial") & (ct != "Unknown"),
        b_label="Other",
    ))

    # ---- save summary CSV ----
    rows = []
    for r in results:
        for label, vals in [(r["label_a"], r["kappa_a"]),
                            (r["label_b"], r["kappa_b"])]:
            rows.append({"contrast": r["name"], "group": label,
                         "kappa_mean": float(vals.mean()) if len(vals) else np.nan,
                         "kappa_std": float(vals.std()) if len(vals) else np.nan,
                         "n_cells": int(len(vals)),
                         "p_welch": r["p_t"],
                         "p_mwu": r["p_u"],
                         "cliffs_delta": r["delta"]})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT / "n5c_ovarian_summary.csv", index=False)
    print(f"\nSaved {OUT / 'n5c_ovarian_summary.csv'}")

    # ---- distribution plot ----
    valid_results = [r for r in results if len(r["kappa_a"]) and len(r["kappa_b"])]
    if valid_results:
        fig, axes = plt.subplots(1, len(valid_results),
                                 figsize=(5.5 * len(valid_results), 4.5),
                                 squeeze=False)
        for i, r in enumerate(valid_results):
            ax = axes[0, i]
            ax.hist(r["kappa_a"], bins=30, alpha=0.55,
                    label=f'{r["label_a"]} (n={len(r["kappa_a"])})',
                    density=True, color="#cc3344")
            ax.hist(r["kappa_b"], bins=30, alpha=0.55,
                    label=f'{r["label_b"]} (n={len(r["kappa_b"])})',
                    density=True, color="#3366cc")
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
            ax.set_ylabel("density")
            ax.set_title(f"{r['name']}\n"
                         f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}  "
                         f"delta={r['delta']:.3f}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig_path = OUT / "n5c_ovarian_distributions.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"Saved {fig_path}")

    # ---- summary text ----
    REFERENCE = {
        "Tirosh melanoma": 0.74,
        "Puram HNSCC":     0.41,
        "Darmanis GBM":    0.38,
        "Li CRC":          0.20,
    }
    txt = []
    txt.append("N5c -- HGSOC (Olalekan 2021, GSE147082) Ollivier-Ricci summary")
    txt.append("=" * 64)
    txt.append(f"Seed: {SEED}")
    txt.append(f"Patients: {len(np.unique(patients))} ({', '.join(sorted(set(patients)))})")
    txt.append(f"Cells (post QC, post Unknown filter): {len(types_f)}")
    txt.append("Cell-type counts:")
    for k, v in counts.items():
        txt.append(f"  {k:14s} {v}")
    txt.append("")
    txt.append("Contrasts (Cliff's delta = Epithelial vs comparator;"
               " positive => malignant > comparator kappa):")
    for r in results:
        if not np.isfinite(r["delta"]):
            txt.append(f"  {r['name']:32s}  SKIPPED (insufficient cells)")
            continue
        txt.append(f"  {r['name']:32s}  delta={r['delta']:+.3f}  "
                   f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}  "
                   f"n_a={len(r['kappa_a'])} n_b={len(r['kappa_b'])}")
    txt.append("")
    txt.append("Reference Cliff's delta (malignant vs lymphocyte/immune):")
    for k, v in REFERENCE.items():
        txt.append(f"  {k:18s} +{v:.2f}")
    # use vs T cells if available, else vs Other
    primary = next((r for r in results
                    if r["name"] == "Epithelial vs Tcell" and np.isfinite(r["delta"])),
                   None)
    if primary is None:
        primary = next((r for r in results
                        if r["name"] == "Epithelial vs Other" and np.isfinite(r["delta"])),
                       None)
    if primary is not None:
        d = primary["delta"]
        if d > 0.30:
            verdict = "REPLICATE (delta in line with prior tumour types)"
        elif d > 0.10:
            verdict = "WEAK REPLICATE (smaller than Tirosh/Puram/Darmanis but same sign)"
        elif d > -0.10:
            verdict = "NULL (no detectable malignant-vs-comparator curvature shift)"
        else:
            verdict = "CONTRADICTS (malignant kappa < comparator kappa)"
        txt.append("")
        txt.append(f"HGSOC primary contrast ({primary['name']}): delta={d:+.3f}")
        txt.append(f"Verdict: {verdict}")
    txt.append("")
    txt.append(f"Wallclock: {time.time() - t0:.1f}s")
    summary_text = "\n".join(txt)
    (OUT / "n5c_summary.txt").write_text(summary_text + "\n")
    print("\n" + summary_text)
    print(f"\nSaved {OUT / 'n5c_summary.txt'}")


if __name__ == "__main__":
    main()
