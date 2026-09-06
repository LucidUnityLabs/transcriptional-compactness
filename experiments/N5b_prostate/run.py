"""
N5b -- Per-cell Ollivier-Ricci curvature on Chen et al. 2021 prostate
adenocarcinoma scRNA-seq (GSE176031). Sixth tumor type to test whether the
malignant > non-malignant per-cell kappa direction replicates beyond
Tirosh melanoma, Darmanis GBM, Puram HNSCC, Li CRC, and the N5a PDAC run.

GSE176031 ships per-sample digital expression matrices (DGE, genes x cells,
UMI counts) with no central cell-type metadata. We therefore:
  1. concatenate the primary tumor (T) and matched normal (N) DGE matrices
     for four PRAD patients (PR5249/PR5251/PR5254/PR5261). Organoids and
     PB benign biopsies are excluded.
  2. log-normalize, then score each cell with prostate-specific marker
     panels (luminal, basal, T-cell, myeloid, fibroblast).
  3. assign a coarse cell-type call per cell and tag tumor-tissue luminal
     calls as "Malignant" (the Chen paper shows luminal-lineage cells are
     the malignant-of-origin compartment in PRAD); normal-tissue luminal
     calls as "Benign luminal"; T-cell-marker calls as "Immune (T)".
  4. run the standard pipeline (HVG-2000, PCA-50, kNN k=15, OR alpha=0.5,
     4000 sampled edges) for two contrasts:
        A: Malignant vs Benign luminal  (cell-type-matched, tight reference)
        B: Malignant vs Immune (T)      (looser reference, parallel to N5a/E3)

Prostate luminal epithelium is glandular and clonally tight even in benign
tissue, so contrast A may show the smallest kappa separation of the panel.
Seed = 20260508.
"""
import os
import sys
import gzip
import re
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
from pathlib import Path

OUT = Path(__file__).parent
DATA_DIR = OUT.parent.parent / "data" / "GSE176031_chen"
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
        out[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan
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
    na, nb = len(a), len(b)
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return (gt - lt) / (na * nb)


# ---------------- Chen GSE176031 loader ----------------
# Marker panels for prostate scRNA-seq (curated from Chen 2021, Karthaus 2020,
# Henry 2018; standard PRAD literature).
MARKERS = {
    "luminal":  ["KLK3", "KLK2", "AR", "MSMB", "ACPP", "NKX3-1", "KRT8", "KRT18", "EPCAM"],
    "basal":    ["KRT5", "KRT14", "TP63", "KRT15", "DST"],
    "tcell":    ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "PTPRC", "TRAC"],
    "myeloid":  ["CD14", "CD68", "LYZ", "C1QA", "C1QB", "AIF1", "CSF1R"],
    "fibro":    ["DCN", "COL1A1", "COL1A2", "LUM", "PDGFRA", "ACTA2"],
    "endo":     ["VWF", "PECAM1", "CDH5", "ENG", "CLDN5"],
}


def parse_sample_meta(fname):
    """Patient and tissue (T/N) from filenames like
       'GSM5353232_PA_PR5249_T1_S3_L001_dge.txt.gz'.
    """
    m = re.search(r"PR(\d+)_([TN])(\d)?", fname)
    if not m:
        return None, None
    return f"PR{m.group(1)}", m.group(2)


def load_chen():
    """Return UMI matrix (genes x cells) plus per-cell metadata."""
    print(f"Loading Chen GSE176031 from {DATA_DIR} ...")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt.gz"))
    print(f"  {len(files)} sample files found")

    dfs = []
    meta_rows = []
    for f in files:
        patient, tissue = parse_sample_meta(f)
        if patient is None:
            print(f"  SKIP unparseable: {f}")
            continue
        path = DATA_DIR / f
        df = pd.read_csv(path, sep="\t", index_col=0)
        # rename columns to be globally unique: <sample>__<barcode>
        df.columns = [f"{f.split('_dge')[0]}__{c}" for c in df.columns]
        dfs.append(df)
        for c in df.columns:
            meta_rows.append({"cell_id": c, "patient": patient,
                              "tissue": tissue, "sample": f.split("_dge")[0]})
        print(f"  {f.split('_dge')[0]:55s} patient={patient} tissue={tissue}  cells={df.shape[1]}")

    print("  concatenating across samples (gene union, fillna=0) ...")
    full = pd.concat(dfs, axis=1, join="outer").fillna(0).astype(np.float32)
    meta = pd.DataFrame(meta_rows).set_index("cell_id").loc[full.columns]
    print(f"  combined: {full.shape[0]} genes x {full.shape[1]} cells")
    return full, meta


def assign_celltype(expr_log, meta, marker_dict):
    """Score each cell by mean log-norm expression over marker panels;
    assign to argmax. Cells whose top score is below a small threshold are
    flagged 'Unknown'.

    expr_log: pandas DataFrame, log1p(CP10K) genes x cells.
    """
    scores = {}
    for ct, markers in marker_dict.items():
        avail = [g for g in markers if g in expr_log.index]
        if not avail:
            scores[ct] = np.full(expr_log.shape[1], -np.inf)
            continue
        scores[ct] = expr_log.loc[avail].mean(axis=0).values
    score_df = pd.DataFrame(scores, index=expr_log.columns)

    # z-score each panel across cells (so panels are comparable scale)
    zs = (score_df - score_df.mean(axis=0)) / (score_df.std(axis=0) + 1e-9)
    call = zs.idxmax(axis=1).values
    top_z = zs.max(axis=1).values
    # Confidence: require top z >= 0.5 and lead over runner-up >= 0.25
    sorted_z = np.sort(zs.values, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]
    confident = (top_z >= 0.5) & (margin >= 0.25)
    call = np.where(confident, call, "Unknown")

    out = meta.copy()
    out["celltype"] = call
    out["max_z"] = top_z
    out["margin"] = margin
    for ct in marker_dict:
        out[f"score_{ct}"] = score_df[ct].values
    return out


def normalize(counts_genes_x_cells):
    """CP10K normalize (per cell) then log1p."""
    X = counts_genes_x_cells.values  # genes x cells
    libsize = X.sum(axis=0)
    libsize = np.where(libsize == 0, 1, libsize)
    Xn = X / libsize[None, :] * 1e4
    Xlog = np.log1p(Xn).astype(np.float32)
    return pd.DataFrame(Xlog, index=counts_genes_x_cells.index,
                        columns=counts_genes_x_cells.columns)


def build_subset(X_log_genes_x_cells, mask_a, mask_b, label_a, label_b,
                 n_per=600, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    idx_a = rng.choice(np.where(mask_a)[0],
                       size=min(n_per, int(mask_a.sum())), replace=False)
    idx_b = rng.choice(np.where(mask_b)[0],
                       size=min(n_per, int(mask_b.sum())), replace=False)
    cells_idx = np.concatenate([idx_a, idx_b])
    labels = np.array([label_a] * len(idx_a) + [label_b] * len(idx_b))

    Xs = X_log_genes_x_cells.values[:, cells_idx]  # genes x cells (already log)
    var = Xs.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xs = Xs[hvg]

    Xc = Xs.T.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50).fit_transform(Xc)
    print(f"  subset: {Xpca.shape[0]} cells "
          f"({(labels == label_a).sum()} {label_a} / "
          f"{(labels == label_b).sum()} {label_b})")
    return Xpca, labels


def analyze(name, X_pca, labels, label_a, label_b, k=15, n_edges=4000, rng=None):
    print(f"\n[{name}]  building kNN (k={k}) graph ...")
    G = build_knn_graph(X_pca, k=k)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Ollivier-Ricci on {n_edges} sampled edges ...")
    kappa = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=rng)
    n_valid = sum(1 for v in kappa.values() if not np.isnan(v))
    print(f"  valid edges: {n_valid}/{len(kappa)}")
    per_cell = per_cell_mean_curvature(G, kappa)

    cells_a = [i for i, l in enumerate(labels) if l == label_a]
    cells_b = [i for i, l in enumerate(labels) if l == label_b]
    kappa_a = np.array([per_cell[c] for c in cells_a if not np.isnan(per_cell[c])])
    kappa_b = np.array([per_cell[c] for c in cells_b if not np.isnan(per_cell[c])])

    t_stat, p_t = ttest_ind(kappa_a, kappa_b, equal_var=False)
    u_stat, p_u = mannwhitneyu(kappa_a, kappa_b)
    d = cliffs_delta(kappa_a, kappa_b)

    print(f"  {label_a:25s}  kappa = {kappa_a.mean():+.4f} +- {kappa_a.std():.4f}  (n={len(kappa_a)})")
    print(f"  {label_b:25s}  kappa = {kappa_b.mean():+.4f} +- {kappa_b.std():.4f}  (n={len(kappa_b)})")
    print(f"  Welch t={t_stat:.3f} p={p_t:.4g}  MWU U={u_stat:.1f} p={p_u:.4g}  Cliff d={d:.3f}")

    return dict(name=name, label_a=label_a, label_b=label_b,
                kappa_a=kappa_a, kappa_b=kappa_b,
                mean_a=float(kappa_a.mean()), mean_b=float(kappa_b.mean()),
                std_a=float(kappa_a.std()), std_b=float(kappa_b.std()),
                n_a=int(len(kappa_a)), n_b=int(len(kappa_b)),
                p_t=float(p_t), p_u=float(p_u), cliffs_d=float(d))


def main():
    print("=" * 60)
    print("N5b Ollivier-Ricci on Chen 2021 prostate adeno (GSE176031)")
    print("=" * 60)

    counts, meta = load_chen()

    # QC: minimum genes per cell
    n_genes = (counts.values > 0).sum(axis=0)
    n_umi = counts.values.sum(axis=0)
    keep = (n_genes >= 200) & (n_umi >= 500)
    print(f"\nQC: {keep.sum()}/{len(keep)} cells pass (>=200 genes, >=500 UMI)")
    counts = counts.iloc[:, keep]
    meta = meta.iloc[keep]

    print("\nNormalising (CP10K + log1p) ...")
    expr_log = normalize(counts)

    print("\nMarker-based cell-type assignment ...")
    meta = assign_celltype(expr_log, meta, MARKERS)
    print("\n  Coarse cell-type counts (overall):")
    print(meta["celltype"].value_counts().to_string())
    print("\n  Cell-type x tissue cross-tab:")
    print(pd.crosstab(meta["celltype"], meta["tissue"]).to_string())

    # Define analysis groups.
    # "Malignant" = luminal cells from tumor (T) tissue.
    # "Benign luminal" = luminal cells from normal (N) tissue.
    # "Immune (T)" = tcell calls from anywhere (not enough of either tissue alone).
    is_lum = (meta["celltype"] == "luminal").values
    is_bas = (meta["celltype"] == "basal").values
    is_t = (meta["tissue"] == "T").values
    is_n = (meta["tissue"] == "N").values
    is_tcell = (meta["celltype"] == "tcell").values

    mask_mal = is_lum & is_t
    mask_benlum = is_lum & is_n
    mask_immune = is_tcell  # both tissues
    mask_basal = is_bas  # benign basal pooled across tissues, secondary check

    print(f"\n  Malignant (luminal in T):     {mask_mal.sum()}")
    print(f"  Benign luminal (luminal in N): {mask_benlum.sum()}")
    print(f"  Immune (T cells, all tissue):  {mask_immune.sum()}")
    print(f"  Basal epithelium (all tissue): {mask_basal.sum()}")

    # Need at least ~150 in each group to have signal; aim for 600 each.
    n_per = 600
    results = []

    # Contrast A: Malignant vs Benign luminal (cell-type-matched, tight reference)
    if mask_mal.sum() >= 100 and mask_benlum.sum() >= 100:
        print("\n--- Contrast A: Malignant (T-luminal) vs Benign luminal (N-luminal) ---")
        Xa, la = build_subset(expr_log, mask_mal, mask_benlum,
                              "Malignant", "Benign-luminal",
                              n_per=n_per, rng=np.random.default_rng(SEED))
        results.append(analyze("Chen PRAD: Malignant vs Benign luminal",
                               Xa, la, "Malignant", "Benign-luminal",
                               rng=np.random.default_rng(SEED)))
    else:
        print("\n--- Contrast A skipped (insufficient cells) ---")

    # Contrast B: Malignant vs Immune (T cells)
    if mask_mal.sum() >= 100 and mask_immune.sum() >= 100:
        print("\n--- Contrast B: Malignant (T-luminal) vs Immune (T cells) ---")
        Xb, lb = build_subset(expr_log, mask_mal, mask_immune,
                              "Malignant", "Immune",
                              n_per=n_per, rng=np.random.default_rng(SEED + 1))
        results.append(analyze("Chen PRAD: Malignant vs Immune",
                               Xb, lb, "Malignant", "Immune",
                               rng=np.random.default_rng(SEED + 1)))
    else:
        print("\n--- Contrast B skipped (insufficient cells) ---")

    # Optional Contrast C: Malignant vs Basal epithelium (different epithelial lineage)
    if mask_mal.sum() >= 100 and mask_basal.sum() >= 100:
        print("\n--- Contrast C: Malignant (T-luminal) vs Basal epithelium ---")
        Xc, lc = build_subset(expr_log, mask_mal, mask_basal,
                              "Malignant", "Basal",
                              n_per=n_per, rng=np.random.default_rng(SEED + 2))
        results.append(analyze("Chen PRAD: Malignant vs Basal",
                               Xc, lc, "Malignant", "Basal",
                               rng=np.random.default_rng(SEED + 2)))

    # ---------------- save outputs ----------------
    rows = []
    for r in results:
        for label, vals, mean, std, n in [
            (r["label_a"], r["kappa_a"], r["mean_a"], r["std_a"], r["n_a"]),
            (r["label_b"], r["kappa_b"], r["mean_b"], r["std_b"], r["n_b"]),
        ]:
            rows.append({
                "contrast": f"{r['label_a']} vs {r['label_b']}",
                "group": label,
                "kappa_mean": mean,
                "kappa_std": std,
                "n_cells": n,
                "p_welch": r["p_t"],
                "p_mwu": r["p_u"],
                "cliffs_delta": r["cliffs_d"],
            })
    summary = pd.DataFrame(rows)
    summary_path = OUT / "n5b_prostate_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # plot
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                                 squeeze=False)
        for i, r in enumerate(results):
            ax = axes[0, i]
            ax.hist(r["kappa_a"], bins=40, alpha=0.55, label=r["label_a"], density=True)
            ax.hist(r["kappa_b"], bins=40, alpha=0.55, label=r["label_b"], density=True)
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
            ax.set_ylabel("density")
            delta = r["mean_a"] - r["mean_b"]
            ax.set_title(f"{r['name']}\n"
                         f"delta_mean={delta:+.4f}  Cliff d={r['cliffs_d']:.3f}\n"
                         f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}")
            ax.legend()
        plt.suptitle("Per-cell Ollivier-Ricci, Chen 2021 PRAD (GSE176031)")
        plt.tight_layout()
        fig_path = OUT / "n5b_prostate_distributions.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"Saved {fig_path}")

    # text summary
    summary_txt = OUT / "n5b_summary.txt"
    with open(summary_txt, "w") as fh:
        fh.write("N5b Prostate (Chen 2021 GSE176031) Ollivier-Ricci summary\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Seed: {SEED}\n")
        fh.write(f"Cells passing QC: {len(meta)}\n")
        fh.write("Cell-type counts (overall):\n")
        for k_, v in meta["celltype"].value_counts().items():
            fh.write(f"  {k_:12s} {v}\n")
        fh.write("\nCell-type x tissue:\n")
        fh.write(pd.crosstab(meta["celltype"], meta["tissue"]).to_string())
        fh.write("\n\nResults:\n")
        for r in results:
            delta = r["mean_a"] - r["mean_b"]
            sign = "MAL > NON" if delta > 0 else "MAL < NON"
            fh.write(f"\n{r['name']}\n")
            fh.write(f"  {r['label_a']:18s} kappa = {r['mean_a']:+.4f} +- {r['std_a']:.4f}  (n={r['n_a']})\n")
            fh.write(f"  {r['label_b']:18s} kappa = {r['mean_b']:+.4f} +- {r['std_b']:.4f}  (n={r['n_b']})\n")
            fh.write(f"  delta_mean = {delta:+.4f}  ({sign})\n")
            fh.write(f"  Cliff's delta = {r['cliffs_d']:+.3f}\n")
            fh.write(f"  Welch p = {r['p_t']:.4g}    MWU p = {r['p_u']:.4g}\n")
        fh.write("\nReference panel (per-cell mean kappa, malignant - non-malignant):\n")
        fh.write("  Tirosh   melanoma (E1):  Cliff +0.74\n")
        fh.write("  Puram    HNSCC    (E3):  Cliff +0.41\n")
        fh.write("  Darmanis GBM      (P1):  Cliff +0.38\n")
        fh.write("  Li       CRC      (E4):  Cliff +0.20\n")
    print(f"Saved {summary_txt}")

    # Direction summary
    print("\nDirection summary (kappa malignant - kappa control):")
    for r in results:
        delta = r["mean_a"] - r["mean_b"]
        sign = "MAL > NON" if delta > 0 else "MAL < NON"
        print(f"  {r['name']:55s}  delta={delta:+.4f}  Cliff_d={r['cliffs_d']:+.3f}  ({sign})")


if __name__ == "__main__":
    main()
