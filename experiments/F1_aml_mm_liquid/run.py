"""
F1 — Per-cell Ollivier-Ricci curvature on Galen et al. 2019 AML scRNA-seq
(GSE116256, Smart-seq2 enriched profiling). Test whether the malignant >
non-malignant kappa direction (replicated across 7 solid tumor types,
Tirosh +0.74 -> PDAC +0.15) holds in a haematological cancer where
biology is fundamentally different: leukemic blasts vs healthy
HSPCs/myeloid lineage, no TME stromal compartment, often more
transcriptionally heterogeneous within the malignant compartment.

Datasets:
  * Diagnostic AML samples (six largest D0): AML556, AML419A, AML1012,
    AML328, AML210A, AML475 (~5451 malignant blasts total).
  * Healthy bone marrow controls: BM1-BM5 (~4149 normal myeloid lineage
    cells when restricted to HSC/Prog/GMP/ProMono/Mono).

Author labels:
  * PredictionRefined column: "malignant" | "normal" (per-cell, by
    genotype/CNV inference -- van Galen et al. 2019).
  * CellType column: HSC, Prog, GMP, ProMono, Mono, cDC, pDC, earlyEry,
    lateEry, ProB, B, Plasma, T, CTL, NK; with "-like" suffix on
    blasts (e.g. ProMono-like).

Pipeline (matches N5a/E3/run_ricci.py):
  HVG-2000 by variance on log1p counts, PCA-50, kNN(k=15), sampled
  Ollivier-Ricci on 4000 edges (alpha=0.5), per-cell mean kappa.

Two contrasts:
  (A) Malignant blasts vs healthy myeloid HSPCs (cell-type-matched:
      restrict healthy to HSC/Prog/GMP/ProMono/Mono pool — sharpest
      and most biology-faithful contrast).
  (B) Malignant blasts vs all healthy BM cells (broader pool: includes
      lymphoid/erythroid).

Hypothesis test:
  H_a: kappa_malignant > kappa_normal (same direction as solid tumors)
  H_b: kappa_malignant < kappa_normal (flipped — myeloid hierarchy
       collapse spreads blasts across multiple states; healthy clusters
       are tighter)
  H_c: no significant effect

Cells per arm: 600. Random seed: 20260508.
"""
import sys
import gzip
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
DATA_DIR = OUT.parent.parent / "data" / "GSE116256_galen"

# Six largest D0 (diagnostic) AML samples by malignant cell count.
# Format: (sample-suffix, tag). Files in DATA_DIR look like
# GSM<num>_<sample-suffix>.{dem,anno}.txt.gz; the dem/anno can have
# different GSM accession numbers, so we glob by sample-suffix.
AML_SAMPLES = [
    ("AML556-D0",  "AML556"),
    ("AML419A-D0", "AML419A"),
    ("AML1012-D0", "AML1012"),
    ("AML328-D0",  "AML328"),
    ("AML210A-D0", "AML210A"),
    ("AML475-D0",  "AML475"),
]
# Healthy bone marrow controls (BM1-5).
BM_SAMPLES = [
    ("BM1",         "BM1"),
    ("BM2",         "BM2"),
    ("BM3",         "BM3"),
    ("BM4",         "BM4"),
    ("BM5-34p",     "BM5-34p"),
    ("BM5-34p38n",  "BM5-34p38n"),
]

# Cell-type-matched myeloid pool: blasts mostly localize within these types.
MYELOID_TYPES = {"HSC", "Prog", "GMP", "ProMono", "Mono"}
# Author "-like" suffix marks malignant cell of that type.
MYELOID_LIKE = {f"{t}-like" for t in MYELOID_TYPES}

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
        sub_nodes = sorted(set(sup_u + sup_v))
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


def cliffs_delta(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return (gt - lt) / (na * nb)


# ---------------- Galen GSE116256 loader ----------------
def _find_file(suffix, kind):
    """kind in {'anno','dem'}; suffix like 'AML556-D0' or 'BM4'.

    Files have form GSM<num>_<suffix>.<kind>.txt.gz; the GSM number for
    dem and anno may differ, so we glob.
    """
    matches = sorted(DATA_DIR.glob(f"*_{suffix}.{kind}.txt.gz"))
    if not matches:
        raise FileNotFoundError(f"no {kind} file matching *_{suffix}.{kind}.txt.gz")
    return matches[0]


def load_anno(suffix):
    path = _find_file(suffix, "anno")
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return df


def load_dem(suffix):
    path = _find_file(suffix, "dem")
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    gene_names = df.iloc[:, 0].astype(str).values
    barcodes = np.array(df.columns[1:].tolist())
    X = df.iloc[:, 1:].values.astype(np.float32)
    return gene_names, X, barcodes


def build_pool(sample_list, malignant_status, allowed_celltypes=None):
    """
    Walk samples, pull cells matching the malignancy filter and (if given)
    cell-type filter. Returns:
      genes (np.array; common across samples)
      X (n_genes x n_cells float32)
      meta (DataFrame: sample, barcode, celltype, malignancy)
    """
    pieces = []
    metas = []
    common_genes = None
    for suffix, tag in sample_list:
        anno = load_anno(suffix)
        # PredictionRefined: "malignant" | "normal" | ""
        keep = anno["PredictionRefined"].astype(str).str.lower() == malignant_status
        if allowed_celltypes is not None:
            keep = keep & anno["CellType"].isin(allowed_celltypes)
        if int(keep.sum()) == 0:
            print(f"    {tag}: 0 cells matched -> skip")
            continue
        sel_barcodes = anno.loc[keep, "Cell"].astype(str).values
        sel_celltype = anno.loc[keep, "CellType"].astype(str).values
        sel_mal      = anno.loc[keep, "PredictionRefined"].astype(str).values
        # Load expression
        genes, X, all_bc = load_dem(suffix)
        bc_to_idx = {b: i for i, b in enumerate(all_bc)}
        sel_idx = [bc_to_idx[b] for b in sel_barcodes if b in bc_to_idx]
        sel_barcodes = np.array([b for b in sel_barcodes if b in bc_to_idx])
        if len(sel_idx) == 0:
            print(f"    {tag}: barcode mismatch -> skip")
            continue
        Xs = X[:, sel_idx]
        # Keep matching meta rows in same order
        bc_to_meta = {b: (c, m) for b, c, m in
                      zip(anno["Cell"].astype(str).values,
                          anno["CellType"].astype(str).values,
                          anno["PredictionRefined"].astype(str).values)}
        sel_celltype = np.array([bc_to_meta[b][0] for b in sel_barcodes])
        sel_mal      = np.array([bc_to_meta[b][1] for b in sel_barcodes])

        if common_genes is None:
            common_genes = pd.Index(genes)
        else:
            common_genes = common_genes.intersection(pd.Index(genes))
        pieces.append((genes, Xs))
        m = pd.DataFrame({
            "sample": tag,
            "barcode": sel_barcodes,
            "celltype": sel_celltype,
            "malignancy": sel_mal,
        })
        metas.append(m)
        print(f"    {tag}: {Xs.shape[1]} cells (status={malignant_status})")

    if not pieces:
        return None, None, None

    common_genes = list(common_genes)
    Xs_aligned = []
    for genes, Xs in pieces:
        idx = pd.Index(genes).get_indexer(common_genes)
        Xs_aligned.append(Xs[idx])
    X = np.concatenate(Xs_aligned, axis=1)
    meta = pd.concat(metas, ignore_index=True)
    return np.array(common_genes), X, meta


# ---------------- analysis driver ----------------
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

    print(f"  {label_a:30s}  kappa = {kappa_a.mean():.4f} +- {kappa_a.std():.4f}  (n={len(kappa_a)})")
    print(f"  {label_b:30s}  kappa = {kappa_b.mean():.4f} +- {kappa_b.std():.4f}  (n={len(kappa_b)})")
    print(f"  Welch t={t_stat:.3f} p={p_t:.4g}  MWU U={u_stat:.1f} p={p_u:.4g}  Cliff d={d:.3f}")

    return dict(name=name, label_a=label_a, label_b=label_b,
                kappa_a=kappa_a, kappa_b=kappa_b,
                mean_a=float(kappa_a.mean()), mean_b=float(kappa_b.mean()),
                std_a=float(kappa_a.std()), std_b=float(kappa_b.std()),
                n_a=int(len(kappa_a)), n_b=int(len(kappa_b)),
                p_t=float(p_t), p_u=float(p_u), cliffs_d=float(d))


def build_subset(X_genes_x_cells_a, X_genes_x_cells_b,
                 genes_a, genes_b,
                 label_a, label_b,
                 n_per=600, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    common = pd.Index(genes_a).intersection(pd.Index(genes_b))
    ia = pd.Index(genes_a).get_indexer(common)
    ib = pd.Index(genes_b).get_indexer(common)
    Xa = X_genes_x_cells_a[ia]
    Xb = X_genes_x_cells_b[ib]
    n_a = min(n_per, Xa.shape[1])
    n_b = min(n_per, Xb.shape[1])
    sel_a = rng.choice(Xa.shape[1], size=n_a, replace=False)
    sel_b = rng.choice(Xb.shape[1], size=n_b, replace=False)
    Xs = np.concatenate([Xa[:, sel_a], Xb[:, sel_b]], axis=1)
    labels = np.array([label_a] * n_a + [label_b] * n_b)
    Xlog = np.log1p(Xs)
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xlog = Xlog[hvg]
    Xc = Xlog.T.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=min(50, Xc.shape[0] - 1, Xc.shape[1]),
               random_state=SEED).fit_transform(Xc)
    print(f"  subset: {Xpca.shape[0]} cells "
          f"({(labels == label_a).sum()} {label_a} / "
          f"{(labels == label_b).sum()} {label_b})  "
          f"common genes={len(common)}")
    return Xpca, labels


def main():
    print("=" * 64)
    print("F1 Ollivier-Ricci on van Galen AML (GSE116256, liquid tumor)")
    print("=" * 64)

    # ---- Load malignant blasts (across diagnostic AML samples) ----
    print("\nLoading malignant AML blasts (D0 diagnostic samples):")
    genes_mal, X_mal, meta_mal = build_pool(
        AML_SAMPLES, malignant_status="malignant",
        allowed_celltypes=None,  # accept all malignant cells
    )
    if X_mal is None:
        print("ERROR: no malignant cells loaded"); sys.exit(1)
    print(f"  malignant pool: {X_mal.shape[1]} cells, {X_mal.shape[0]} genes")
    print("  malignant celltype distribution:")
    for ct, n in meta_mal["celltype"].value_counts().head(10).items():
        print(f"    {ct:20s} {n}")

    # ---- Load healthy myeloid HSPCs (cell-type-matched comparator) ----
    print("\nLoading healthy myeloid HSPCs from BM controls "
          "(HSC/Prog/GMP/ProMono/Mono):")
    genes_hm, X_hm, meta_hm = build_pool(
        BM_SAMPLES, malignant_status="normal",
        allowed_celltypes=MYELOID_TYPES,
    )
    if X_hm is None:
        print("ERROR: no healthy myeloid cells loaded"); sys.exit(1)
    print(f"  healthy myeloid pool: {X_hm.shape[1]} cells, {X_hm.shape[0]} genes")
    print("  celltype distribution:")
    for ct, n in meta_hm["celltype"].value_counts().items():
        print(f"    {ct:20s} {n}")

    # ---- Load all healthy BM (broader contrast) ----
    print("\nLoading all healthy BM cells (broader comparator):")
    genes_hall, X_hall, meta_hall = build_pool(
        BM_SAMPLES, malignant_status="normal",
        allowed_celltypes=None,
    )
    print(f"  all healthy BM pool: {X_hall.shape[1]} cells, {X_hall.shape[0]} genes")

    results = []

    # Contrast A: malignant vs healthy myeloid HSPCs (cell-type-matched)
    print("\n--- Contrast A: AML blasts vs healthy myeloid HSPCs ---")
    Xa, la = build_subset(X_mal, X_hm, genes_mal, genes_hm,
                          "AML-blasts", "Healthy-myeloid",
                          n_per=600, rng=np.random.default_rng(SEED))
    results.append(analyze("AML: blasts vs healthy myeloid",
                           Xa, la, "AML-blasts", "Healthy-myeloid",
                           rng=np.random.default_rng(SEED)))

    # Contrast B: malignant vs all healthy BM
    print("\n--- Contrast B: AML blasts vs all healthy BM ---")
    Xb, lb = build_subset(X_mal, X_hall, genes_mal, genes_hall,
                          "AML-blasts", "Healthy-BM-all",
                          n_per=600, rng=np.random.default_rng(SEED + 1))
    results.append(analyze("AML: blasts vs all healthy BM",
                           Xb, lb, "AML-blasts", "Healthy-BM-all",
                           rng=np.random.default_rng(SEED + 1)))

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
    summary_path = OUT / "f1_liquid_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # plot
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[0, i]
        ax.hist(r["kappa_a"], bins=40, alpha=0.55, label=r["label_a"],
                density=True, color="C3")
        ax.hist(r["kappa_b"], bins=40, alpha=0.55, label=r["label_b"],
                density=True, color="C0")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
        ax.set_ylabel("density")
        delta = r["mean_a"] - r["mean_b"]
        ax.set_title(f"{r['name']}\n"
                     f"delta_mean={delta:+.4f}  Cliff d={r['cliffs_d']:.3f}\n"
                     f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}")
        ax.legend()
    plt.suptitle("F1 - Per-cell Ollivier-Ricci, van Galen AML (GSE116256, liquid tumor)")
    plt.tight_layout()
    fig_path = OUT / "f1_liquid_distributions.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved {fig_path}")

    # text summary + verdict
    txt_path = OUT / "f1_summary.txt"
    with open(txt_path, "w") as f:
        f.write("F1 - Liquid-tumor test of Ollivier-Ricci kappa direction\n")
        f.write("=" * 64 + "\n")
        f.write("Dataset: GSE116256 (van Galen et al. 2019 AML scRNA-seq)\n")
        f.write(f"  AML diagnostic samples (D0): "
                f"{', '.join(t for _, t in AML_SAMPLES)}\n")
        f.write(f"  Healthy BM controls: "
                f"{', '.join(t for _, t in BM_SAMPLES)}\n")
        f.write(f"  malignant blast pool: {X_mal.shape[1]} cells\n")
        f.write(f"  healthy myeloid HSPC pool: {X_hm.shape[1]} cells\n")
        f.write(f"  all healthy BM pool: {X_hall.shape[1]} cells\n")
        f.write(f"\nseed = {SEED}\n")
        f.write("pipeline: log1p, HVG-2000 by var, PCA-50, "
                "kNN k=15, OR alpha=0.5, 4000 sampled edges\n\n")
        for r in results:
            delta = r["mean_a"] - r["mean_b"]
            sign = "MAL > NORM" if delta > 0 else "MAL < NORM"
            f.write(f"{r['name']}\n")
            f.write(f"  {r['label_a']:25s}  "
                    f"kappa={r['mean_a']:.4f} +- {r['std_a']:.4f}  n={r['n_a']}\n")
            f.write(f"  {r['label_b']:25s}  "
                    f"kappa={r['mean_b']:.4f} +- {r['std_b']:.4f}  n={r['n_b']}\n")
            f.write(f"  delta_mean = {delta:+.4f}  ({sign})\n")
            f.write(f"  Welch t  p={r['p_t']:.4g}\n")
            f.write(f"  MWU U    p={r['p_u']:.4g}\n")
            f.write(f"  Cliff's delta = {r['cliffs_d']:+.3f}\n\n")
        f.write("Solid-tumor anchor sweep (all kappa_malignant > kappa_normal):\n")
        f.write("  Tirosh melanoma   Cliff d = +0.74\n")
        f.write("  Puram HNSCC       Cliff d = +0.41\n")
        f.write("  Darmanis GBM      Cliff d = +0.38\n")
        f.write("  Li CRC            Cliff d = +0.20\n")
        f.write("  PDAC              Cliff d = +0.15\n\n")

        # verdict
        d_a = results[0]["cliffs_d"]
        d_b = results[1]["cliffs_d"]
        p_a = results[0]["p_u"]
        p_b = results[1]["p_u"]
        sig_a = p_a < 0.05
        sig_b = p_b < 0.05
        same_sign = (d_a < 0) == (d_b < 0)
        f.write("VERDICT (using cell-type-matched contrast A as primary):\n")
        if d_a > 0.10 and sig_a:
            f.write(f"  H_a supported: AML kappa > healthy "
                    f"(d_A={d_a:+.3f}, d_B={d_b:+.3f}). "
                    f"Direction agrees with solid-tumor sweep.\n")
        elif d_a < -0.10 and sig_a:
            f.write(f"  H_b supported: AML kappa < healthy "
                    f"(d_A={d_a:+.3f}, d_B={d_b:+.3f}). "
                    f"DIRECTION FLIPPED relative to solid tumors -- "
                    f"liquid-tumor biology (myeloid hierarchy collapse) "
                    f"reframes the result.\n")
        elif same_sign and d_a < 0 and sig_a:
            f.write(f"  Weak H_b (suggestive flip): contrast A is "
                    f"directionally flipped (d_A={d_a:+.3f}, "
                    f"MWU p={p_a:.4g}) and contrast B agrees in "
                    f"direction (d_B={d_b:+.3f}, MWU p={p_b:.4g}). "
                    f"Magnitude is far below the +0.15 (PDAC) "
                    f"solid-tumor floor and below the |0.10| "
                    f"negligible-effect threshold, so the signal is "
                    f"present but small. Read as: the per-cell "
                    f"malignancy-tightness signature does NOT transfer "
                    f"to liquid AML; instead AML blasts show a small "
                    f"inverse signature consistent with "
                    f"transcriptional spread across myeloid "
                    f"differentiation states (blasts span HSC/Prog/"
                    f"GMP/ProMono/Mono-like) versus tighter healthy "
                    f"HSPC clusters.\n")
        elif same_sign and d_a > 0 and sig_a:
            f.write(f"  Weak H_a: same-direction effect, magnitude "
                    f"(d_A={d_a:+.3f}) below solid floor.\n")
        else:
            f.write(f"  H_c supported: no consistent direction "
                    f"(d_A={d_a:+.3f}, d_B={d_b:+.3f}). "
                    f"Phenotype appears to be solid-tumor-specific.\n")
    print(f"Saved {txt_path}")

    print("\nDirection summary (kappa malignant - kappa control):")
    for r in results:
        delta = r["mean_a"] - r["mean_b"]
        sign = "MAL > NORM" if delta > 0 else "MAL < NORM"
        print(f"  {r['name']:50s}  "
              f"delta={delta:+.4f}  Cliff d={r['cliffs_d']:+.3f}  ({sign})")


if __name__ == "__main__":
    main()
