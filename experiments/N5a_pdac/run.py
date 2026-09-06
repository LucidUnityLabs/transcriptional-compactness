"""
N5a — Per-cell Ollivier-Ricci curvature on Peng/Moncada PDAC scRNA-seq
(GSE111672, inDrop). Fifth tumor-type replication of the malignant >
non-malignant per-cell kappa direction (extends Tirosh melanoma,
Darmanis GBM, Puram HNSCC, Li CRC).

Source files (downloaded from
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/GSE111672/suppl/):
  GSE111672_PDAC-A-indrop-filtered-expMat.txt.gz   (1926 cells)
  GSE111672_PDAC-B-indrop-filtered-expMat.txt.gz   (1733 cells)

Both files have one row of cell-type labels as the column header. Across
both samples the malignant compartment is "Cancer clone A" + "Cancer
clone B" (~635 cells) and the largest non-malignant compartment is the
combined non-malignant ductal pool (~2480 cells). Non-malignant ductal
is the right comparator here, because Cancer clones in PDAC are
descended from ductal epithelium — using ductal puts the contrast at
the most stringent (closest neighbor in transcriptional space).

Pipeline (matches exp/E3_puram_hnscc and path1_hyperbolic/run_ricci.py):
  HVG-2000 by variance on log1p counts, PCA-50, kNN(k=15), sampled
  Ollivier-Ricci on 4000 edges (alpha=0.5, lazy walk), per-cell mean
  kappa.

Two contrasts:
  (a) Malignant vs all non-malignant
  (b) Malignant vs non-malignant Ductal (largest specific subtype)
With min(600, available) cells per arm, seed=20260508.
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
DATA_DIR = OUT.parent.parent / "data"
PDAC_A = DATA_DIR / "GSE111672_PDAC-A-indrop-filtered-expMat.txt.gz"
PDAC_B = DATA_DIR / "GSE111672_PDAC-B-indrop-filtered-expMat.txt.gz"
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


# ---------------- Peng/Moncada PDAC loader ----------------
def load_pdac_sample(path, sample_tag):
    """Read GSE111672 inDrop expmat (genes x cells, header row = labels).

    Returns:
      labels (np.array len=n_cells): author cell-type label per column
      gene_names (np.array len=n_genes)
      X (float32 n_genes x n_cells): raw counts
    """
    print(f"  reading {path.name} ({path.stat().st_size/1e6:.1f} MB) ...")
    # The first row is "Genes\t<label1>\t<label2>...". Use pandas with
    # tab sep. Duplicate column labels (intended) — we pull labels as the
    # raw header strings before pandas mangles, then drop them.
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    labels = np.array([h.strip() for h in header[1:]], dtype=object)

    df = pd.read_csv(path, sep="\t", compression="gzip", header=0,
                     low_memory=False)
    gene_names = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].values.astype(np.float32)
    assert X.shape[1] == len(labels), \
        f"{sample_tag}: {X.shape[1]} cells but {len(labels)} labels"
    print(f"    {sample_tag}: {X.shape[0]} genes x {X.shape[1]} cells")
    return labels, gene_names, X


def load_pdac_combined():
    la, ga, Xa = load_pdac_sample(PDAC_A, "PDAC-A")
    lb, gb, Xb = load_pdac_sample(PDAC_B, "PDAC-B")
    # Intersect gene sets and align (they should match — same pipeline).
    if list(ga) == list(gb):
        genes = ga
        X = np.concatenate([Xa, Xb], axis=1)
    else:
        common = pd.Index(ga).intersection(pd.Index(gb))
        ia = pd.Index(ga).get_indexer(common)
        ib = pd.Index(gb).get_indexer(common)
        genes = common.values
        X = np.concatenate([Xa[ia], Xb[ib]], axis=1)
    labels = np.concatenate([la, lb])
    sample = np.array(["PDAC-A"] * len(la) + ["PDAC-B"] * len(lb))
    print(f"  combined: {X.shape[0]} genes x {X.shape[1]} cells "
          f"(PDAC-A={len(la)} + PDAC-B={len(lb)})")
    return labels, sample, genes, X


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


def build_subset(X_genes_x_cells, mask_a, mask_b, label_a, label_b,
                 n_per=600, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    idx_a = rng.choice(np.where(mask_a)[0],
                       size=min(n_per, int(mask_a.sum())), replace=False)
    idx_b = rng.choice(np.where(mask_b)[0],
                       size=min(n_per, int(mask_b.sum())), replace=False)
    cells_idx = np.concatenate([idx_a, idx_b])
    labels = np.array([label_a] * len(idx_a) + [label_b] * len(idx_b))

    Xs = X_genes_x_cells[:, cells_idx]
    # log1p (counts -> log) then HVG by variance
    Xlog = np.log1p(Xs)
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xlog = Xlog[hvg]

    Xc = Xlog.T.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=min(50, Xc.shape[0] - 1, Xc.shape[1])).fit_transform(Xc)
    print(f"  subset: {Xpca.shape[0]} cells "
          f"({(labels == label_a).sum()} {label_a} / "
          f"{(labels == label_b).sum()} {label_b})")
    return Xpca, labels


def main():
    print("=" * 60)
    print("N5a Ollivier-Ricci on Peng/Moncada PDAC (GSE111672)")
    print("=" * 60)
    cell_labels, sample_tag, gene_names, X = load_pdac_combined()

    # Define malignant vs non-malignant. Cancer clone A/B == malignant.
    mal_set = {"Cancer clone A", "Cancer clone B"}
    mask_mal = np.array([l in mal_set for l in cell_labels])

    # Non-malignant ductal — closest transcriptional neighbor; sharpest contrast.
    ductal_set = {
        "Ductal - terminal ductal like",
        "Ductal - CRISP3 high/centroacinar like",
        "Ductal - MHC Class II",
        "Ductal - APOL1 high/hypoxic",
    }
    mask_duct = np.array([l in ductal_set for l in cell_labels])

    # All non-malignant (everything that isn't a Cancer clone).
    mask_non = ~mask_mal

    # Print compartment counts
    print("\n  cell-type counts (combined):")
    counts = pd.Series(cell_labels).value_counts()
    for k, v in counts.items():
        flag = ""
        if k in mal_set:
            flag = "  [MALIGNANT]"
        elif k in ductal_set:
            flag = "  [non-mal ductal]"
        print(f"    {k:45s} {v}{flag}")
    print(f"\n  malignant (Cancer clone A+B):  {int(mask_mal.sum())}")
    print(f"  non-malignant ductal:          {int(mask_duct.sum())}")
    print(f"  all non-malignant:             {int(mask_non.sum())}")

    if int(mask_mal.sum()) < 100 or int(mask_non.sum()) < 100:
        print("ERROR: insufficient cells in malignant or non-malignant arm")
        sys.exit(1)

    results = []

    # Contrast (a): malignant vs all non-malignant
    print("\n--- Contrast A: Malignant vs all non-malignant ---")
    Xa, la = build_subset(X, mask_mal, mask_non,
                          "Malignant", "Non-malignant",
                          n_per=600, rng=np.random.default_rng(SEED))
    results.append(analyze("PDAC: Malignant vs Non-malignant",
                           Xa, la, "Malignant", "Non-malignant",
                           rng=np.random.default_rng(SEED)))

    # Contrast (b): malignant vs non-malignant Ductal (closest comparator)
    print("\n--- Contrast B: Malignant vs Non-malignant Ductal ---")
    Xb, lb = build_subset(X, mask_mal, mask_duct,
                          "Malignant", "Ductal-non-mal",
                          n_per=600, rng=np.random.default_rng(SEED + 1))
    results.append(analyze("PDAC: Malignant vs Ductal-non-mal",
                           Xb, lb, "Malignant", "Ductal-non-mal",
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
    summary_path = OUT / "n5a_pdac_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    # plot
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[0, i]
        ax.hist(r["kappa_a"], bins=40, alpha=0.55, label=r["label_a"],
                density=True)
        ax.hist(r["kappa_b"], bins=40, alpha=0.55, label=r["label_b"],
                density=True)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
        ax.set_ylabel("density")
        delta = r["mean_a"] - r["mean_b"]
        ax.set_title(f"{r['name']}\n"
                     f"delta_mean={delta:+.4f}  Cliff d={r['cliffs_d']:.3f}\n"
                     f"Welch p={r['p_t']:.3g}  MWU p={r['p_u']:.3g}")
        ax.legend()
    plt.suptitle("Per-cell Ollivier-Ricci curvature, Peng/Moncada PDAC (GSE111672)")
    plt.tight_layout()
    fig_path = OUT / "n5a_pdac_distributions.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved {fig_path}")

    # text summary
    txt_path = OUT / "n5a_summary.txt"
    with open(txt_path, "w") as f:
        f.write("N5a — PDAC Ollivier-Ricci replication\n")
        f.write("=" * 60 + "\n")
        f.write("Dataset: GSE111672 (Peng/Moncada PDAC inDrop scRNA-seq)\n")
        f.write(f"  PDAC-A + PDAC-B combined\n")
        f.write(f"  total cells: {len(cell_labels)}\n")
        f.write(f"  malignant (Cancer clone A+B): {int(mask_mal.sum())}\n")
        f.write(f"  non-malignant ductal: {int(mask_duct.sum())}\n")
        f.write(f"  all non-malignant: {int(mask_non.sum())}\n")
        f.write(f"\nseed = {SEED}\n")
        f.write("pipeline: log1p, HVG-2000 by var, PCA-50, kNN k=15, "
                "OR alpha=0.5, 4000 sampled edges\n\n")
        for r in results:
            delta = r["mean_a"] - r["mean_b"]
            sign = "MAL > NON" if delta > 0 else "MAL < NON"
            f.write(f"{r['name']}\n")
            f.write(f"  {r['label_a']:30s}  "
                    f"kappa={r['mean_a']:.4f} +- {r['std_a']:.4f}  n={r['n_a']}\n")
            f.write(f"  {r['label_b']:30s}  "
                    f"kappa={r['mean_b']:.4f} +- {r['std_b']:.4f}  n={r['n_b']}\n")
            f.write(f"  delta_mean = {delta:+.4f}  ({sign})\n")
            f.write(f"  Welch t  p={r['p_t']:.4g}\n")
            f.write(f"  MWU U    p={r['p_u']:.4g}\n")
            f.write(f"  Cliff's delta = {r['cliffs_d']:+.3f}\n\n")
        f.write("Comparison anchors (per-cell kappa malignant > non-malignant):\n")
        f.write("  Tirosh melanoma   Cliff d = +0.74\n")
        f.write("  Puram HNSCC       Cliff d = +0.41\n")
        f.write("  Darmanis GBM      Cliff d = +0.38\n")
        f.write("  Li CRC            Cliff d = +0.20\n")
    print(f"Saved {txt_path}")

    # one-line direction summary
    print("\nDirection summary (kappa malignant - kappa control):")
    for r in results:
        delta = r["mean_a"] - r["mean_b"]
        sign = "MAL > NON" if delta > 0 else "MAL < NON"
        print(f"  {r['name']:50s}  delta={delta:+.4f}  Cliff d={r['cliffs_d']:+.3f}  ({sign})")


if __name__ == "__main__":
    main()
