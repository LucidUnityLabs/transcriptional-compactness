"""
T4 — Visium spatial replication of the Ollivier-Ricci kappa malignant-vs-stroma
direction.

Question: does the per-cell kappa malignant > stroma direction observed on a
transcriptomic-similarity kNN graph also appear on a *physical* Delaunay/k=6
neighbor graph over Visium spots?  If yes -> substrate-independent biological
organization principle; if only on G_trans -> transcriptional-similarity
artifact.

Dataset: 10X Visium CytAssist Human Breast Cancer FFPE (filtered feature
matrix + tissue positions in ../../data/visium/).

Pipeline:
  1) Load h5 + tissue_positions.csv, restrict to in_tissue==1 spots and
     Gene Expression features.
  2) Normalize_total + log1p + HVG=2000 (seurat_v3 on raw counts) + scale +
     PCA-50 with seed.
  3) Score each spot for tumor vs stroma signatures (KRT8/KRT18/KRT19/EPCAM
     /ESR1 vs CD45=PTPRC, COL1A1, ACTA2, PECAM1) using scanpy.tl.score_genes;
     label malignant if tumor>stroma & tumor>median; stroma if stroma>tumor &
     stroma>median; rest unlabeled.
  4) Build G_phys: Delaunay triangulation on (array_row, array_col) keeping
     edges shorter than 1.5 * median Delaunay edge length (drops cross-tissue
     long edges); weight = Euclidean distance.
  5) Build G_trans: kNN k=15 in PCA-50, mutual edges with weight = Euclidean
     PCA distance.
  6) Sample 4000 edges per graph; lazy walk alpha=0.5 OR kappa per edge
     -> per-spot mean kappa.
  7) Welch t-test, MWU, Cliff's delta on malignant vs stroma per graph.
  8) Outputs: t4_visium_summary.csv, t4_visium_spatial.png,
     t4_visium_distributions.png and a stdout report.
"""

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "visium"
SEED = 20260507
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)


# ---------- OR kappa (same as exp/E1_within_patient/run.py) -----------------
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


def per_node_mean_curvature(G, edge_kappa):
    by = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k):
            continue
        by[u].append(k)
        by[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by.items()}


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    B = np.sort(b)
    n_less = np.searchsorted(B, a, side="left")
    n_leq = np.searchsorted(B, a, side="right")
    n_greater = len(B) - n_leq
    gt = n_less.sum(); lt = n_greater.sum()
    return float((gt - lt) / (len(a) * len(b)))


# ---------- data loading -----------------------------------------------------
def load_visium():
    h5_path = DATA / "filtered_feature_bc_matrix.h5"
    pos_path = DATA / "spatial" / "tissue_positions.csv"
    sf_path = DATA / "spatial" / "scalefactors_json.json"

    print(f"[load] {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = m["shape"][:]
        barcodes = [b.decode() for b in m["barcodes"][:]]
        feat = m["features"]
        feat_id = [b.decode() for b in feat["id"][:]]
        feat_name = [b.decode() for b in feat["name"][:]]
        feat_type = [b.decode() for b in feat["feature_type"][:]]
    # CSC -> CSR (cells x genes)
    n_genes, n_cells = int(shape[0]), int(shape[1])
    from scipy.sparse import csc_matrix
    X_csc = csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
    X = X_csc.T.tocsr().astype(np.float32)  # cells x genes
    print(f"  matrix: {X.shape} (spots x features)")

    var = pd.DataFrame({
        "gene_ids": feat_id,
        "feature_types": feat_type,
    }, index=feat_name)
    obs = pd.DataFrame(index=barcodes)
    adata = AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    # Use Gene Expression features only (drop antibody capture)
    is_ge = adata.var["feature_types"].values == "Gene Expression"
    adata = adata[:, is_ge].copy()
    print(f"  after Gene Expression filter: {adata.shape}")

    # Tissue positions
    pos = pd.read_csv(pos_path)
    pos = pos.set_index("barcode")
    keep = pos.loc[adata.obs.index]
    adata.obs["in_tissue"] = keep["in_tissue"].astype(int).values
    adata.obs["array_row"] = keep["array_row"].astype(int).values
    adata.obs["array_col"] = keep["array_col"].astype(int).values
    adata.obs["pxl_row"] = keep["pxl_row_in_fullres"].astype(float).values
    adata.obs["pxl_col"] = keep["pxl_col_in_fullres"].astype(float).values

    n_before = adata.n_obs
    adata = adata[adata.obs["in_tissue"] == 1].copy()
    print(f"  in_tissue==1: {adata.n_obs}/{n_before}")

    sf = json.loads(sf_path.read_text())
    return adata, sf


# ---------- preprocessing ----------------------------------------------------
def preprocess(adata, n_hvg=2000, n_pcs=50, seed=SEED):
    print("[preprocess] normalize_total + log1p + HVG + PCA")
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # cell_ranger flavor on log-normalized X (no skmisc dependency)
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=n_hvg)
    # subset to HVGs and scale
    adata_h = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_h, max_value=10)
    X_hvg = adata_h.X
    if hasattr(X_hvg, "toarray"):
        X_hvg = X_hvg.toarray()
    pca = PCA(n_components=n_pcs, random_state=seed).fit(X_hvg)
    adata.obsm["X_pca"] = pca.transform(X_hvg).astype(np.float32)
    print(f"  PCA: {adata.obsm['X_pca'].shape}")
    return adata


# ---------- labelling --------------------------------------------------------
TUMOR_GENES = ["KRT8", "KRT18", "KRT19", "EPCAM", "ESR1"]
STROMA_GENES = ["PTPRC", "COL1A1", "ACTA2", "PECAM1"]


def label_spots(adata):
    print("[label] scoring tumor vs stroma signatures")
    avail_t = [g for g in TUMOR_GENES if g in adata.var_names]
    avail_s = [g for g in STROMA_GENES if g in adata.var_names]
    print(f"  tumor markers found: {avail_t}")
    print(f"  stroma markers found: {avail_s}")
    if not avail_t or not avail_s:
        raise RuntimeError("Missing key marker genes")
    sc.tl.score_genes(adata, gene_list=avail_t, score_name="tumor_score",
                      random_state=SEED)
    sc.tl.score_genes(adata, gene_list=avail_s, score_name="stroma_score",
                      random_state=SEED)
    t = adata.obs["tumor_score"].values
    s = adata.obs["stroma_score"].values
    label = np.array(["unlabeled"] * adata.n_obs, dtype=object)
    diff = t - s
    # high-confidence assignment: use 33/66 quantiles of (t - s).
    q_lo, q_hi = np.quantile(diff, [0.33, 0.67])
    label[diff >= q_hi] = "malignant"
    label[diff <= q_lo] = "stroma"
    adata.obs["label"] = label
    counts = pd.Series(label).value_counts()
    print(f"  labels: {counts.to_dict()}")
    return adata


# ---------- graphs -----------------------------------------------------------
def build_phys_graph(adata):
    """Delaunay triangulation on (pxl_row, pxl_col); drop edges longer than
    1.5x median Delaunay edge length (kills cross-tissue jumps).
    """
    pts = np.column_stack([adata.obs["pxl_row"].values,
                           adata.obs["pxl_col"].values])
    tri = Delaunay(pts)
    edges = set()
    for simp in tri.simplices:
        for i in range(3):
            a, b = simp[i], simp[(i + 1) % 3]
            if a > b:
                a, b = b, a
            edges.add((int(a), int(b)))
    edges = np.array(sorted(edges))
    d = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    cutoff = 1.5 * np.median(d)
    keep = d <= cutoff
    edges = edges[keep]
    d = d[keep]
    G = nx.Graph()
    G.add_nodes_from(range(adata.n_obs))
    for (u, v), w in zip(edges, d):
        G.add_edge(int(u), int(v), weight=float(w))
    print(f"[G_phys] Delaunay edges kept: {G.number_of_edges()} "
          f"(median d={np.median(d):.1f}, cutoff={cutoff:.1f})")
    return G


def build_trans_graph(adata, k=15):
    X = adata.obsm["X_pca"]
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, inds = nbr.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    for i in range(X.shape[0]):
        for j_idx in range(1, k + 1):
            j = int(inds[i, j_idx]); d = float(dists[i, j_idx])
            if not G.has_edge(i, j) or G[i][j].get("weight", np.inf) > d:
                G.add_edge(i, j, weight=d)
    print(f"[G_trans] kNN(k={k}): {G.number_of_edges()} edges")
    return G


# ---------- analysis ---------------------------------------------------------
def analyse(name, G, label, n_edges=4000):
    print(f"\n[{name}] OR kappa on {n_edges} sampled edges")
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=RNG)
    n_valid = sum(1 for v in edge_k.values() if not np.isnan(v))
    print(f"  valid: {n_valid}/{len(edge_k)}")
    per = per_node_mean_curvature(G, edge_k)
    kappa = np.array([per.get(i, np.nan) for i in range(len(label))])
    n_with_k = np.isfinite(kappa).sum()
    print(f"  spots with valid kappa: {n_with_k}/{len(label)}")
    a = kappa[(label == "malignant") & np.isfinite(kappa)]
    b = kappa[(label == "stroma") & np.isfinite(kappa)]
    if len(a) == 0 or len(b) == 0:
        return kappa, dict(name=name, n_mal=int(len(a)), n_str=int(len(b)),
                           mean_mal=float("nan"), mean_str=float("nan"),
                           diff=float("nan"), welch_t=float("nan"),
                           welch_p=float("nan"), mwu_U=float("nan"),
                           mwu_p=float("nan"), cliff_d=float("nan"))
    welch_t, welch_p = ttest_ind(a, b, equal_var=False)
    mwu_U, mwu_p = mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    print(f"  mean kappa  malignant={a.mean():+.4f} (n={len(a)})  "
          f"stroma={b.mean():+.4f} (n={len(b)})  diff={a.mean()-b.mean():+.4f}")
    print(f"  Welch t={welch_t:.3f} p={welch_p:.3g}  "
          f"MWU U={mwu_U:.0f} p={mwu_p:.3g}  Cliff d={d:+.3f}")
    return kappa, dict(name=name, n_mal=int(len(a)), n_str=int(len(b)),
                       mean_mal=float(a.mean()), mean_str=float(b.mean()),
                       diff=float(a.mean() - b.mean()),
                       welch_t=float(welch_t), welch_p=float(welch_p),
                       mwu_U=float(mwu_U), mwu_p=float(mwu_p),
                       cliff_d=float(d))


# ---------- plotting ---------------------------------------------------------
def plot_spatial(adata, kappa_phys, label, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    pr = adata.obs["pxl_row"].values
    pc = adata.obs["pxl_col"].values
    ax = axes[0]
    valid = np.isfinite(kappa_phys)
    sc1 = ax.scatter(pc[valid], -pr[valid], c=kappa_phys[valid], cmap="RdBu_r",
                     s=8, vmin=np.nanpercentile(kappa_phys, 2),
                     vmax=np.nanpercentile(kappa_phys, 98))
    plt.colorbar(sc1, ax=ax, label="kappa (G_phys)")
    ax.set_title(f"Per-spot Ollivier-Ricci kappa on G_phys "
                 f"(n={int(valid.sum())} spots)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    color = {"malignant": "#d62728", "stroma": "#1f77b4",
             "unlabeled": "#cccccc"}
    for lab in ("unlabeled", "stroma", "malignant"):
        m = label == lab
        ax.scatter(pc[m], -pr[m], c=color[lab], s=8, label=f"{lab} (n={m.sum()})")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Spot labels (tumor vs stroma signature)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()


def plot_distributions(kappa_phys, kappa_trans, label, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, kappa, name in zip(axes, (kappa_phys, kappa_trans),
                                ("G_phys", "G_trans")):
        a = kappa[(label == "malignant") & np.isfinite(kappa)]
        b = kappa[(label == "stroma") & np.isfinite(kappa)]
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 40)
        ax.hist(b, bins=bins, alpha=0.55, label=f"stroma n={len(b)}",
                color="#1f77b4")
        ax.hist(a, bins=bins, alpha=0.55, label=f"malignant n={len(a)}",
                color="#d62728")
        ax.axvline(a.mean(), color="#d62728", ls="--")
        ax.axvline(b.mean(), color="#1f77b4", ls="--")
        ax.set_title(f"{name} kappa: mal mean={a.mean():+.3f}, "
                     f"str mean={b.mean():+.3f}, "
                     f"diff={a.mean()-b.mean():+.3f}")
        ax.set_xlabel("per-spot mean kappa"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close()


def main():
    print("=" * 72)
    print("T4 — Visium spatial replication of OR kappa malignant-vs-stroma")
    print(f"     seed={SEED}")
    print("=" * 72)

    adata, sf = load_visium()
    adata = preprocess(adata)
    adata = label_spots(adata)

    G_phys = build_phys_graph(adata)
    G_trans = build_trans_graph(adata, k=15)

    label = adata.obs["label"].values

    kappa_phys, row_phys = analyse("G_phys", G_phys, label, n_edges=4000)
    kappa_trans, row_trans = analyse("G_trans", G_trans, label, n_edges=4000)

    summary = pd.DataFrame([row_phys, row_trans])
    summary_path = HERE / "t4_visium_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote: {summary_path}")
    print(summary.to_string(index=False))

    spatial_path = HERE / "t4_visium_spatial.png"
    plot_spatial(adata, kappa_phys, label, spatial_path)
    print(f"Wrote: {spatial_path}")

    dist_path = HERE / "t4_visium_distributions.png"
    plot_distributions(kappa_phys, kappa_trans, label, dist_path)
    print(f"Wrote: {dist_path}")

    # Verdict
    print("\n--- HEADLINE ---")
    sign_phys = np.sign(row_phys["diff"])
    sign_trans = np.sign(row_trans["diff"])
    sig_phys = row_phys["welch_p"] < 0.05 and abs(row_phys["cliff_d"]) >= 0.10
    sig_trans = row_trans["welch_p"] < 0.05 and abs(row_trans["cliff_d"]) >= 0.10
    print(f"  G_phys : diff={row_phys['diff']:+.4f} p={row_phys['welch_p']:.3g} "
          f"d={row_phys['cliff_d']:+.3f} -> "
          f"{'mal>str' if sign_phys>0 else 'mal<str'} "
          f"{'(SIGNIFICANT)' if sig_phys else '(weak/ns)'}")
    print(f"  G_trans: diff={row_trans['diff']:+.4f} p={row_trans['welch_p']:.3g} "
          f"d={row_trans['cliff_d']:+.3f} -> "
          f"{'mal>str' if sign_trans>0 else 'mal<str'} "
          f"{'(SIGNIFICANT)' if sig_trans else '(weak/ns)'}")
    if sig_phys and sig_trans and sign_phys == sign_trans == 1:
        verdict = "SUBSTRATE-INDEPENDENT (mal>str on both)"
    elif sig_phys and sig_trans and sign_phys == sign_trans == -1:
        verdict = "SUBSTRATE-INDEPENDENT but REVERSED (mal<str on both)"
    elif sig_trans and not sig_phys:
        verdict = "TRANSCRIPTOMIC-SIMILARITY-SPECIFIC (only G_trans)"
    elif sig_phys and not sig_trans:
        verdict = "PHYSICAL-TISSUE-SPECIFIC (only G_phys)"
    elif sig_phys and sig_trans and sign_phys != sign_trans:
        verdict = "DIVERGENT (different sign on each substrate)"
    else:
        verdict = "DOES NOT TRANSFER (neither significant)"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
