"""
T5 — Proliferating-immune control for malignant kappa elevation.

Question
--------
Is the elevated per-cell Ollivier-Ricci kappa in malignant cells a malignancy-
specific geometric signature, or merely a readout of cell-cycle / proliferation
activity? If the latter, immune cells caught in active cell-cycle (proliferating
T/B/Macro/NK) should look "malignant-like".

Design
------
Tirosh 2016 melanoma (GSE72056) provides malignant cells AND non-malignant
immune cells (T, B, Macrophage, NK) in the same tumors. We:

  1) Score each immune cell on a curated G2M/S-phase signature
     (MKI67, TOP2A, CCNB1, CCNA2, CDK1, CDC20, BIRC5, AURKA, AURKB, CENPF,
      MCM2..MCM7, PCNA). Z-score across all cells.
  2) Split immune cells by z-score: proliferating (z>1, top ~16%),
     resting (z<0).
  3) Run the standard pipeline on a 600/600/600 sample of malignant /
     proliferating-immune / resting-immune: HVG-2000, PCA-50, kNN(k=15),
     Ollivier-Ricci alpha=0.5, 4000 sampled edges, per-cell mean kappa.
  4) Three-way pairwise contrasts: Welch p, MWU p, Cliff's delta.

Verdicts
--------
  - Pure cell-cycle confound: |Cliff(mal vs prolif-imm)| < 0.10 AND
                              Cliff(prolif-imm vs resting-imm) > 0.30
    => kappa is a proliferation signal, not a malignancy signal.
  - Partial: roughly equal staircase mal > prolif > resting (each ~0.2)
    => proliferation explains some but not all of the malignant elevation.
  - No proliferation confound: Cliff(prolif vs resting) ~ 0
    => malignant elevation is genuinely tumor-specific.

Outputs (in this directory)
---------------------------
  t5_proliferating_summary.csv   three-way pairwise table
  t5_distributions.png           overlay histograms
  t5_summary.txt                 plain-text headline + verdict
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
SEED = 20260507
RNG = np.random.default_rng(SEED)


# ---------------- Ollivier-Ricci primitives (copied from path1_hyperbolic) ----
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


# ---------------- effect-size helper ----------------
def cliffs_delta(a, b):
    """Cliff's delta = P(a>b) - P(a<b). Positive => a stochastically larger."""
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


# ---------------- cell-cycle signature ----------------
CYCLE_GENES = [
    "MKI67", "TOP2A", "CCNB1", "CCNA2", "CDK1", "CDC20", "BIRC5",
    "AURKA", "AURKB", "CENPF",
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "PCNA",
]


def score_cell_cycle(X_genes_x_cells, gene_names):
    """Mean log-TPM of cell-cycle markers per cell, z-scored across all cells."""
    upper = np.array([str(g).upper() for g in gene_names])
    found_idx = []
    found_genes = []
    for g in CYCLE_GENES:
        hits = np.where(upper == g)[0]
        if len(hits) == 0:
            continue
        found_idx.append(int(hits[0]))
        found_genes.append(g)
    if len(found_idx) == 0:
        raise RuntimeError("No cell-cycle marker genes found in dataset.")
    expr = X_genes_x_cells[found_idx, :]            # markers x cells
    score = expr.mean(axis=0)                       # one value per cell
    z = (score - score.mean()) / (score.std() + 1e-12)
    return z, found_genes


# ---------------- Tirosh loader ----------------
def load_tirosh():
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("[Tirosh] loading ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    gene_names = expr.iloc[:, 0].values
    X = expr.iloc[:, 1:].values.astype(np.float32)  # genes x cells
    print(f"  matrix: {X.shape}  (genes x cells)")
    print(f"  malignant=2: {int((malignant == 2).sum())}")
    immune_mask = (malignant == 1) & np.isin(nonmal_type, [1, 2, 3, 6])
    print(f"  immune (T+B+Macro+NK): {int(immune_mask.sum())}  "
          f"(T={int(((malignant==1)&(nonmal_type==1)).sum())} "
          f"B={int(((malignant==1)&(nonmal_type==2)).sum())} "
          f"Macro={int(((malignant==1)&(nonmal_type==3)).sum())} "
          f"NK={int(((malignant==1)&(nonmal_type==6)).sum())})")
    return X, gene_names, malignant, nonmal_type


# ---------------- main ----------------
def main():
    print("=" * 70)
    print("T5 — Proliferating-immune control for malignant kappa")
    print(f"     seed={SEED}")
    print("=" * 70)

    X, gene_names, malignant, nonmal_type = load_tirosh()

    # cell-cycle z-score across ALL cells (so the threshold is comparable)
    cc_z, cc_used = score_cell_cycle(X, gene_names)
    print(f"  cell-cycle markers found ({len(cc_used)}/{len(CYCLE_GENES)}): "
          f"{cc_used}")

    mask_mal = (malignant == 2)
    mask_immune = (malignant == 1) & np.isin(nonmal_type, [1, 2, 3, 6])

    # immune cells split by cell-cycle z
    cc_imm = cc_z[mask_immune]
    n_imm = mask_immune.sum()
    n_prolif = int((cc_imm > 1.0).sum())
    n_rest = int((cc_imm < 0.0).sum())
    print(f"  immune n={n_imm}: proliferating(z>1)={n_prolif}  "
          f"resting(z<0)={n_rest}")

    # also report cell-cycle z stats for malignant (sanity context)
    cc_mal = cc_z[mask_mal]
    print(f"  malignant cell-cycle z: mean={cc_mal.mean():+.3f} "
          f"median={np.median(cc_mal):+.3f} "
          f"frac(z>1)={(cc_mal>1).mean():.3f}")
    print(f"  immune    cell-cycle z: mean={cc_imm.mean():+.3f} "
          f"median={np.median(cc_imm):+.3f} "
          f"frac(z>1)={(cc_imm>1).mean():.3f}")

    # build group masks (boolean over all cells)
    immune_idx_all = np.where(mask_immune)[0]
    prolif_idx_all = immune_idx_all[cc_imm > 1.0]
    rest_idx_all = immune_idx_all[cc_imm < 0.0]
    mal_idx_all = np.where(mask_mal)[0]

    rng = np.random.default_rng(SEED)
    n_per = 600

    def _sample(arr, n):
        if len(arr) <= n:
            return arr.copy()
        return rng.choice(arr, size=n, replace=False)

    sel_mal = _sample(mal_idx_all, n_per)
    sel_prolif = _sample(prolif_idx_all, n_per)
    sel_rest = _sample(rest_idx_all, n_per)
    print(f"  sampled: mal={len(sel_mal)}  prolif_imm={len(sel_prolif)}  "
          f"rest_imm={len(sel_rest)}")

    sel = np.concatenate([sel_mal, sel_prolif, sel_rest])
    labels = np.array(
        ["Malignant"] * len(sel_mal)
        + ["Prolif_imm"] * len(sel_prolif)
        + ["Rest_imm"] * len(sel_rest)
    )

    # HVG-2000 over the sampled cells (matches path1 convention; var across
    # cells in the sample). Then PCA-50.
    Xs_full = X[:, sel]
    var = Xs_full.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xs = Xs_full[hvg].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
    print(f"  PCA: {Xpca.shape}")

    # kNN + Ollivier-Ricci
    k = 15
    n_edges = 4000
    print(f"  building kNN(k={k}) ...")
    G = build_knn_graph(Xpca, k=k)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  computing Ollivier-Ricci on {n_edges} sampled edges ...")
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=RNG)
    n_valid = sum(1 for v in edge_k.values() if not np.isnan(v))
    print(f"  valid edges: {n_valid}/{len(edge_k)}")

    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa_all = np.array([per_cell.get(i, np.nan) for i in range(len(labels))])
    valid_mask = ~np.isnan(kappa_all)

    groups = {
        "Malignant":  kappa_all[(labels == "Malignant") & valid_mask],
        "Prolif_imm": kappa_all[(labels == "Prolif_imm") & valid_mask],
        "Rest_imm":   kappa_all[(labels == "Rest_imm") & valid_mask],
    }
    for name, vals in groups.items():
        print(f"  {name:10s}  n={len(vals):4d}  "
              f"kappa mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"median={np.median(vals):+.4f}")

    # three-way pairwise table
    pair_rows = []
    pairs = [
        ("Malignant",  "Prolif_imm"),
        ("Malignant",  "Rest_imm"),
        ("Prolif_imm", "Rest_imm"),
    ]
    for a_name, b_name in pairs:
        a = groups[a_name]
        b = groups[b_name]
        t_stat, p_t = ttest_ind(a, b, equal_var=False)
        u_stat, p_u = mannwhitneyu(a, b)
        d = cliffs_delta(a, b)
        pair_rows.append({
            "group_a": a_name,
            "group_b": b_name,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "diff_a_minus_b": float(a.mean() - b.mean()),
            "welch_t": float(t_stat),
            "welch_p": float(p_t),
            "mwu_u": float(u_stat),
            "mwu_p": float(p_u),
            "cliff_delta": float(d),
        })
        print(f"  {a_name} vs {b_name}: "
              f"diff={a.mean()-b.mean():+.4f}  "
              f"Welch p={p_t:.3g}  MWU p={p_u:.3g}  Cliff d={d:+.3f}")

    pair_df = pd.DataFrame(pair_rows)
    pair_path = HERE / "t5_proliferating_summary.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"\nWrote: {pair_path}")

    # plot overlay histograms
    fig, ax = plt.subplots(figsize=(8, 5))
    color = {"Malignant": "C3", "Prolif_imm": "C1", "Rest_imm": "C0"}
    for name, vals in groups.items():
        ax.hist(vals, bins=40, alpha=0.5, density=True, color=color[name],
                label=f"{name} (n={len(vals)}, "
                      f"mean={vals.mean():+.4f})")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("per-cell mean Ollivier-Ricci kappa")
    ax.set_ylabel("density")
    ax.set_title("T5 — kappa across malignant, proliferating immune, "
                 "resting immune\n"
                 f"Cliff(mal vs prolif)={pair_rows[0]['cliff_delta']:+.3f}  "
                 f"Cliff(mal vs rest)={pair_rows[1]['cliff_delta']:+.3f}  "
                 f"Cliff(prolif vs rest)={pair_rows[2]['cliff_delta']:+.3f}")
    ax.legend()
    plt.tight_layout()
    png_path = HERE / "t5_distributions.png"
    plt.savefig(png_path, dpi=130)
    plt.close()
    print(f"Wrote: {png_path}")

    # verdict
    d_mal_prolif = pair_rows[0]["cliff_delta"]
    d_mal_rest = pair_rows[1]["cliff_delta"]
    d_prolif_rest = pair_rows[2]["cliff_delta"]

    if abs(d_mal_prolif) < 0.10 and d_prolif_rest > 0.30:
        verdict = "PURE_CELL_CYCLE_CONFOUND"
        headline = ("kappa tracks cell-cycle activity: malignant cells are "
                    "indistinguishable from proliferating immune cells, and "
                    "proliferating immune cells differ strongly from resting "
                    "ones.")
    elif abs(d_prolif_rest) < 0.10:
        verdict = "NO_PROLIFERATION_CONFOUND"
        headline = ("Proliferation does not move kappa: proliferating and "
                    "resting immune cells are indistinguishable. The "
                    "malignant elevation is tumor-specific.")
    elif d_mal_prolif > 0.10 and d_prolif_rest > 0.10:
        verdict = "PARTIAL_PROLIFERATION_CONTRIBUTION"
        headline = ("Cell-cycle accounts for part of the malignant kappa "
                    "elevation, but a residual malignant-vs-proliferating "
                    "gap remains.")
    else:
        verdict = "AMBIGUOUS"
        headline = ("Pattern does not match any pre-registered verdict cell.")

    summary_lines = [
        "=" * 70,
        "T5 — Proliferating-immune control for malignant kappa",
        f"seed={SEED}",
        "=" * 70,
        "",
        f"Cell-cycle markers used: {len(cc_used)}/{len(CYCLE_GENES)} "
        f"({', '.join(cc_used)})",
        "",
        f"Per-group n and mean kappa:",
        f"  Malignant     n={len(groups['Malignant']):4d}  "
        f"kappa = {groups['Malignant'].mean():+.4f}  "
        f"std={groups['Malignant'].std():.4f}",
        f"  Prolif_imm    n={len(groups['Prolif_imm']):4d}  "
        f"kappa = {groups['Prolif_imm'].mean():+.4f}  "
        f"std={groups['Prolif_imm'].std():.4f}",
        f"  Rest_imm      n={len(groups['Rest_imm']):4d}  "
        f"kappa = {groups['Rest_imm'].mean():+.4f}  "
        f"std={groups['Rest_imm'].std():.4f}",
        "",
        "Pairwise contrasts (Cliff's delta with first group as reference):",
        f"  Mal vs Prolif: diff={pair_rows[0]['diff_a_minus_b']:+.4f}  "
        f"Welch p={pair_rows[0]['welch_p']:.3g}  "
        f"MWU p={pair_rows[0]['mwu_p']:.3g}  "
        f"Cliff d={d_mal_prolif:+.3f}",
        f"  Mal vs Rest:   diff={pair_rows[1]['diff_a_minus_b']:+.4f}  "
        f"Welch p={pair_rows[1]['welch_p']:.3g}  "
        f"MWU p={pair_rows[1]['mwu_p']:.3g}  "
        f"Cliff d={d_mal_rest:+.3f}",
        f"  Prolif vs Rest: diff={pair_rows[2]['diff_a_minus_b']:+.4f}  "
        f"Welch p={pair_rows[2]['welch_p']:.3g}  "
        f"MWU p={pair_rows[2]['mwu_p']:.3g}  "
        f"Cliff d={d_prolif_rest:+.3f}",
        "",
        f"VERDICT: {verdict}",
        f"  {headline}",
    ]
    txt_path = HERE / "t5_summary.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote: {txt_path}")

    print("\n--- HEADLINE ---")
    print(f"  VERDICT: {verdict}")
    print(f"  {headline}")


if __name__ == "__main__":
    main()
