"""
N3 — Per-bin Ollivier-Ricci kappa on 3D-genome contact graphs.

Tests whether the per-cell OR-kappa direction observed in transcriptomic kNN
graphs (cancer > normal) extends to chromatin contact graphs.

Pipeline:
  1) Stream chr22 intra-chromosomal contact records at 1 Mb resolution via
     hic-straw from Rao 2014 / GSE63525:
       K562   (chronic myeloid leukemia, female)   = cancer
       GM12878 (lymphoblastoid, female)            = normal
     Use observed/KR-balanced counts; KR-balanced collapses systematic
     coverage bias so cross-cell-line comparison is meaningful.
  2) Build a per-cell-line contact graph:
       nodes = 1-Mb bins on chr22
       remove the "self" diagonal (genomic adjacency baseline removed by
       distance-decay correction below)
       distance-decay normalize: observed/expected via per-genomic-distance
         median, so we keep only contacts above the polymer baseline.
       keep top-K edges by O/E (K = 3*N_nodes) to obtain a sparse,
         cancer-vs-normal-comparable graph.
       edge weight = 1 / log(1 + observed_counts).
  3) Compute per-edge Ollivier-Ricci kappa (alpha=0.5, exact W1) and
     aggregate to per-bin mean kappa.
  4) Compare K562 vs GM12878 per-bin kappa with Mann-Whitney U + Cliff's
     delta. Plot histograms + log-contact maps.

Outputs (in this directory):
  n3_hic_kappa.csv
  n3_hic_distributions.png
  n3_hic_contact_maps.png
  n3_summary.txt
  run.log
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).parent
LOG_FH = open(HERE / "run.log", "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG_FH.write(msg + "\n")
    LOG_FH.flush()


SEED = 20260508
RNG = np.random.default_rng(SEED)
RES_BP = 1_000_000
CHROM = "22"

URLS = {
    "K562": "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/HIC069.hic",
    "GM12878": "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
}


# ---------------- Hi-C fetch ----------------
def fetch_chr_matrix(url: str, chrom: str, res: int):
    """Stream KR-balanced observed counts for chrom-vs-chrom from a remote .hic."""
    import hicstraw

    hic = hicstraw.HiCFile(url)
    chroms = {c.name: c.length for c in hic.getChromosomes()}
    if chrom not in chroms:
        # try with "chr" prefix variation
        alt = "chr" + chrom if not chrom.startswith("chr") else chrom.lstrip("chr")
        if alt in chroms:
            chrom = alt
        else:
            raise KeyError(f"chrom {chrom} not in {list(chroms)[:5]}...")
    nbp = chroms[chrom]
    mzd = hic.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", res)
    records = mzd.getRecords(0, nbp, 0, nbp)
    n_bins = int(np.ceil(nbp / res))
    M = np.zeros((n_bins, n_bins), dtype=np.float64)
    for r in records:
        i = r.binX // res
        j = r.binY // res
        c = r.counts
        if not np.isfinite(c):
            continue
        M[i, j] = c
        M[j, i] = c
    return M, n_bins, nbp


# ---------------- Graph construction ----------------
def contact_to_graph(M: np.ndarray, top_k_factor: int = 3, min_sep: int = 2):
    """
    Build a graph from a Hi-C contact matrix.

    Steps:
      1) Drop intra-bin (diagonal) and adjacent bins (|i-j| < min_sep) to
         remove the trivial polymer-distance signal that overwhelms everything.
      2) Compute per-genomic-distance expected as the median over off-diag
         band, then O/E ratio.
      3) Mask bins that are entirely empty (centromere, etc.).
      4) Keep top-K edges by O/E (K = top_k_factor * n_bins) -> sparse graph
         with comparable density across cell lines.
      5) Edge weight = 1 / log(1 + observed_counts) so high-contact = short
         distance, like the protocol asks.
    """
    n = M.shape[0]
    coverage = M.sum(axis=1)
    valid = coverage > 0
    valid_idx = np.where(valid)[0]

    # Per-distance expected (median of nonzero)
    expected = np.zeros(n)
    for d in range(n):
        vals = []
        for i in range(n - d):
            j = i + d
            if valid[i] and valid[j] and M[i, j] > 0:
                vals.append(M[i, j])
        if vals:
            expected[d] = float(np.median(vals))
        else:
            expected[d] = np.nan

    # Build candidate edges (i<j, |i-j|>=min_sep, O>0)
    cand = []
    for i in valid_idx:
        for j in valid_idx:
            if j <= i:
                continue
            d = j - i
            if d < min_sep:
                continue
            o = M[i, j]
            e = expected[d]
            if not np.isfinite(e) or e <= 0 or o <= 0:
                continue
            cand.append((float(o / e), i, j, o))
    cand.sort(reverse=True, key=lambda t: t[0])

    K = top_k_factor * len(valid_idx)
    keep = cand[:K]

    G = nx.Graph()
    G.add_nodes_from(int(i) for i in valid_idx)
    for oe, i, j, o in keep:
        w = 1.0 / np.log1p(o)
        if w <= 0 or not np.isfinite(w):
            continue
        G.add_edge(int(i), int(j), weight=float(w), oe=float(oe), obs=float(o))
    return G, valid_idx, expected


# ---------------- Ollivier-Ricci primitives ----------------
def ollivier_ricci_edges(G: nx.Graph, alpha: float = 0.5):
    """Per-edge OR kappa, exact W1 via POT. Adapted from E1."""
    out = {}
    for (u, v) in list(G.edges()):
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
            dists = {nd: nx.single_source_dijkstra_path_length(sub, nd)
                     for nd in sup_u}
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


def per_node_mean_curv(G, edge_kappa):
    by = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if not np.isfinite(k):
            continue
        by[u].append(k)
        by[v].append(k)
    return {n: float(np.mean(v)) if v else np.nan for n, v in by.items()}


# ---------------- Effect size ----------------
def cliffs_delta(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    B = np.sort(b)
    n_less = np.searchsorted(B, a, side="left")
    n_leq = np.searchsorted(B, a, side="right")
    n_greater = len(B) - n_leq
    gt = n_less.sum()
    lt = n_greater.sum()
    return float((gt - lt) / (len(a) * len(b)))


# ---------------- Main ----------------
def main():
    log("=" * 72)
    log("N3 — Hi-C OR-kappa: K562 (cancer) vs GM12878 (normal) on chr22 @ 1 Mb")
    log("=" * 72)
    log(f"seed={SEED}  res={RES_BP}  chrom={CHROM}")

    matrices = {}
    for tag, url in URLS.items():
        t0 = time.time()
        try:
            M, n_bins, nbp = fetch_chr_matrix(url, CHROM, RES_BP)
        except Exception as e:
            log(f"[{tag}] DOWNLOAD FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            (HERE / "n3_summary.txt").write_text(
                "SKIPPED: Hi-C download failed.\n"
                f"  cell line: {tag}\n"
                f"  url: {url}\n"
                f"  error: {type(e).__name__}: {e}\n"
            )
            return
        nz = int((M > 0).sum())
        log(f"[{tag}] fetched  bins={n_bins}  nz_entries={nz}  "
            f"sum={M.sum():.3g}  ({time.time()-t0:.1f}s)")
        matrices[tag] = (M, n_bins, nbp)

    # Sanity: matrices should have same shape
    n_bins = matrices["K562"][1]
    assert matrices["GM12878"][1] == n_bins, "bin-count mismatch"

    # ----- contact-map plot -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, tag in zip(axes, ["K562", "GM12878"]):
        M = matrices[tag][0]
        im = ax.imshow(np.log10(M + 1.0), cmap="Reds", aspect="equal")
        ax.set_title(f"{tag} chr22 @ {RES_BP//1_000_000}Mb (log10 KR-counts+1)")
        ax.set_xlabel("bin")
        ax.set_ylabel("bin")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(HERE / "n3_hic_contact_maps.png", dpi=130)
    plt.close(fig)
    log("wrote n3_hic_contact_maps.png")

    # ----- graph build + curvature -----
    per_bin_kappa = {}
    graph_stats = {}
    for tag in ["K562", "GM12878"]:
        M = matrices[tag][0]
        G, valid_idx, expected = contact_to_graph(M, top_k_factor=3, min_sep=2)
        log(f"[{tag}] graph: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
            f"valid_bins={len(valid_idx)}")
        # global stats
        try:
            cc = nx.average_clustering(G, weight=None)
        except Exception:
            cc = float("nan")
        deg = np.array([d for _, d in G.degree()], float)
        graph_stats[tag] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "mean_degree": float(deg.mean()) if len(deg) else float("nan"),
            "clustering": float(cc),
            "ncc": int(nx.number_connected_components(G)),
        }

        # OR-kappa
        t0 = time.time()
        edge_k = ollivier_ricci_edges(G, alpha=0.5)
        log(f"[{tag}] OR-kappa on {len(edge_k)} edges in {time.time()-t0:.1f}s")
        per = per_node_mean_curv(G, edge_k)
        per_bin_kappa[tag] = per

    # ----- per-bin kappa CSV -----
    rows = []
    all_bins = sorted(set(per_bin_kappa["K562"]).union(per_bin_kappa["GM12878"]))
    for b in all_bins:
        rows.append({
            "bin": int(b),
            "start_bp": int(b) * RES_BP,
            "end_bp": (int(b) + 1) * RES_BP,
            "kappa_K562": per_bin_kappa["K562"].get(b, np.nan),
            "kappa_GM12878": per_bin_kappa["GM12878"].get(b, np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "n3_hic_kappa.csv", index=False)
    log(f"wrote n3_hic_kappa.csv  ({len(df)} bins)")

    # ----- statistical comparison -----
    a = df["kappa_K562"].values
    b = df["kappa_GM12878"].values
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    log(f"finite kappas:  K562={len(a)}  GM12878={len(b)}")

    if len(a) >= 3 and len(b) >= 3:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        delta = cliffs_delta(a, b)
        mean_k, mean_g = float(np.mean(a)), float(np.mean(b))
        med_k, med_g = float(np.median(a)), float(np.median(b))
        log(f"K562    mean kappa = {mean_k:+.4f}   median = {med_k:+.4f}   n = {len(a)}")
        log(f"GM12878 mean kappa = {mean_g:+.4f}   median = {med_g:+.4f}   n = {len(b)}")
        log(f"Mann-Whitney U = {u:.1f}  p = {p:.4g}")
        log(f"Cliff's delta (K562 vs GM12878) = {delta:+.4f}")
        if mean_k > mean_g:
            interp = ("K562 (cancer) > GM12878 (normal): cancer chromatin shows "
                      "MORE positive curvature, consistent with the scRNA-seq "
                      "cancer>normal direction. Extends OR-kappa cancer signature "
                      "from transcriptomic to 3D-genome topology.")
        else:
            interp = ("K562 (cancer) < GM12878 (normal): cancer chromatin shows "
                      "MORE NEGATIVE curvature, opposite to the scRNA-seq "
                      "direction but consistent with reported TAD-boundary loss "
                      "/ compartment switching in cancer (Flavahan 2016, "
                      "Hnisz 2016).")
        log("Interpretation: " + interp)
    else:
        u = p = delta = mean_k = mean_g = med_k = med_g = float("nan")
        interp = "Too few finite kappa values for statistics."
        log(interp)

    # ----- distribution plot -----
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    bins = np.linspace(min(a.min(), b.min()) - 0.05,
                       max(a.max(), b.max()) + 0.05, 18)
    ax.hist(b, bins=bins, alpha=0.55, color="steelblue", label=f"GM12878 (n={len(b)})", edgecolor="k")
    ax.hist(a, bins=bins, alpha=0.55, color="firebrick", label=f"K562 (n={len(a)})", edgecolor="k")
    ax.axvline(mean_g, color="steelblue", lw=2, ls="--")
    ax.axvline(mean_k, color="firebrick", lw=2, ls="--")
    ax.set_xlabel("per-bin Ollivier-Ricci kappa  (chr22 1 Mb)")
    ax.set_ylabel("# bins")
    ax.set_title(f"Hi-C OR-kappa  K562 vs GM12878   MWU p={p:.3g}  delta={delta:+.3f}")
    ax.legend()
    fig.savefig(HERE / "n3_hic_distributions.png", dpi=130)
    plt.close(fig)
    log("wrote n3_hic_distributions.png")

    # ----- summary text -----
    summary_lines = [
        "N3 — Hi-C OR-kappa: K562 (cancer) vs GM12878 (normal)",
        "=" * 60,
        f"chrom = chr{CHROM}    resolution = {RES_BP} bp    seed = {SEED}",
        "norm = KR-balanced; edges = top-3*N by O/E with |i-j| >= 2",
        "",
        f"  K562    bins (finite kappa) = {len(a)}",
        f"          mean kappa   = {mean_k:+.4f}",
        f"          median kappa = {med_k:+.4f}",
        f"  GM12878 bins (finite kappa) = {len(b)}",
        f"          mean kappa   = {mean_g:+.4f}",
        f"          median kappa = {med_g:+.4f}",
        "",
        f"Mann-Whitney U = {u:.2f}    p = {p:.4g}",
        f"Cliff's delta (K562 vs GM12878) = {delta:+.4f}",
        "",
        "Graph stats:",
    ]
    for tag, gs in graph_stats.items():
        summary_lines.append(
            f"  {tag}: V={gs['nodes']} E={gs['edges']} "
            f"<deg>={gs['mean_degree']:.2f} "
            f"clust={gs['clustering']:.3f} ncc={gs['ncc']}"
        )
    summary_lines += ["", "Interpretation:", "  " + interp]
    (HERE / "n3_summary.txt").write_text("\n".join(summary_lines) + "\n")
    log("wrote n3_summary.txt")
    log("done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        LOG_FH.close()
