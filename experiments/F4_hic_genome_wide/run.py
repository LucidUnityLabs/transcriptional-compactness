"""
F4 — Genome-wide per-bin Ollivier-Ricci kappa on Hi-C contact graphs at 100 kb.

Settles whether the chr22-vs-chr11 disagreement seen in N3 (1 Mb single-chrom)
reflects chromosome-specific noise or a real chromatin-vs-genome curvature
direction. Pulls all autosomes (chr1-22) + chrX from the same Rao 2014
GSE63525 in-situ Hi-C used by N3, but at 100 kb resolution.

Pipeline (per chromosome, both cell lines):
  1) Stream KR-balanced observed contacts via hic-straw at 100 kb.
  2) Distance-decay normalize: per-genomic-distance median expected,
     compute O/E.
  3) Edges = top 3*N pairs by O/E (with O/E > 1.5 floor), |i-j| >= 2.
     Edge weight = 1 / log(1 + observed_count).
  4) Per-edge OR-kappa (alpha=0.5 lazy walk), exact W1 via POT, sampled
     to <=4000 edges if the graph is larger.
  5) Aggregate edge-kappa to per-bin mean.
  6) Per-chromosome MWU + Cliff's delta on K562 vs GM12878 per-bin kappa.
  7) Pool across chromosomes for a single genome-wide comparison.

Outputs:
  f4_hic_per_chromosome.csv
  f4_hic_genome_wide_summary.csv
  f4_hic_per_chromosome.png
  f4_hic_distributions.png
  f4_summary.txt
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
    print(msg, flush=True)
    LOG_FH.write(msg + "\n")
    LOG_FH.flush()


SEED = 20260508
RNG = np.random.default_rng(SEED)
RES_BP = 100_000
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
MAX_EDGES_FOR_OR = 4000
TOP_K_FACTOR = 3
MIN_SEP = 2
OE_FLOOR = 1.5

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
        alt = "chr" + chrom if not chrom.startswith("chr") else chrom.lstrip("chr")
        if alt in chroms:
            chrom = alt
        else:
            raise KeyError(f"chrom {chrom} not in {list(chroms)[:5]}...")
    nbp = chroms[chrom]
    mzd = hic.getMatrixZoomData(chrom, chrom, "observed", "KR", "BP", res)
    records = mzd.getRecords(0, nbp, 0, nbp)
    n_bins = int(np.ceil(nbp / res))

    # Build sparse upper-tri arrays then materialize a dense matrix.
    # At 100 kb, max chrom (chr1) is ~2480 bins -> ~6 GB float64 dense, too big.
    # Use float32 -> ~24 MB worst-case for chr1, fine.
    M = np.zeros((n_bins, n_bins), dtype=np.float32)
    for r in records:
        i = r.binX // res
        j = r.binY // res
        c = r.counts
        if not np.isfinite(c) or c <= 0:
            continue
        if i >= n_bins or j >= n_bins:
            continue
        M[i, j] = c
        if i != j:
            M[j, i] = c
    return M, n_bins, nbp


# ---------------- Graph construction (vectorized) ----------------
def contact_to_graph(M: np.ndarray,
                     top_k_factor: int = TOP_K_FACTOR,
                     min_sep: int = MIN_SEP,
                     oe_floor: float = OE_FLOOR):
    """
    Vectorized version of N3 graph build, designed for genome-wide use.

    Returns:
      G: nx.Graph
      valid_idx: bins with positive coverage
      expected: per-distance median (size n)
    """
    n = M.shape[0]
    # row sum to find empty bins (centromeres, telomeres)
    coverage = np.asarray(M.sum(axis=1)).ravel()
    valid = coverage > 0
    valid_idx = np.where(valid)[0]
    n_valid = len(valid_idx)
    if n_valid == 0:
        return nx.Graph(), valid_idx, np.full(n, np.nan)

    # Vectorized per-distance median over upper tri using diagonals.
    # Skip d < min_sep (handled in edge selection).
    expected = np.full(n, np.nan, dtype=np.float64)
    for d in range(min_sep, n):
        diag = M.diagonal(offset=d)
        # mask: bins on either end must be valid AND nonzero
        i_arr = np.arange(n - d)
        j_arr = i_arr + d
        ok = valid[i_arr] & valid[j_arr] & (diag > 0)
        if ok.any():
            expected[d] = float(np.median(diag[ok]))

    # Build candidate edges from the upper-tri of M restricted to valid bins.
    rows, cols = np.where(np.triu(M, k=min_sep) > 0)
    # restrict to valid
    valid_mask = valid[rows] & valid[cols]
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    if rows.size == 0:
        G = nx.Graph()
        G.add_nodes_from(int(i) for i in valid_idx)
        return G, valid_idx, expected

    obs = M[rows, cols].astype(np.float64)
    d_arr = cols - rows
    exp = expected[d_arr]
    valid2 = np.isfinite(exp) & (exp > 0)
    rows = rows[valid2]
    cols = cols[valid2]
    obs = obs[valid2]
    d_arr = d_arr[valid2]
    exp = exp[valid2]

    oe = obs / exp
    # apply O/E floor
    keep = oe >= oe_floor
    rows = rows[keep]; cols = cols[keep]; obs = obs[keep]; oe = oe[keep]

    # then top K = top_k_factor * n_valid by OE among the floor-passing edges
    K = top_k_factor * n_valid
    if rows.size > K:
        idx = np.argpartition(-oe, K - 1)[:K]
        rows = rows[idx]; cols = cols[idx]; obs = obs[idx]; oe = oe[idx]

    G = nx.Graph()
    G.add_nodes_from(int(i) for i in valid_idx)
    weights = 1.0 / np.log1p(obs)
    finite = np.isfinite(weights) & (weights > 0)
    rows = rows[finite]; cols = cols[finite]; obs = obs[finite]
    oe = oe[finite]; weights = weights[finite]
    for i, j, w, o, r in zip(rows.tolist(), cols.tolist(),
                             weights.tolist(), obs.tolist(), oe.tolist()):
        G.add_edge(int(i), int(j), weight=float(w),
                   oe=float(r), obs=float(o))
    return G, valid_idx, expected


# ---------------- Ollivier-Ricci primitives ----------------
def ollivier_ricci_edges(G: nx.Graph, alpha: float = 0.5,
                          max_edges: int = MAX_EDGES_FOR_OR,
                          rng=None):
    """
    Per-edge OR kappa, exact W1 via POT.

    If |E| > max_edges, sample max_edges edges uniformly without replacement.
    """
    edges = list(G.edges())
    if rng is None:
        rng = np.random.default_rng(SEED)
    if len(edges) > max_edges:
        sel = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in sel]

    out = {}
    for (u, v) in edges:
        nu = list(G.neighbors(u))
        nv = list(G.neighbors(v))
        sup_u = [u] + nu
        sup_v = [v] + nv
        if nu:
            w_u = np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu))
        else:
            w_u = np.array([1.0])
        if nv:
            w_v = np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv))
        else:
            w_v = np.array([1.0])
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


def cliffs_delta_ci_bootstrap(a, b, n_boot=200, alpha=0.05, rng=None):
    """Percentile bootstrap CI for Cliff's delta."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(SEED)
    point = cliffs_delta(a, b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.integers(0, len(a), size=len(a))
        bi = rng.integers(0, len(b), size=len(b))
        deltas[i] = cliffs_delta(a[ai], b[bi])
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return point, np.nan, np.nan
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return point, lo, hi


# ---------------- Per-chromosome processing ----------------
def process_chromosome(chrom: str, matrices: dict, rng):
    """
    matrices[tag] = (M, n_bins, nbp).
    Returns (per_chrom_row_dict, per_bin_records_for_pool).
    """
    out_kappas = {}
    graph_stats = {}
    for tag in ["K562", "GM12878"]:
        M = matrices[tag][0]
        G, valid_idx, expected = contact_to_graph(
            M, top_k_factor=TOP_K_FACTOR, min_sep=MIN_SEP, oe_floor=OE_FLOOR
        )
        log(f"  [{tag}] |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
            f"valid_bins={len(valid_idx)}")

        if G.number_of_edges() == 0:
            out_kappas[tag] = {}
            graph_stats[tag] = {"nodes": G.number_of_nodes(), "edges": 0}
            continue

        t0 = time.time()
        edge_k = ollivier_ricci_edges(G, alpha=0.5,
                                       max_edges=MAX_EDGES_FOR_OR, rng=rng)
        log(f"  [{tag}] OR-kappa on {len(edge_k)} edges in "
            f"{time.time()-t0:.1f}s")
        per = per_node_mean_curv(G, edge_k)
        out_kappas[tag] = per
        graph_stats[tag] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }

    # Aggregate this chromosome's stats
    a_vals = np.array([v for v in out_kappas.get("K562", {}).values()
                       if np.isfinite(v)], float)
    b_vals = np.array([v for v in out_kappas.get("GM12878", {}).values()
                       if np.isfinite(v)], float)
    if len(a_vals) >= 3 and len(b_vals) >= 3:
        u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        delta, lo, hi = cliffs_delta_ci_bootstrap(a_vals, b_vals,
                                                   n_boot=200, rng=rng)
    else:
        u_stat = p_val = delta = lo = hi = np.nan

    row = {
        "chrom": f"chr{chrom}",
        "n_bins_K562": len(a_vals),
        "n_bins_GM12878": len(b_vals),
        "K562_kappa_mean": float(np.mean(a_vals)) if len(a_vals) else np.nan,
        "K562_kappa_median": float(np.median(a_vals)) if len(a_vals) else np.nan,
        "GM12878_kappa_mean": float(np.mean(b_vals)) if len(b_vals) else np.nan,
        "GM12878_kappa_median": float(np.median(b_vals)) if len(b_vals) else np.nan,
        "mwu_p": float(p_val) if np.isfinite(p_val) else np.nan,
        "cliff_delta": float(delta) if np.isfinite(delta) else np.nan,
        "cliff_delta_lo95": lo,
        "cliff_delta_hi95": hi,
        "K562_edges": graph_stats.get("K562", {}).get("edges", 0),
        "GM12878_edges": graph_stats.get("GM12878", {}).get("edges", 0),
    }
    return row, out_kappas


# ---------------- Main ----------------
def main():
    log("=" * 78)
    log("F4 — Genome-wide Hi-C OR-kappa: K562 (cancer) vs GM12878 (normal) @ 100 kb")
    log("=" * 78)
    log(f"seed={SEED}  res={RES_BP}  chroms={CHROMS}")
    log(f"top_k_factor={TOP_K_FACTOR}  min_sep={MIN_SEP}  oe_floor={OE_FLOOR}")
    log(f"max_edges_for_or={MAX_EDGES_FOR_OR}")

    rng = np.random.default_rng(SEED)
    per_chrom_rows = []
    pool_K = []
    pool_G = []
    pool_chr = []  # chrom label per bin for diagnostics

    for chrom in CHROMS:
        log("-" * 70)
        log(f"chrom={chrom}")
        matrices = {}
        ok = True
        for tag, url in URLS.items():
            t0 = time.time()
            try:
                M, n_bins, nbp = fetch_chr_matrix(url, chrom, RES_BP)
            except Exception as e:
                log(f"  [{tag}] DOWNLOAD FAILED chrom={chrom}: "
                    f"{type(e).__name__}: {e}")
                # If it's the first chrom failing for both, abort entirely.
                ok = False
                break
            nz = int((M > 0).sum())
            log(f"  [{tag}] fetched bins={n_bins} nz={nz} "
                f"sum={M.sum():.3g} ({time.time()-t0:.1f}s)")
            matrices[tag] = (M, n_bins, nbp)

        if not ok:
            log(f"  skipping chrom={chrom}")
            continue
        if matrices["K562"][1] != matrices["GM12878"][1]:
            log(f"  bin-count mismatch K562={matrices['K562'][1]} "
                f"GM12878={matrices['GM12878'][1]}, skip chrom")
            continue

        row, kappas = process_chromosome(chrom, matrices, rng)
        per_chrom_rows.append(row)
        log(f"  -> n_K={row['n_bins_K562']}  n_G={row['n_bins_GM12878']}  "
            f"meanK={row['K562_kappa_mean']:+.4f}  "
            f"meanG={row['GM12878_kappa_mean']:+.4f}  "
            f"MWUp={row['mwu_p']:.4g}  delta={row['cliff_delta']:+.4f}  "
            f"95%CI=[{row['cliff_delta_lo95']:+.3f},"
            f"{row['cliff_delta_hi95']:+.3f}]")

        # pool
        for v in kappas.get("K562", {}).values():
            if np.isfinite(v):
                pool_K.append(v); pool_chr.append(chrom)
        for v in kappas.get("GM12878", {}).values():
            if np.isfinite(v):
                pool_G.append(v)

        # Free matrices early
        del matrices

    if not per_chrom_rows:
        msg = "SKIPPED: no chromosomes successfully processed (Hi-C fetch failed)."
        log(msg)
        (HERE / "f4_summary.txt").write_text(msg + "\n")
        return

    df_chrom = pd.DataFrame(per_chrom_rows)
    df_chrom.to_csv(HERE / "f4_hic_per_chromosome.csv", index=False)
    log(f"wrote f4_hic_per_chromosome.csv  ({len(df_chrom)} chroms)")

    # ----- pooled genome-wide stats -----
    a = np.array(pool_K, float)
    b = np.array(pool_G, float)
    if len(a) >= 3 and len(b) >= 3:
        u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
        delta, lo, hi = cliffs_delta_ci_bootstrap(a, b, n_boot=400, rng=rng)
    else:
        u_stat = p_val = delta = lo = hi = np.nan

    pooled = {
        "n_chroms": len(df_chrom),
        "n_bins_K562_pooled": len(a),
        "n_bins_GM12878_pooled": len(b),
        "K562_kappa_mean": float(np.mean(a)) if len(a) else np.nan,
        "K562_kappa_median": float(np.median(a)) if len(a) else np.nan,
        "GM12878_kappa_mean": float(np.mean(b)) if len(b) else np.nan,
        "GM12878_kappa_median": float(np.median(b)) if len(b) else np.nan,
        "mwu_U": float(u_stat) if np.isfinite(u_stat) else np.nan,
        "mwu_p": float(p_val) if np.isfinite(p_val) else np.nan,
        "cliff_delta": float(delta) if np.isfinite(delta) else np.nan,
        "cliff_delta_lo95": lo,
        "cliff_delta_hi95": hi,
    }
    pd.DataFrame([pooled]).to_csv(
        HERE / "f4_hic_genome_wide_summary.csv", index=False
    )
    log(f"wrote f4_hic_genome_wide_summary.csv")
    log(f"POOLED:  K562 mean={pooled['K562_kappa_mean']:+.4f}  "
        f"GM12878 mean={pooled['GM12878_kappa_mean']:+.4f}  "
        f"MWUp={pooled['mwu_p']:.4g}  delta={pooled['cliff_delta']:+.4f}  "
        f"95%CI=[{lo:+.3f},{hi:+.3f}]")

    # Direction agreement count
    same_dir_neg = int((df_chrom["cliff_delta"] <= -0.05).sum())
    same_dir_pos = int((df_chrom["cliff_delta"] >= +0.05).sum())
    near_zero = int(df_chrom["cliff_delta"].abs().lt(0.05).sum())
    log(f"per-chrom: cancer<normal (delta<=-0.05) = {same_dir_neg}/{len(df_chrom)}  "
        f"cancer>normal (delta>=+0.05) = {same_dir_pos}/{len(df_chrom)}  "
        f"near-zero = {near_zero}/{len(df_chrom)}")

    # ----- per-chromosome bar plot -----
    df_plot = df_chrom.copy()
    df_plot["chrom_order"] = df_plot["chrom"].apply(
        lambda s: 23 if s == "chrX" else int(s.replace("chr", ""))
    )
    df_plot = df_plot.sort_values("chrom_order").reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(df_plot))
    deltas = df_plot["cliff_delta"].values
    lo_arr = df_plot["cliff_delta_lo95"].values
    hi_arr = df_plot["cliff_delta_hi95"].values
    err = np.vstack([deltas - lo_arr, hi_arr - deltas])
    colors = ["firebrick" if d <= 0 else "steelblue" for d in deltas]
    ax.bar(x, deltas, yerr=err, color=colors, edgecolor="k",
           error_kw=dict(lw=1.0, capsize=3))
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(pooled["cliff_delta"], color="orange", lw=1.5, ls="--",
               label=f"genome-wide delta = {pooled['cliff_delta']:+.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["chrom"], rotation=70, fontsize=8)
    ax.set_ylabel("Cliff's delta (K562 vs GM12878)")
    ax.set_title("Per-chromosome Hi-C OR-kappa (K562 cancer vs GM12878 normal) @ 100 kb")
    ax.legend(loc="best")
    fig.savefig(HERE / "f4_hic_per_chromosome.png", dpi=130)
    plt.close(fig)
    log("wrote f4_hic_per_chromosome.png")

    # ----- pooled distribution -----
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), constrained_layout=True)
    if len(a) and len(b):
        bins = np.linspace(min(a.min(), b.min()) - 0.02,
                           max(a.max(), b.max()) + 0.02, 40)
        ax.hist(b, bins=bins, alpha=0.55, color="steelblue",
                label=f"GM12878 (n={len(b)})", edgecolor="k")
        ax.hist(a, bins=bins, alpha=0.55, color="firebrick",
                label=f"K562 (n={len(a)})", edgecolor="k")
        ax.axvline(pooled["GM12878_kappa_mean"], color="steelblue",
                   lw=2, ls="--")
        ax.axvline(pooled["K562_kappa_mean"], color="firebrick",
                   lw=2, ls="--")
    ax.set_xlabel("per-bin Ollivier-Ricci kappa (genome-wide @ 100 kb)")
    ax.set_ylabel("# bins (pooled)")
    ax.set_title(
        f"Pooled Hi-C OR-kappa K562 vs GM12878  "
        f"MWUp={pooled['mwu_p']:.3g}  delta={pooled['cliff_delta']:+.3f}"
    )
    ax.legend()
    fig.savefig(HERE / "f4_hic_distributions.png", dpi=130)
    plt.close(fig)
    log("wrote f4_hic_distributions.png")

    # ----- summary text -----
    lines = [
        "F4 — Genome-wide Hi-C OR-kappa K562 vs GM12878 @ 100 kb",
        "=" * 70,
        f"resolution = {RES_BP} bp     seed = {SEED}",
        f"chromosomes processed = {len(df_chrom)} / {len(CHROMS)}",
        f"edges per chrom = top {TOP_K_FACTOR}*N by O/E with O/E >= {OE_FLOOR}, "
        f"|i-j| >= {MIN_SEP}",
        f"OR-kappa: alpha=0.5; sample <={MAX_EDGES_FOR_OR} edges if larger",
        "",
        "POOLED GENOME-WIDE:",
        f"  K562    n_bins = {pooled['n_bins_K562_pooled']}   "
        f"mean_kappa = {pooled['K562_kappa_mean']:+.4f}   "
        f"median = {pooled['K562_kappa_median']:+.4f}",
        f"  GM12878 n_bins = {pooled['n_bins_GM12878_pooled']}   "
        f"mean_kappa = {pooled['GM12878_kappa_mean']:+.4f}   "
        f"median = {pooled['GM12878_kappa_median']:+.4f}",
        f"  Mann-Whitney U = {pooled['mwu_U']}   p = {pooled['mwu_p']:.4g}",
        f"  Cliff's delta = {pooled['cliff_delta']:+.4f}   "
        f"95% CI = [{pooled['cliff_delta_lo95']:+.3f}, "
        f"{pooled['cliff_delta_hi95']:+.3f}]",
        "",
        f"per-chrom direction:",
        f"  cancer<normal (delta<=-0.05):  {same_dir_neg}/{len(df_chrom)}",
        f"  cancer>normal (delta>=+0.05):  {same_dir_pos}/{len(df_chrom)}",
        f"  near-zero (|delta|<0.05):       {near_zero}/{len(df_chrom)}",
        "",
        "N3 reference (1 Mb):",
        "  chr22: delta = -0.50  (cancer < normal)",
        "  chr11: delta = +0.14  (cancer > normal)",
        "",
        "Per-chromosome:",
    ]
    for _, r in df_plot.iterrows():
        lines.append(
            f"  {r['chrom']:>6}: nK={r['n_bins_K562']:>4} nG={r['n_bins_GM12878']:>4} "
            f"meanK={r['K562_kappa_mean']:+.4f} meanG={r['GM12878_kappa_mean']:+.4f} "
            f"MWUp={r['mwu_p']:.3g} delta={r['cliff_delta']:+.4f} "
            f"CI=[{r['cliff_delta_lo95']:+.3f},{r['cliff_delta_hi95']:+.3f}]"
        )

    # Verdict
    pd_delta = pooled["cliff_delta"]
    frac_neg = same_dir_neg / max(len(df_chrom), 1)
    frac_pos = same_dir_pos / max(len(df_chrom), 1)
    if np.isfinite(pd_delta) and pd_delta <= -0.20 and frac_neg >= 0.80:
        verdict = (
            "VERDICT: Real result. Genome-wide chromatin contact graphs DO show a "
            "robust cancer-vs-normal curvature signature, but in the OPPOSITE "
            "direction from scRNA-seq (cancer < normal). The Flavahan/Hnisz "
            "TAD-disorganization story is correct; transcriptomic and chromatin "
            "kappa measure structurally different things. The chr22 N3 result "
            "(delta=-0.50) is consistent with the genome-wide signal; the chr11 "
            "result (delta=+0.14 at 1 Mb) was the outlier."
        )
    elif np.isfinite(pd_delta) and pd_delta >= +0.20 and frac_pos >= 0.80:
        verdict = (
            "VERDICT: Surprise. Chromatin contact graphs show the SAME direction "
            "as scRNA-seq (cancer > normal). This is a substrate-independent "
            "extension of the OR-kappa cancer phenotype to 3D genome topology, "
            "and forces a major reframe of the mechanistic reading."
        )
    elif np.isfinite(pd_delta) and abs(pd_delta) < 0.20 and (
        abs(frac_neg - frac_pos) < 0.30
    ):
        verdict = (
            "VERDICT: chr22 in N3 was an outlier. Genome-wide pooled Cliff's "
            "delta is near zero / direction split. Chromatin doesn't share the "
            "cancer kappa signature at 100 kb; phenotype is transcriptomic-only "
            "at the cell-graph level."
        )
    else:
        verdict = (
            f"VERDICT: ambiguous. genome-wide delta={pd_delta:+.3f}, "
            f"{same_dir_neg}/{len(df_chrom)} negative-direction, "
            f"{same_dir_pos}/{len(df_chrom)} positive-direction. "
            "Doesn't match a clean threshold; reread per-chromosome table."
        )
    lines += ["", verdict]
    (HERE / "f4_summary.txt").write_text("\n".join(lines) + "\n")
    log("wrote f4_summary.txt")
    log(verdict)
    log("done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        LOG_FH.close()
