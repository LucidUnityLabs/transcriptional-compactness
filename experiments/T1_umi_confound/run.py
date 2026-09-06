"""
T1 — UMI / library-size confound test for the per-cell Ollivier-Ricci kappa
finding on Darmanis 2017 GBM (GSE84465).

Hypothesis under test:
  Malignant cells often carry higher total UMI than infiltrating immune cells.
  PCA-of-log-counts may inherit a depth gradient that makes high-UMI cells
  geometrically dense, raising kappa mechanically. If the malignant-vs-immune
  kappa separation collapses *within* UMI quintiles, the headline was depth.

Protocol:
  1. Load Darmanis raw counts (genes x cells, space-separated).
  2. Per-cell total counts = library size = sum across genes.
  3. Stratify all (neoplastic + immune) cells into 5 UMI quintiles.
  4. Per-quintile counts of neoplastic vs immune.
  5. Within each quintile run: HVG-2000 (within), PCA-50, kNN k=15,
     Ollivier-Ricci alpha=0.5 on min(2500,|E|) sampled edges, per-cell mean kappa.
     Compare neoplastic vs immune via Welch t, MWU, Cliff's delta.
  6. Reference: same pipeline on all combined cells (no stratification).
  7. Optional: also stratify on mean kNN distance (a "geometric depth" proxy).

Outputs:
  t1_umi_quintiles.csv
  t1_umi_confound_summary.png
  t1_summary.txt
  (optional) t1_knn_distance_quintiles.csv
"""

import re
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


# ---------------- Ollivier-Ricci primitives ----------------
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
    return G, dists


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


# ---------------- effect-size helpers ----------------
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


def cliffs_delta_bootstrap_ci(a, b, n_boot=500, rng=None, alpha=0.05):
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        deltas[i] = cliffs_delta(ai, bi)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------- data loading ----------------
def load_darmanis_full():
    """Return raw counts (genes x cells), cell_type array, total_counts array.

    Restricts cells to Neoplastic or Immune cell.
    """
    EXPR = DATA / "GSE84465_GBM.csv"
    META = DATA / "GSE84465_meta.txt"
    print("[load] parsing meta ...")
    text = META.read_text(errors="replace")
    fields = {}
    for line in text.splitlines():
        if not line.startswith("!Sample_characteristics_ch1"):
            continue
        parts = re.findall(r'"([^"]*)"', line)
        if not parts:
            continue
        prefix = parts[0].split(":", 1)[0].strip()
        values = [p.split(":", 1)[1].strip() if ":" in p else "" for p in parts]
        fields[prefix] = values
    meta = pd.DataFrame(fields)
    meta["cell_id"] = meta["plate id"] + "." + meta["well"]

    print(f"[load] reading expression matrix ({EXPR.stat().st_size/1e6:.0f} MB) ...")
    df = pd.read_csv(EXPR, sep=r"\s+", header=0, index_col=0, engine="c")
    df.columns = [c.strip('"') for c in df.columns]
    print(f"[load] raw expression: {df.shape} (genes x cells)")

    # align metadata
    meta_idx = meta.set_index("cell_id").loc[df.columns]
    cell_type = meta_idx["cell type"].values

    counts = df.values.astype(np.float32)  # raw counts; genes x cells

    # total UMI per cell = column sum
    total_counts = counts.sum(axis=0)

    # mask to neoplastic + immune
    keep = (cell_type == "Neoplastic") | (cell_type == "Immune cell")
    print(f"[load] keeping {keep.sum()} cells "
          f"(neo={int((cell_type=='Neoplastic').sum())}, "
          f"imm={int((cell_type=='Immune cell').sum())})")
    counts = counts[:, keep]
    cell_type = cell_type[keep]
    total_counts = total_counts[keep]
    return counts, cell_type, total_counts


# ---------------- pipeline within a stratum ----------------
def run_pipeline(counts_strat, cell_type_strat, label="(strat)",
                 k=15, n_edges=2500, n_pca=50, n_hvg=2000):
    """
    counts_strat: raw counts genes x cells (subset)
    cell_type_strat: array length cells
    Returns dict of summary stats, and per-cell kappa array (len=cells).
    """
    n_cells = counts_strat.shape[1]
    if n_cells < 30:
        print(f"  {label}: only {n_cells} cells, skip")
        return None, None

    Xlog = np.log1p(counts_strat)
    var = Xlog.var(axis=1)
    n_hvg_eff = min(n_hvg, (var > 0).sum())
    hvg = np.argsort(var)[::-1][:n_hvg_eff]
    Xlog = Xlog[hvg]

    # cells x genes, mean-centered
    Xs = Xlog.T
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    n_pca_eff = min(n_pca, Xs.shape[0] - 1, Xs.shape[1] - 1)
    Xpca = PCA(n_components=n_pca_eff, random_state=SEED).fit_transform(Xs)

    G, knn_dists = build_knn_graph(Xpca, k=k)
    n_e = G.number_of_edges()
    n_use = min(n_edges, n_e)
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_use, rng=RNG)
    n_valid = sum(1 for v in edge_k.values() if not np.isnan(v))
    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa = np.array([per_cell.get(i, np.nan) for i in range(n_cells)])

    a = kappa[(cell_type_strat == "Neoplastic") & ~np.isnan(kappa)]
    b = kappa[(cell_type_strat == "Immune cell") & ~np.isnan(kappa)]

    res = {
        "label": label,
        "n_cells": int(n_cells),
        "n_neoplastic": int((cell_type_strat == "Neoplastic").sum()),
        "n_immune": int((cell_type_strat == "Immune cell").sum()),
        "n_edges": int(n_e),
        "n_edges_used": int(n_use),
        "n_edges_valid": int(n_valid),
        "n_neo_valid_kappa": int(len(a)),
        "n_imm_valid_kappa": int(len(b)),
        "mean_kappa_neo": float(a.mean()) if len(a) else np.nan,
        "mean_kappa_imm": float(b.mean()) if len(b) else np.nan,
        "std_kappa_neo": float(a.std()) if len(a) else np.nan,
        "std_kappa_imm": float(b.std()) if len(b) else np.nan,
        "diff_neo_minus_imm": float(a.mean() - b.mean())
        if len(a) and len(b) else np.nan,
        "welch_p": np.nan,
        "welch_t": np.nan,
        "mwu_p": np.nan,
        "cliff_delta": np.nan,
        "cliff_ci_lo": np.nan,
        "cliff_ci_hi": np.nan,
    }
    if len(a) >= 5 and len(b) >= 5:
        t_stat, p_t = ttest_ind(a, b, equal_var=False)
        u_stat, p_u = mannwhitneyu(a, b, alternative="two-sided")
        d = cliffs_delta(a, b)
        lo, hi = cliffs_delta_bootstrap_ci(a, b, n_boot=400, rng=RNG)
        res.update({
            "welch_t": float(t_stat),
            "welch_p": float(p_t),
            "mwu_p": float(p_u),
            "cliff_delta": float(d),
            "cliff_ci_lo": float(lo),
            "cliff_ci_hi": float(hi),
        })
    print(f"  {label}: n_cells={n_cells}  neo={res['n_neoplastic']} "
          f"imm={res['n_immune']}  diff_kappa={res['diff_neo_minus_imm']:+.4f}  "
          f"Welch p={res['welch_p']:.3g}  Cliff d={res['cliff_delta']:+.3f} "
          f"[{res['cliff_ci_lo']:+.3f}, {res['cliff_ci_hi']:+.3f}]")
    return res, kappa


# ---------------- stratified runner ----------------
def stratify_and_run(counts, cell_type, score, score_name, n_q=5):
    """
    Bin cells into n_q quantiles by `score`, run pipeline within each.
    Returns DataFrame with one row per quintile.
    """
    print(f"\n--- stratifying by {score_name} into {n_q} quintiles ---")
    qedges = np.quantile(score, np.linspace(0, 1, n_q + 1))
    qedges[0] -= 1e-9
    qedges[-1] += 1e-9
    q_idx = np.digitize(score, qedges[1:-1], right=False)
    rows = []
    for q in range(n_q):
        mask = q_idx == q
        c = counts[:, mask]
        ct = cell_type[mask]
        lo = float(np.quantile(score[mask], 0.0))
        hi = float(np.quantile(score[mask], 1.0))
        label = f"Q{q+1} [{lo:.0f}-{hi:.0f}]"
        n_neo = int((ct == "Neoplastic").sum())
        n_imm = int((ct == "Immune cell").sum())
        print(f"  {label}: n={mask.sum()}  neo={n_neo}  imm={n_imm}")
        if n_neo < 10 or n_imm < 10:
            rows.append({
                "q_index": q + 1,
                f"q_low_{score_name}": lo,
                f"q_high_{score_name}": hi,
                "n_total": int(mask.sum()),
                "n_neoplastic": n_neo,
                "n_immune": n_imm,
                "mean_kappa_neo": np.nan,
                "mean_kappa_imm": np.nan,
                "diff_kappa": np.nan,
                "welch_p": np.nan,
                "mwu_p": np.nan,
                "cliff_delta": np.nan,
                "cliff_ci_lo": np.nan,
                "cliff_ci_hi": np.nan,
                "skipped_low_n": True,
            })
            continue
        res, _ = run_pipeline(c, ct, label=label)
        rows.append({
            "q_index": q + 1,
            f"q_low_{score_name}": lo,
            f"q_high_{score_name}": hi,
            "n_total": int(mask.sum()),
            "n_neoplastic": n_neo,
            "n_immune": n_imm,
            "mean_kappa_neo": res["mean_kappa_neo"],
            "mean_kappa_imm": res["mean_kappa_imm"],
            "diff_kappa": res["diff_neo_minus_imm"],
            "welch_p": res["welch_p"],
            "mwu_p": res["mwu_p"],
            "cliff_delta": res["cliff_delta"],
            "cliff_ci_lo": res["cliff_ci_lo"],
            "cliff_ci_hi": res["cliff_ci_hi"],
            "skipped_low_n": False,
        })
    return pd.DataFrame(rows)


def make_forest(df_quint, all_cliff, all_lo, all_hi, all_p, out_png,
                title_extra=""):
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df_quint) + 3))
    y = np.arange(len(df_quint))[::-1]  # plot Q1 at top
    deltas = df_quint["cliff_delta"].values
    los = df_quint["cliff_ci_lo"].values
    his = df_quint["cliff_ci_hi"].values
    ax.scatter(deltas, y, color="C0", s=80, zorder=3, label="quintile Cliff d")
    for i, (d, lo, hi) in enumerate(zip(deltas, los, his)):
        if np.isnan(lo) or np.isnan(hi):
            continue
        ax.hlines(y[i], lo, hi, color="C0", lw=2.0)
    # all-cells reference
    ax.axvline(all_cliff, color="red", ls="--",
               label=f"all-cells Cliff d={all_cliff:+.3f} (p={all_p:.2g})")
    ax.axvspan(all_lo, all_hi, color="red", alpha=0.10)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(0.15, color="grey", lw=0.4, ls=":")
    ax.axvline(-0.15, color="grey", lw=0.4, ls=":")
    ax.axvline(0.20, color="grey", lw=0.4, ls=":")
    ax.set_yticks(y)
    labels = []
    for _, r in df_quint.iterrows():
        # find low/high col
        col_lo = [c for c in df_quint.columns if c.startswith("q_low_")][0]
        col_hi = [c for c in df_quint.columns if c.startswith("q_high_")][0]
        labels.append(f"Q{int(r['q_index'])}\n[{r[col_lo]:.0f}-{r[col_hi]:.0f}]\n"
                      f"neo={int(r['n_neoplastic'])} imm={int(r['n_immune'])}")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cliff's delta (Neoplastic - Immune)  per-cell mean kappa")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"T1 UMI confound — Cliff's delta per quintile{title_extra}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()


def write_summary_txt(out_path, q_df_umi, all_res, q_df_knn=None):
    lines = []
    lines.append("T1 UMI / library-size confound test — Darmanis 2017 GBM")
    lines.append(f"seed={SEED}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Reference (no stratification, all neoplastic + immune cells):")
    lines.append(f"  n_neo={all_res['n_neoplastic']}  n_imm={all_res['n_immune']}")
    lines.append(f"  mean_kappa_neo = {all_res['mean_kappa_neo']:+.4f}  "
                 f"mean_kappa_imm = {all_res['mean_kappa_imm']:+.4f}")
    lines.append(f"  diff (neo - imm) = {all_res['diff_neo_minus_imm']:+.4f}")
    lines.append(f"  Welch p = {all_res['welch_p']:.3g}   "
                 f"MWU p = {all_res['mwu_p']:.3g}")
    lines.append(f"  Cliff d = {all_res['cliff_delta']:+.3f} "
                 f"[{all_res['cliff_ci_lo']:+.3f}, {all_res['cliff_ci_hi']:+.3f}]")
    lines.append("")
    lines.append("Within UMI quintiles:")
    for _, r in q_df_umi.iterrows():
        lines.append(
            f"  Q{int(r['q_index'])}  UMI=[{r['q_low_umi']:.0f}-{r['q_high_umi']:.0f}]  "
            f"n={int(r['n_total'])} (neo={int(r['n_neoplastic'])} "
            f"imm={int(r['n_immune'])})  "
            f"diff_kappa={r['diff_kappa']:+.4f}  "
            f"Welch p={r['welch_p']:.3g}  MWU p={r['mwu_p']:.3g}  "
            f"Cliff d={r['cliff_delta']:+.3f} "
            f"[{r['cliff_ci_lo']:+.3f}, {r['cliff_ci_hi']:+.3f}]"
        )
    lines.append("")

    # Verdict
    valid = q_df_umi.dropna(subset=["cliff_delta"])
    if len(valid):
        n_q = len(valid)
        n_below_015 = int((valid["cliff_delta"].abs() < 0.15).sum())
        n_above_02_pos = int((valid["cliff_delta"] > 0.2).sum())
        n_above_02_neg = int((valid["cliff_delta"] < -0.2).sum())
        n_p_ns = int((valid["welch_p"] > 0.05).sum())

        lines.append("Decision criteria:")
        lines.append(f"  quintiles with |Cliff d| < 0.15: {n_below_015}/{n_q}")
        lines.append(f"  quintiles with Cliff d > +0.20 (neo > imm): {n_above_02_pos}/{n_q}")
        lines.append(f"  quintiles with Cliff d < -0.20 (imm > neo): {n_above_02_neg}/{n_q}")
        lines.append(f"  quintiles with Welch p > 0.05: {n_p_ns}/{n_q}")
        lines.append("")

        kill = (n_below_015 == n_q) or (n_p_ns == n_q)
        survive_pos = n_above_02_pos >= max(4, n_q - 1)
        if kill:
            verdict = "KILLED — depth confound explains the headline kappa effect"
        elif survive_pos:
            verdict = "SURVIVES — kappa signal is biology, not depth"
        else:
            verdict = "INTERMEDIATE — partial depth-mediated; see per-quintile values"
        lines.append(f"VERDICT: {verdict}")

    if q_df_knn is not None:
        lines.append("")
        lines.append("=" * 70)
        lines.append("Inverse stratification: by mean kNN distance (geometric depth proxy)")
        for _, r in q_df_knn.iterrows():
            col_lo = [c for c in q_df_knn.columns if c.startswith("q_low_")][0]
            col_hi = [c for c in q_df_knn.columns if c.startswith("q_high_")][0]
            lines.append(
                f"  Q{int(r['q_index'])}  knnd=[{r[col_lo]:.4f}-{r[col_hi]:.4f}]  "
                f"n={int(r['n_total'])} (neo={int(r['n_neoplastic'])} "
                f"imm={int(r['n_immune'])})  "
                f"diff_kappa={r['diff_kappa']:+.4f}  "
                f"Welch p={r['welch_p']:.3g}  MWU p={r['mwu_p']:.3g}  "
                f"Cliff d={r['cliff_delta']:+.3f} "
                f"[{r['cliff_ci_lo']:+.3f}, {r['cliff_ci_hi']:+.3f}]"
            )

    out_path.write_text("\n".join(lines))


def main():
    print("=" * 70)
    print("T1 — UMI / library-size confound test for Ollivier-Ricci kappa")
    print(f"     seed={SEED}  dataset=Darmanis 2017 GBM (GSE84465)")
    print("=" * 70)

    counts, cell_type, total_counts = load_darmanis_full()

    # Quick sanity print: UMI distribution by cell type
    for ct_lab in ["Neoplastic", "Immune cell"]:
        v = total_counts[cell_type == ct_lab]
        print(f"  {ct_lab:14s}  n={len(v)}  median UMI={np.median(v):.0f}  "
              f"mean={v.mean():.0f}  q25={np.quantile(v,.25):.0f}  "
              f"q75={np.quantile(v,.75):.0f}")

    # 1) Reference: all-cells comparison
    print("\n--- Reference: all neoplastic + immune cells (no stratification) ---")
    all_res, _ = run_pipeline(counts, cell_type, label="ALL",
                              k=15, n_edges=2500)

    # 2) UMI-quintile stratification
    print("\n=== UMI-quintile stratification ===")
    q_df_umi = stratify_and_run(counts, cell_type, total_counts,
                                score_name="umi", n_q=5)
    q_df_umi.to_csv(HERE / "t1_umi_quintiles.csv", index=False)
    print(f"\nWrote {HERE / 't1_umi_quintiles.csv'}")

    # 3) Plot
    make_forest(
        q_df_umi,
        all_cliff=all_res["cliff_delta"],
        all_lo=all_res["cliff_ci_lo"],
        all_hi=all_res["cliff_ci_hi"],
        all_p=all_res["welch_p"],
        out_png=HERE / "t1_umi_confound_summary.png",
        title_extra=" (Darmanis GBM)",
    )
    print(f"Wrote {HERE / 't1_umi_confound_summary.png'}")

    # 4) Optional: stratify by mean kNN distance on the all-cells embedding
    q_df_knn = None
    try:
        print("\n=== inverse stratification: by mean kNN distance "
              "(global PCA-50 embedding) ===")
        # build embedding once on all cells
        Xlog = np.log1p(counts)
        var = Xlog.var(axis=1)
        hvg = np.argsort(var)[::-1][:2000]
        Xlog_h = Xlog[hvg].T
        Xlog_h = Xlog_h - Xlog_h.mean(axis=0, keepdims=True)
        Xpca_all = PCA(n_components=50, random_state=SEED).fit_transform(Xlog_h)
        nbr = NearestNeighbors(n_neighbors=16).fit(Xpca_all)
        knn_d, _ = nbr.kneighbors(Xpca_all)
        mean_knn_d = knn_d[:, 1:].mean(axis=1)  # exclude self
        q_df_knn = stratify_and_run(counts, cell_type, mean_knn_d,
                                    score_name="knnd", n_q=5)
        q_df_knn.to_csv(HERE / "t1_knn_distance_quintiles.csv", index=False)
        print(f"Wrote {HERE / 't1_knn_distance_quintiles.csv'}")

        make_forest(
            q_df_knn,
            all_cliff=all_res["cliff_delta"],
            all_lo=all_res["cliff_ci_lo"],
            all_hi=all_res["cliff_ci_hi"],
            all_p=all_res["welch_p"],
            out_png=HERE / "t1_knn_distance_summary.png",
            title_extra=" (Darmanis GBM, kNN-distance strat)",
        )
        print(f"Wrote {HERE / 't1_knn_distance_summary.png'}")
    except Exception as e:
        print(f"  kNN-distance stratification skipped: {e}")

    # 5) Summary
    write_summary_txt(HERE / "t1_summary.txt", q_df_umi, all_res, q_df_knn)
    print(f"\nWrote {HERE / 't1_summary.txt'}")


if __name__ == "__main__":
    main()
