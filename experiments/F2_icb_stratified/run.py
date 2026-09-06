"""
F2 — Per-cell Ollivier-Ricci kappa as ICB-response biomarker, therapy-stratified
================================================================================
Follow-up to N1 (Sade-Feldman 2018, n=19, AUC=0.64). The N1 cohort mixed three
therapy arms (anti-PD1, anti-CTLA4, anti-CTLA4+PD1). Stratifying by therapy
might recover signal that was diluted by mechanism heterogeneity.

Datasets attempted:
  1. Bassez et al. 2021 EGAS00001004809   - controlled access (skipped).
  2. Pauken et al. 2021 GSE186143         - not available.
  3. Yost et al. 2019 GSE123813           - USED. n=11 BCC anti-PD1 patients,
                                             pre/post + paired TCR; cluster-level
                                             metadata identifies CD8 T cells.
                                             Patient response labels from the
                                             published paper.
  Combined with N1 per-patient kappas already computed (n=19) for mega-analysis.

Patient response labels (Yost 2019 BCC anti-PD1):
  Responders   (n=6): su001, su002, su003, su004, su009, su012
  Non-resp.    (n=5): su005, su006, su007, su008, su010

Pipeline (matches N1):
  - Restrict to Pre, CD8_(mem|ex|act)_T_cells per cluster annotation.
  - HVG-2000, log1p(counts), mean-center, PCA-50, kNN k=15.
  - Ollivier-Ricci alpha=0.5, ~4000 sampled edges, per-cell mean kappa.
  - Per-patient mean kappa, MWU + AUC vs response.
  - Per-arm: anti-PD1 only (Yost), anti-PD1 only (N1 subset), anti-CTLA4 only
    (N1 subset), combo (N1 subset).
  - Mega-analysis: pool all anti-PD1 patient means across both cohorts;
    pool all 30 patients across both cohorts.

Outputs (this dir):
  f2_icb_summary.csv   per-arm + cross-cohort mega summary
  f2_per_patient.csv   patient_id, dataset, therapy, response, mean_kappa,
                       n_cells
  f2_distributions.png
  f2_summary.txt
  run.log

Random seed 20260508.
"""

import sys
import time
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import ot
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, ttest_ind

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
ROOT = HERE.parent.parent
LOG  = HERE / "run.log"

# Yost data lives in the E7 directory
E7_DIR = ROOT / "exp" / "E7_clonality"
COUNTS = E7_DIR / "GSE123813_bcc_scRNA_counts.txt.gz"
META   = E7_DIR / "GSE123813_bcc_all_metadata.txt.gz"

# N1 per-patient kappa table (already computed for Sade-Feldman)
N1_PER_PATIENT = ROOT / "exp" / "N1_sadefeldman_icb" / "n1_sadefeldman_per_patient.csv"

SEED    = 20260508
N_HVG   = 2000
N_PCS   = 50
KNN_K   = 15
ALPHA   = 0.5
N_EDGES = 4000

CD8_CLUSTERS = {"CD8_mem_T_cells", "CD8_ex_T_cells", "CD8_act_T_cells"}

# Yost 2019 BCC anti-PD1 response labels (paper supplement / public reanalyses)
YOST_RESPONSE = {
    "su001": "Responder",      "su002": "Responder",
    "su003": "Responder",      "su004": "Responder",
    "su009": "Responder",      "su012": "Responder",
    "su005": "Non-responder",  "su006": "Non-responder",
    "su007": "Non-responder",  "su008": "Non-responder",
    "su010": "Non-responder",
}

RNG = np.random.default_rng(SEED)


# --------------------------- logging ---------------------------
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


# --------------------------- Ollivier-Ricci ---------------------------
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
    if rng is None: rng = np.random.default_rng(0)
    edges = list(G.edges())
    if n_edges is not None and n_edges < len(edges):
        idx = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in idx]
    out = {}
    t0 = time.time()
    for k_edge, (u, v) in enumerate(edges):
        nu = list(G.neighbors(u)); nv = list(G.neighbors(v))
        sup_u = [u] + nu;          sup_v = [v] + nv
        w_u = (np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu))
               if nu else np.array([1.0]))
        w_v = (np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv))
               if nv else np.array([1.0]))
        sub_nodes = list(set(sup_u + sup_v))
        sub = G.subgraph(sub_nodes)
        try:
            dists = {n_: nx.single_source_dijkstra_path_length(sub, n_)
                     for n_ in sup_u}
        except Exception:
            out[(u, v)] = np.nan; continue
        C = np.zeros((len(sup_u), len(sup_v)))
        ok = True
        for i, a in enumerate(sup_u):
            for j, b in enumerate(sup_v):
                d = dists[a].get(b, np.inf)
                if not np.isfinite(d):
                    ok = False; break
                C[i, j] = d
            if not ok: break
        if not ok:
            out[(u, v)] = np.nan; continue
        W = ot.emd2(w_u, w_v, C)
        d_uv = G[u][v]["weight"]
        out[(u, v)] = float(1.0 - W / d_uv) if d_uv > 0 else np.nan
        if (k_edge + 1) % 500 == 0:
            log(f"      OR edge {k_edge+1}/{len(edges)} "
                f"({time.time()-t0:.1f}s)")
    return out


def per_cell_mean_curvature(G, edge_kappa):
    by_cell = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k): continue
        by_cell[u].append(k); by_cell[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan
            for n, vs in by_cell.items()}


# --------------------------- counts streaming ---------------------------
def stream_counts(cells_wanted):
    """Stream gzipped counts file, return (gene_names, cells_kept, X[g, n]).
    Yost layout: header line is just cell-IDs (n_cells columns), data rows
    have an extra leading gene-name column.
    """
    log(f"  streaming counts (target {len(cells_wanted)} cells) ...")
    with gzip.open(COUNTS, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        keep_mask = np.array([c in cells_wanted for c in header])
        keep_idx = np.where(keep_mask)[0]
        cells_kept = [header[i] for i in keep_idx]
        log(f"  matched {len(keep_idx)} of {len(header)} cell columns")
        # data column for header[k] is at index k+1 in the data row
        target_data_cols = (keep_idx + 1).tolist()
        gene_names = []
        rows = []
        t0 = time.time()
        for ln, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(target_data_cols) + 1:
                continue
            gene_names.append(parts[0])
            arr = np.array([parts[c] for c in target_data_cols],
                           dtype=np.float32)
            rows.append(arr)
            if (ln + 1) % 5000 == 0:
                log(f"    {ln+1} genes streamed "
                    f"({time.time()-t0:.1f}s)")
    X = np.vstack(rows).astype(np.float32)
    log(f"  counts matrix: {X.shape} (genes x cells), "
        f"{time.time()-t0:.1f}s")
    return gene_names, np.array(cells_kept), X


# --------------------------- pipeline core ---------------------------
def kappa_pipeline(X_genes_by_cells, label="cohort"):
    """Run HVG -> PCA -> kNN -> OR-kappa, return per-cell kappa array
    aligned to cells (columns of X)."""
    log(f"  [{label}] HVG-{N_HVG} from {X_genes_by_cells.shape[0]} genes ...")
    Xlog = np.log1p(X_genes_by_cells)
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:N_HVG]
    Xlog = Xlog[hvg]
    log(f"  HVG matrix: {Xlog.shape}")

    Xc = Xlog.T.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n_pcs = min(N_PCS, min(Xc.shape) - 1)
    Xpca = PCA(n_components=n_pcs, random_state=SEED).fit_transform(Xc)
    log(f"  PCA -> {Xpca.shape}")

    G = build_knn_graph(Xpca, k=KNN_K)
    log(f"  kNN graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges")

    n_edges = min(N_EDGES, G.number_of_edges())
    log(f"  Ollivier-Ricci alpha={ALPHA} on {n_edges} edges ...")
    t0 = time.time()
    ek = ollivier_ricci_edges(G, alpha=ALPHA, n_edges=n_edges, rng=RNG)
    n_valid = sum(1 for v in ek.values() if not np.isnan(v))
    log(f"  valid edges: {n_valid}/{len(ek)} "
        f"({time.time()-t0:.1f}s)")

    pcm = per_cell_mean_curvature(G, ek)
    return np.array([pcm[i] for i in range(Xc.shape[0])])


# --------------------------- analyses ---------------------------
def directional_auc(scores, labels):
    """Return (auc_R, auc_NR, best, label).

    scores: float, labels: 0/1 with 1 = Responder."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan, np.nan, np.nan, "n/a"
    auc_R  = roc_auc_score(labels, scores)
    auc_NR = roc_auc_score(labels, -scores)
    best = max(auc_R, auc_NR)
    direction = ("high-kappa = Responder" if auc_R >= auc_NR
                 else "high-kappa = Non-responder")
    return auc_R, auc_NR, best, direction


def arm_stats(arm_name, df):
    """Compute per-patient MWU + AUC for one therapy arm."""
    if df.empty:
        return None
    R  = df.loc[df["response"] == "Responder",     "mean_kappa"].values
    NR = df.loc[df["response"] == "Non-responder", "mean_kappa"].values
    n_R, n_NR = len(R), len(NR)
    if n_R < 2 or n_NR < 2:
        return dict(arm=arm_name, n_R=n_R, n_NR=n_NR,
                    mwu_p=np.nan, auc_R=np.nan, auc_NR=np.nan,
                    auc_best=np.nan, direction="insufficient",
                    mean_R=float(np.nanmean(R)) if n_R else np.nan,
                    mean_NR=float(np.nanmean(NR)) if n_NR else np.nan,
                    n_cells_total=int(df["n_cells"].sum()))
    u, p = mannwhitneyu(R, NR, alternative="two-sided")
    y = (df["response"] == "Responder").astype(int).values
    score = df["mean_kappa"].values
    auc_R, auc_NR, best, direction = directional_auc(score, y)
    return dict(arm=arm_name, n_R=n_R, n_NR=n_NR,
                mwu_U=float(u), mwu_p=float(p),
                auc_R=float(auc_R), auc_NR=float(auc_NR),
                auc_best=float(best), direction=direction,
                mean_R=float(np.mean(R)), mean_NR=float(np.mean(NR)),
                n_cells_total=int(df["n_cells"].sum()))


# --------------------------- main ---------------------------
def main():
    t_start = time.time()
    log("=" * 72)
    log("F2 Yost+SadeFeldman -- per-cell kappa as ICB biomarker, therapy-stratified")
    log(f"seed={SEED}  hvg={N_HVG}  pcs={N_PCS}  k={KNN_K}  "
        f"alpha={ALPHA}  n_edges={N_EDGES}")
    log("=" * 72)

    # ---------- Yost: load metadata, pick Pre CD8 T cells ----------
    log("\n[1] loading Yost metadata + identifying Pre CD8 T cells ...")
    meta = pd.read_csv(META, sep="\t")
    log(f"  meta rows={len(meta)}  cols={list(meta.columns)}")
    pre_cd8 = meta[(meta["treatment"] == "pre") &
                   (meta["cluster"].isin(CD8_CLUSTERS))].copy()
    pre_cd8["response"] = pre_cd8["patient"].map(YOST_RESPONSE)
    pre_cd8 = pre_cd8.dropna(subset=["response"])
    log(f"  Pre CD8 T cells (cluster-based): {len(pre_cd8)}")
    log("  per-patient cell counts:")
    counts_per_pt = pre_cd8.groupby(["patient", "response"]).size()
    for (pt, resp), n in counts_per_pt.items():
        log(f"    {pt:6s} {resp:14s} {n}")

    cells_wanted = set(pre_cd8["cell.id"].astype(str))

    # ---------- Yost: stream counts, compute kappa ----------
    log("\n[2] streaming Yost counts + running kappa pipeline ...")
    gene_names, cells_kept, X = stream_counts(cells_wanted)
    # align metadata to columns of X
    cell_to_pt = dict(zip(pre_cd8["cell.id"].astype(str), pre_cd8["patient"]))
    cell_to_resp = dict(zip(pre_cd8["cell.id"].astype(str), pre_cd8["response"]))
    pat_arr  = np.array([cell_to_pt.get(c, "") for c in cells_kept])
    resp_arr = np.array([cell_to_resp.get(c, "") for c in cells_kept])
    log(f"  cells aligned: {len(cells_kept)}, "
        f"unique patients: {len(set(pat_arr))}")

    kappa = kappa_pipeline(X, label="Yost-CD8-Pre")
    log(f"  kappa stats: mean={np.nanmean(kappa):.4f} "
        f"std={np.nanstd(kappa):.4f}  "
        f"valid={(~np.isnan(kappa)).sum()}/{len(kappa)}")

    # per-patient mean kappa for Yost
    yost_df = pd.DataFrame({
        "patient_id": pat_arr,
        "response":   resp_arr,
        "kappa":      kappa,
    })
    yost_df = yost_df.dropna(subset=["kappa"])
    yost_per_pt = (yost_df.groupby("patient_id")
                   .agg(n_cells=("kappa", "size"),
                        mean_kappa=("kappa", "mean"),
                        std_kappa=("kappa", "std"),
                        response=("response", "first"))
                   .reset_index())
    yost_per_pt["dataset"] = "Yost2019_BCC"
    yost_per_pt["therapy"] = "anti-PD1"
    log("\n[3] Yost per-patient mean kappa:")
    log(yost_per_pt.to_string(index=False))

    # ---------- N1 per-patient (Sade-Feldman) load ----------
    log("\n[4] loading N1 Sade-Feldman per-patient kappa ...")
    n1 = pd.read_csv(N1_PER_PATIENT)
    n1["dataset"] = "SadeFeldman2018_Mel"
    n1 = n1.rename(columns={"patient_id": "patient_id"})
    log(f"  N1 patients: {len(n1)}")
    log(n1.to_string(index=False))

    # ---------- combine into single per-patient table ----------
    cols_keep = ["dataset", "patient_id", "therapy", "response",
                 "n_cells", "mean_kappa"]
    combined = pd.concat([
        yost_per_pt[["dataset", "patient_id", "therapy",
                     "response", "n_cells", "mean_kappa"]],
        n1[["dataset", "patient_id", "therapy", "response",
            "n_cells", "mean_kappa"]],
    ], ignore_index=True)
    log(f"\n[5] combined per-patient table: {len(combined)} rows, "
        f"{combined['dataset'].nunique()} datasets")

    # ---------- per-arm + mega analysis ----------
    log("\n[6] per-arm and mega-cohort statistics ...")
    arm_results = []

    # Yost anti-PD1 only
    arm_results.append(arm_stats("Yost_antiPD1",
                                 combined[combined["dataset"]
                                          == "Yost2019_BCC"]))
    # N1 stratified arms
    n1_only = combined[combined["dataset"] == "SadeFeldman2018_Mel"]
    arm_results.append(arm_stats("N1_antiPD1_only",
                                 n1_only[n1_only["therapy"] == "anti-PD1"]))
    arm_results.append(arm_stats("N1_antiCTLA4_only",
                                 n1_only[n1_only["therapy"] == "anti-CTLA4"]))
    arm_results.append(arm_stats("N1_combo_CTLA4_PD1",
                                 n1_only[n1_only["therapy"] == "anti-CTLA4+PD1"]))

    # Combined anti-PD1 mega (Yost + N1 anti-PD1 subset)
    pd1_mega = combined[
        ((combined["dataset"] == "Yost2019_BCC") &
         (combined["therapy"] == "anti-PD1")) |
        ((combined["dataset"] == "SadeFeldman2018_Mel") &
         (combined["therapy"] == "anti-PD1"))
    ]
    arm_results.append(arm_stats("Mega_antiPD1_only", pd1_mega))

    # Combined all-patients mega
    arm_results.append(arm_stats("Mega_all", combined))

    arm_results = [r for r in arm_results if r is not None]
    summary_df = pd.DataFrame(arm_results)
    log("\nPer-arm summary:")
    log(summary_df.to_string(index=False))

    # ---------- outputs ----------
    log("\n[7] writing outputs ...")
    summary_df.to_csv(HERE / "f2_icb_summary.csv", index=False)
    combined.to_csv(HERE / "f2_per_patient.csv", index=False)
    log(f"  wrote f2_icb_summary.csv  ({len(summary_df)} rows)")
    log(f"  wrote f2_per_patient.csv  ({len(combined)} rows)")

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    arms_to_plot = [
        ("Yost_antiPD1", combined[combined["dataset"] == "Yost2019_BCC"]),
        ("N1_antiPD1_only",
         n1_only[n1_only["therapy"] == "anti-PD1"]),
        ("Mega_antiPD1", pd1_mega),
    ]
    for ax, (name, df) in zip(axes, arms_to_plot):
        groups = ["Non-responder", "Responder"]
        data_box = [df.loc[df["response"] == g, "mean_kappa"].values
                    for g in groups]
        bp = ax.boxplot(data_box, positions=[0, 1], widths=0.4,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["#d75555", "#5588cc"]):
            patch.set_facecolor(color); patch.set_alpha(0.4)
        rng = np.random.default_rng(SEED)
        for i, g in enumerate(groups):
            sub = df[df["response"] == g]
            x = i + (rng.random(len(sub)) - 0.5) * 0.18
            ax.scatter(x, sub["mean_kappa"].values, s=70, alpha=0.85,
                       color=("#d75555" if g == "Non-responder" else "#5588cc"),
                       edgecolor="k", linewidth=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(groups)
        ax.set_ylabel("mean per-cell kappa")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        # annotate
        row = next((r for r in arm_results if r["arm"] == name
                    or r["arm"].startswith(name)), None)
        if row is not None:
            ax.set_title(f"{name}\n"
                         f"n_R={row.get('n_R','?')}  "
                         f"n_NR={row.get('n_NR','?')}  "
                         f"AUC={row.get('auc_best',float('nan')):.3f}  "
                         f"p={row.get('mwu_p',float('nan')):.3g}",
                         fontsize=9)
    fig.suptitle("F2 ICB-response biomarker: per-patient mean Ollivier-Ricci "
                 "kappa, therapy-stratified", fontsize=11)
    plt.tight_layout()
    plt.savefig(HERE / "f2_distributions.png", dpi=130)
    plt.close()
    log(f"  wrote f2_distributions.png")

    # ---------- text summary + verdict ----------
    lines = []
    lines.append("F2 ICB-response biomarker: per-cell Ollivier-Ricci kappa, "
                 "therapy-stratified")
    lines.append("=" * 72)
    lines.append(f"seed={SEED}  hvg={N_HVG}  pcs={N_PCS}  k={KNN_K}  "
                 f"alpha={ALPHA}  n_edges={N_EDGES}")
    lines.append("")
    lines.append("Datasets:")
    lines.append("  Yost 2019 BCC (GSE123813), anti-PD1, n=11 patients")
    lines.append("  Sade-Feldman 2018 (GSE120575), n=19 patients (mixed therapy)")
    lines.append("")
    lines.append("Yost per-patient kappa table:")
    lines.append(yost_per_pt.to_string(index=False))
    lines.append("")
    lines.append("Combined per-patient table:")
    lines.append(combined.to_string(index=False))
    lines.append("")
    lines.append("Per-arm summary (mean kappa as single-feature classifier):")
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("VERDICTS")
    lines.append("-" * 72)
    for r in arm_results:
        auc = r.get("auc_best", np.nan)
        if np.isnan(auc):
            verdict = "INSUFFICIENT DATA"
        elif auc >= 0.70:
            verdict = "REAL BIOMARKER SIGNAL (AUC>=0.70)"
        elif auc >= 0.55:
            verdict = "WEAK TREND (0.55-0.70)"
        else:
            verdict = "NO SIGNAL (<=0.55)"
        lines.append(f"  {r['arm']:24s}  AUC={auc:.3f}  "
                     f"({r.get('direction','?')})  -> {verdict}")
    lines.append("")
    lines.append(f"Total time: {time.time()-t_start:.1f}s")

    (HERE / "f2_summary.txt").write_text("\n".join(lines))
    log(f"  wrote f2_summary.txt")

    log(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
