"""
Batch-correction sensitivity analysis for the Ollivier-Ricci curvature (kappa)
malignancy finding.

Reviewer question: does kappa(malignant) > kappa(non-malignant) survive
within-cohort batch correction?

Each of the 7 cohorts is processed independently (own kNN graph), so
cross-cohort batch effects are irrelevant. But within-cohort effects
(multiple lanes / patients / runs) could in principle inflate kappa.

Design (per cohort, 3 largest: Tirosh melanoma, Puram HNSCC, Darmanis GBM):
  (1) ORIGINAL  : HVG-2000 -> PCA-50 -> kNN(k=15) -> Ollivier-Ricci (alpha=0.5).
  (2) HARMONY   : same PCA-50, run harmonypy with patient (or k-means
                  pseudo-batch) as batch var, rebuild kNN on Z_corr, recompute.
  (3) RANDOM    : PCA-50 of i.i.d. Gaussian noise matching the HVG subset
                  shape. Null model: if delta > 0 here too, the finding is a
                  kNN-graph-construction artifact. 3 seeds.
  (4) UMI-ONLY  : 1D embedding = log(1 + total UMI per cell). If delta > 0,
                  sequencing depth, not biology, drives kappa.

Effect size reported is Cliff's delta (malignant vs non-malignant) on per-cell
mean kappa, plus mean-difference and n valid edges.

Pipeline reuses the exact loaders/ORicci primitives from exp/E1_within_patient
and exp/E3_puram_hnscc so numbers are directly comparable to the preprint.

Outputs: results.json + console summary in this directory.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import ot
import pandas as pd
import harmonypy
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
SEED = 20260507
RNG = np.random.default_rng(SEED)


# ---------------- Ollivier-Ricci primitives (verbatim from E1) ---------------
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


def cliffs_delta(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return float((gt - lt) / (len(a) * len(b)))


# ---------------- kappa pipeline (one embedding -> effect-size block) ---------
def kappa_block(X_emb, mal_mask, k=15, n_edges=4000, rng=None):
    """Build kNN on X_emb, compute per-cell kappa, return Cliff's delta
    (malignant vs non), mean diff, group means, n valid edges."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    X_emb = np.asarray(X_emb, float)
    if X_emb.ndim == 1:
        X_emb = X_emb.reshape(-1, 1)
    G = build_knn_graph(X_emb, k=k)
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=rng)
    n_valid = int(sum(1 for v in edge_k.values() if not np.isnan(v)))
    per_cell = per_cell_mean_curvature(G, edge_k)
    kappa = np.array([per_cell.get(i, np.nan) for i in range(X_emb.shape[0])])
    mal = kappa[mal_mask]
    non = kappa[~mal_mask]
    mal = mal[~np.isnan(mal)]
    non = non[~np.isnan(non)]
    return {
        "n_cells": int(X_emb.shape[0]),
        "n_valid_edges": n_valid,
        "n_mal": int(len(mal)),
        "n_non": int(len(non)),
        "mean_mal": float(mal.mean()) if len(mal) else float("nan"),
        "mean_non": float(non.mean()) if len(non) else float("nan"),
        "mean_diff": float(mal.mean() - non.mean())
                     if len(mal) and len(non) else float("nan"),
        "welch_p": float(ttest_ind(mal, non, equal_var=False)[1])
                   if len(mal) > 1 and len(non) > 1 else float("nan"),
        "cliff_delta": float(cliffs_delta(mal, non)),
    }


# ---------------- stratified sample (from E1) --------------------------------
def stratified_sample(group, patient, n_per_group=600, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    sel = []
    for lab in np.unique(group):
        idx_lab = np.where(group == lab)[0]
        pats = patient[idx_lab]
        unique_pats = np.unique(pats)
        per_p = n_per_group // max(len(unique_pats), 1)
        chosen = []
        for p in unique_pats:
            ids = idx_lab[pats == p]
            take = min(len(ids), per_p)
            if take > 0:
                chosen.append(rng.choice(ids, size=take, replace=False))
        chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        remaining = np.setdiff1d(idx_lab, chosen)
        need = n_per_group - len(chosen)
        if need > 0 and len(remaining) > 0:
            extra = rng.choice(remaining,
                               size=min(need, len(remaining)), replace=False)
            chosen = np.concatenate([chosen, extra])
        sel.append(chosen)
    return np.concatenate(sel)


def pca_50(X_hvg, seed=SEED):
    """Mean-center + PCA-50. X_hvg is cells x genes."""
    Xc = X_hvg.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    return PCA(n_components=50, random_state=seed).fit_transform(Xc)


# ---------------- cohort loaders ---------------------------------------------
def load_tirosh():
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("[Tirosh] loading ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    patient = pd.to_numeric(header.iloc[1, 1:], errors="coerce").values
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)

    mask_mal = (malignant == 2)
    mask_T = (malignant == 1) & (nonmal_type == 1)
    keep = mask_mal | mask_T
    Xk = X[:, keep]
    umi_all = Xk.sum(axis=0)
    var = Xk.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xk = Xk[hvg]

    pat = patient[keep]
    grp = np.where(mask_mal[keep], "Malignant", "T_cells")

    rng = np.random.default_rng(SEED)
    sel = stratified_sample(grp, pat, n_per_group=600, rng=rng)

    X_hvg = Xk[:, sel].T  # cells x genes
    return {
        "name": "Tirosh_melanoma",
        "group_pair": ("Malignant", "T_cells"),
        "X_hvg": X_hvg,
        "group": grp[sel],
        "malignant": (grp[sel] == "Malignant"),
        "batch": pat[sel].astype(int).astype(str),  # patient as batch
        "batch_kind": "patient_id",
        "umi": np.log1p(umi_all[sel]),
    }


def load_darmanis():
    EXPR = DATA / "GSE84465_GBM.csv"
    META = DATA / "GSE84465_meta.txt"
    print("[Darmanis] loading ...")
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

    df_expr = pd.read_csv(EXPR, sep=r"\s+", header=0, index_col=0, engine="c")
    df_expr.columns = [c.strip('"') for c in df_expr.columns]
    meta_idx = meta.set_index("cell_id").loc[df_expr.columns]
    cell_type = meta_idx["cell type"].values
    patient = meta_idx["patient id"].values

    Xlog = np.log1p(df_expr.values.astype(np.float32))

    mask_neo = (cell_type == "Neoplastic")
    mask_imm = (cell_type == "Immune cell")
    keep = mask_neo | mask_imm
    Xk = Xlog[:, keep]
    umi_all = Xk.sum(axis=0)
    var = Xk.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xk = Xk[hvg]

    pat = patient[keep]
    grp = np.where(mask_neo[keep], "Neoplastic", "Immune")

    rng = np.random.default_rng(SEED)
    sel = stratified_sample(grp, pat, n_per_group=600, rng=rng)

    X_hvg = Xk[:, sel].T
    return {
        "name": "Darmanis_GBM",
        "group_pair": ("Neoplastic", "Immune"),
        "X_hvg": X_hvg,
        "group": grp[sel],
        "malignant": (grp[sel] == "Neoplastic"),
        "batch": np.asarray(pat[sel]).astype(str),
        "batch_kind": "patient_id",
        "umi": np.log1p(umi_all[sel]),
    }


def load_puram():
    DATA_P = DATA / "GSE103322_HNSCC.txt"
    print("[Puram] loading ...")
    header = pd.read_csv(DATA_P, sep="\t", nrows=6, header=None,
                         low_memory=False)
    cell_ids = header.iloc[0, 1:].values
    cancer = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    noncancer = pd.to_numeric(header.iloc[4, 1:], errors="coerce").values

    expr = pd.read_csv(DATA_P, sep="\t", skiprows=6, header=None,
                       low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)

    mask_mal = (cancer == 1)
    mask_non = (noncancer == 1)
    rng = np.random.default_rng(SEED)
    idx_mal = rng.choice(np.where(mask_mal)[0],
                         size=min(600, int(mask_mal.sum())), replace=False)
    idx_non = rng.choice(np.where(mask_non)[0],
                         size=min(600, int(mask_non.sum())), replace=False)
    sel = np.concatenate([idx_mal, idx_non])

    Xs = X[:, sel]
    umi_sel = Xs.sum(axis=0)
    var = Xs.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xs = Xs[hvg]
    X_hvg = Xs.T  # cells x genes

    grp = np.array(["Malignant"] * len(idx_mal)
                   + ["Non-malignant"] * len(idx_non))
    malignant = (grp == "Malignant")

    # Puram cell IDs encode patient as the HNxx / HNSCCxx prefix
    # (e.g. "HN28_P15_D06_S330_comb" -> patient 28). Use it if it yields a
    # usable number of levels; otherwise fall back to k-means pseudo-batch.
    cell_sel = cell_ids[sel]
    lead = []
    for c in cell_sel:
        c = str(c).strip().strip('"')
        m = re.match(r"\s*HN(?:SCC)?(\d+)", c, re.IGNORECASE)
        lead.append(int(m.group(1)) if m else -1)
    lead = np.array(lead)
    n_unique = len(set(lead[lead >= 0]))
    if n_unique >= 3:
        batch = lead.astype(str)
        batch_kind = f"patient_id_from_cellid ({n_unique} levels)"
    else:
        # k-means pseudo-batch on the PCA embedding (proxy)
        Xp = pca_50(X_hvg)
        km = KMeans(n_clusters=8, random_state=SEED, n_init=10).fit(Xp)
        batch = km.labels_.astype(str)
        batch_kind = "kmeans_pseudo_batch (k=8)"

    return {
        "name": "Puram_HNSCC",
        "group_pair": ("Malignant", "Non-malignant"),
        "X_hvg": X_hvg,
        "group": grp,
        "malignant": malignant,
        "batch": batch,
        "batch_kind": batch_kind,
        "umi": np.log1p(umi_sel),
    }


# ---------------- Harmony correction -----------------------------------------
def run_harmony(X_pca, batch):
    """Return Harmony-corrected embedding (cells x features), matching
    X_pca orientation. harmonypy 1.x stores Z_corr as features x cells;
    2.x stores cells x features -- so we orient by shape, not by version."""
    n_cells = X_pca.shape[0]
    meta = pd.DataFrame({"batch": np.asarray(batch).astype(str)})
    ho = harmonypy.run_harmony(X_pca, meta, "batch", max_iter_harmony=20)
    Z = np.array(ho.Z_corr)
    if Z.shape[0] != n_cells:
        Z = Z.T
    return Z


# ---------------- one cohort --------------------------------------------------
def analyze_cohort(cohort):
    name = cohort["name"]
    X_hvg = cohort["X_hvg"]
    mal_mask = cohort["malignant"]
    batch = cohort["batch"]
    umi = cohort["umi"]
    n_batches = len(np.unique(batch))

    print(f"\n=== {name}  (n={len(mal_mask)}, batches={n_batches} "
          f"[{cohort['batch_kind']}]) ===")

    # (1) ORIGINAL
    print(f"  [1/4] original PCA-50 -> kappa ...")
    X_orig = pca_50(X_hvg)
    orig = kappa_block(X_orig, mal_mask)

    # (2) HARMONY
    print(f"  [2/4] Harmony correction -> kappa ...")
    X_harm = run_harmony(X_orig, batch)
    harm = kappa_block(X_harm, mal_mask)

    # (3) RANDOM NULL (3 seeds): PCA-50 of iid Gaussian noise, same shape
    rand_results = []
    n_cells, n_genes = X_hvg.shape
    for s in (SEED + 100, SEED + 200, SEED + 300):
        print(f"  [3/4] random null seed={s} ...")
        rrng = np.random.default_rng(s)
        X_rand_expr = rrng.standard_normal(size=(n_cells, n_genes)).astype(
            np.float32)
        X_rand = pca_50(X_rand_expr, seed=s)
        rand_results.append({"seed": s, **kappa_block(X_rand, mal_mask,
                                                      rng=rrng)})

    # (4) UMI-ONLY (1D embedding)
    print(f"  [4/4] UMI-only 1D embedding -> kappa ...")
    umi_block = kappa_block(umi.reshape(-1, 1), mal_mask)

    # verdict
    d_orig = orig["cliff_delta"]
    d_harm = harm["cliff_delta"]
    d_rand = np.nanmean([r["cliff_delta"] for r in rand_results])
    d_umi = umi_block["cliff_delta"]
    sign_flip = np.sign(d_orig) != np.sign(d_harm)
    harm_abs_drop = (abs(d_orig) - abs(d_harm)) / abs(d_orig) \
        if d_orig != 0 else float("nan")

    print(f"  --- {name} effect sizes (Cliff delta, mal vs non) ---")
    print(f"      ORIGINAL : {d_orig:+.4f}")
    print(f"      HARMONY  : {d_harm:+.4f}   "
          f"(sign flip={sign_flip}, |d| drop={harm_abs_drop*100:.1f}%)")
    print(f"      RANDOM   : {d_rand:+.4f}   (mean of 3)")
    for r in rand_results:
        print(f"                 seed {r['seed']}: {r['cliff_delta']:+.4f}")
    print(f"      UMI-ONLY : {d_umi:+.4f}")

    return {
        "n_cells": int(len(mal_mask)),
        "batch_kind": cohort["batch_kind"],
        "n_batches": int(n_batches),
        "original": orig,
        "harmony": harm,
        "random_null": rand_results,
        "umi_only": umi_block,
        "cliff_delta_original": d_orig,
        "cliff_delta_harmony": d_harm,
        "cliff_delta_random_mean": float(d_rand),
        "cliff_delta_umi_only": d_umi,
        "sign_flip_after_harmony": bool(sign_flip),
        "harmony_abs_delta_drop_pct": float(harm_abs_drop * 100)
                                      if not np.isnan(harm_abs_drop) else None,
    }


def main():
    print("=" * 72)
    print("BATCH-CORRECTION SENSITIVITY FOR OLLIVIER-RICCI KAPPA (MALIGNANCY)")
    print(f"  seed={SEED}   kNN k=15   n_edges=4000   alpha=0.5")
    print("=" * 72)

    cohorts = [load_tirosh(), load_puram(), load_darmanis()]
    out = {"seed": SEED, "cohorts": {}}
    for c in cohorts:
        out["cohorts"][c["name"]] = analyze_cohort(c)

    # overall verdict per cohort
    summary_lines = []
    for name, r in out["cohorts"].items():
        d_o = r["cliff_delta_original"]
        d_h = r["cliff_delta_harmony"]
        d_r = r["cliff_delta_random_mean"]
        d_u = r["cliff_delta_umi_only"]
        survive = (not r["sign_flip_after_harmony"]) and \
                  (r["harmony_abs_delta_drop_pct"] is None or
                   r["harmony_abs_delta_drop_pct"] < 50)
        if r["sign_flip_after_harmony"]:
            v = "KILLED (sign flip after Harmony)"
        elif r["harmony_abs_delta_drop_pct"] is not None and \
                r["harmony_abs_delta_drop_pct"] >= 50:
            v = "WEAKENED (>50% |delta| drop after Harmony)"
        elif abs(d_r) > 0.2:
            v = "NULL-MATCHED (random noise gives similar delta -> artifact)"
        elif abs(d_u) > 0.2 and np.sign(d_u) == np.sign(d_o):
            v = "DEPTH-CONFOUNDED (UMI-only reproduces delta)"
        else:
            v = "SURVIVES (robust to batch, null, and UMI controls)"
        r["verdict"] = v
        summary_lines.append(
            f"  {name:20s} d_orig={d_o:+.3f}  d_harm={d_h:+.3f}  "
            f"d_rand={d_r:+.3f}  d_umi={d_u:+.3f}  -> {v}"
        )

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        import json
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")

    print("\n" + "=" * 72)
    print("SUMMARY  (Cliff delta malignant vs non, per-cell kappa)")
    print("=" * 72)
    print(f"  {'cohort':20s} {'original':>10s} {'harmony':>10s} "
          f"{'random':>10s} {'umi-only':>10s}   verdict")
    for line in summary_lines:
        print(line)

    # headline
    all_survive = all("SURVIVES" in r["verdict"]
                      for r in out["cohorts"].values())
    print("\nHEADLINE: " +
          ("ALL 3 cohorts robust -> kappa-malignancy finding is NOT a batch/"
           "null/depth artifact." if all_survive else
           "At least one control challenges the finding in one or more "
           "cohorts (see per-cohort verdicts)."))


if __name__ == "__main__":
    main()
