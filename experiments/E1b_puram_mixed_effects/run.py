"""
E1b -- Within-patient mixed-effects test of Ollivier-Ricci curvature on
Puram et al. 2017 HNSCC (GSE103322).

E1 ran the within-patient mixed-effects model for Tirosh melanoma
(fe = +0.148, p = 1e-79) and Darmanis GBM (fe = +0.071, p = 6e-29) but
NOT for Puram HNSCC. Puram is the cohort that sign-flipped under Harmony
batch correction (see exp/BATCH_CORRECTION/), so a reviewer will demand
the raw within-patient effect here to confirm the curvature signal is not
a Harmony artifact in either direction.

This script replicates the E1 protocol exactly (same curvature primitives,
same stratified sampling, same MixedLM formula, same per-patient >=30/30
threshold) on Puram. Patient ID is parsed from the cell barcode
(`HN<n>_...` or `HNSCC_<n>_...`); 109 `HNSCC_combo1_...` pooled cells with
no patient origin are dropped, leaving 17 patients.

Pipeline:
  1) Load expression + per-cell patient/malignant labels.
  2) HVG-2000 -> mean-center -> PCA-50.
  3) Stratified sample 600 mal + 600 non across patients.
  4) kNN(k=15) Euclidean graph; Ollivier-Ricci kappa per edge
     (alpha=0.5, lazy walk) via POT ot.emd2 on 4000 sampled edges;
     aggregate to per-cell mean kappa.
  5) Per patient with >=30 cells in BOTH groups: mean kappa each, Welch t,
     Cliff's delta.
  6) MixedLM: kappa ~ malignant + (1|patient_id); report fixed effect, SE,
     95% CI, p, random-effect variance, shrinkage vs naive pooled.

Outputs:
  results.json, e1b_per_patient.csv, e1b_mixed_effects.csv
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import networkx as nx
import ot
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "GSE103322_HNSCC.txt"
SEED = 20260507
RNG = np.random.default_rng(SEED)


# ---------------- Ollivier-Ricci primitives (verbatim from E1) ------------
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


def cliffs_delta_correct(a, b):
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


def stratified_sample(group, patient, n_per_group=600, rng=None):
    """Sample n_per_group from each unique label in `group`, distributing as
    evenly as possible across patients. Returns indices into original arrays.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sel = []
    for lab in np.unique(group):
        idx_lab = np.where(group == lab)[0]
        pats = patient[idx_lab]
        unique_pats = np.unique(pats)
        per_p = n_per_group // len(unique_pats)
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
            extra = rng.choice(remaining, size=min(need, len(remaining)),
                               replace=False)
            chosen = np.concatenate([chosen, extra])
        sel.append(chosen)
    return np.concatenate(sel)


# ---------------- Puram loader (E3 loader + patient parsing) --------------
def load_puram():
    """Puram 2017 HNSCC (GSE103322).

    Header (6 metadata rows):
      row 0: cell barcode (`HN<n>_...` or `HNSCC_<n>_...`; pooled
             `HNSCC_combo1_...` have no patient and are dropped)
      row 1: processed by Maxima enzyme     {0,1}
      row 2: Lymph node                      {0,1}
      row 3: classified as cancer cell       {0,1}
      row 4: classified as non-cancer cells  {0,1}
      row 5: non-cancer cell type            {string}
      rows 6+: gene symbol | log2(TPM/10+1) values

    Returns (Xpca, df, group_pair) matching the E1 loader interface.
    """
    print("[Puram] loading ...")
    header = pd.read_csv(DATA, sep="\t", nrows=6, header=None, low_memory=False)
    cell_ids = header.iloc[0, 1:].values
    cancer = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    noncancer = pd.to_numeric(header.iloc[4, 1:], errors="coerce").values

    def patient_of(cid):
        m = re.match(r"HN(?:SCC)?_?(\d+)_", str(cid))
        return m.group(1) if m else None

    patient = np.array([patient_of(c) for c in cell_ids], dtype=object)

    mask_mal = (cancer == 1)
    mask_non = (noncancer == 1)
    keep = (mask_mal | mask_non) & (patient != None)
    n_dropped = int((~(patient != None)).sum())
    print(f"  total cells: {len(cell_ids)}")
    print(f"  malignant (cancer=1): {int(mask_mal.sum())}")
    print(f"  non-malignant (non-cancer=1): {int(mask_non.sum())}")
    print(f"  dropped (no patient / 'combo1' pool): {n_dropped}")
    print(f"  kept (mal|non with patient): {int(keep.sum())}")

    print("  reading expression matrix ...")
    expr = pd.read_csv(DATA, sep="\t", skiprows=6, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)  # genes x cells

    Xk = X[:, keep]
    var = Xk.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]

    pat = patient[keep]
    grp = np.where(mask_mal[keep], "Malignant", "Non_malignant")
    Xk = Xk[hvg]  # genes x cells (subset)

    rng = np.random.default_rng(SEED)
    sel = stratified_sample(grp, pat, n_per_group=600, rng=rng)

    Xs = Xk[:, sel].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50, random_state=SEED).fit_transform(Xs)

    df = pd.DataFrame({
        "patient_id": pat[sel].astype(str),
        "group": grp[sel],
    })
    df["malignant"] = (df["group"] == "Malignant").astype(int)
    print(f"  sampled: {len(df)}  mal={df.malignant.sum()}  "
          f"non={(1 - df.malignant).sum()}")
    print(f"  patients in sample: {sorted(df.patient_id.unique(), key=int)}")
    return Xpca, df, ("Malignant", "Non_malignant")


# ---------------- analysis (mirrors E1.analyse_dataset) -------------------
def analyse_dataset(name, X, meta, group_pair, k=15, n_edges=4000):
    print(f"\n=== {name} ===")
    print(f"  building kNN (k={k}) ...")
    G = build_knn_graph(X, k=k)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  computing Ollivier-Ricci on {n_edges} sampled edges ...")
    edge_k = ollivier_ricci_edges(G, alpha=0.5, n_edges=n_edges, rng=RNG)
    n_valid = sum(1 for v in edge_k.values() if not np.isnan(v))
    print(f"  valid edges: {n_valid}/{len(edge_k)}")
    per_cell = per_cell_mean_curvature(G, edge_k)
    meta = meta.copy().reset_index(drop=True)
    meta["kappa"] = [per_cell.get(i, np.nan) for i in range(len(meta))]
    valid = meta.dropna(subset=["kappa"]).reset_index(drop=True)
    print(f"  cells with valid kappa: {len(valid)}/{len(meta)}")

    mal_lab, non_lab = group_pair
    # naive pooled effect
    a = valid.loc[valid.group == mal_lab, "kappa"].values
    b = valid.loc[valid.group == non_lab, "kappa"].values
    naive_diff = float(a.mean() - b.mean())
    naive_t, naive_p = ttest_ind(a, b, equal_var=False)
    naive_delta = cliffs_delta_correct(a, b)
    print(f"  NAIVE POOLED  mean(mal)-mean(non) = {naive_diff:+.4f}  "
          f"Welch p={naive_p:.3g}  Cliff d={naive_delta:+.3f}")

    # per-patient
    rows = []
    for pid, sub in valid.groupby("patient_id"):
        a = sub.loc[sub.group == mal_lab, "kappa"].values
        b = sub.loc[sub.group == non_lab, "kappa"].values
        if len(a) < 30 or len(b) < 30:
            continue
        t, p = ttest_ind(a, b, equal_var=False)
        d = cliffs_delta_correct(a, b)
        rows.append({
            "dataset": name,
            "patient_id": pid,
            "n_malignant": int(len(a)),
            "n_non_malignant": int(len(b)),
            "mean_kappa_mal": float(a.mean()),
            "mean_kappa_non": float(b.mean()),
            "diff_mal_minus_non": float(a.mean() - b.mean()),
            "welch_t": float(t),
            "welch_p": float(p),
            "cliff_delta": float(d),
        })
    pp = pd.DataFrame(rows)
    print(f"  per-patient table ({len(pp)} patients pass >=30/30):")
    if len(pp):
        print(pp.to_string(index=False))

    # mixed-effects (lbfgs, matches E1 for direct comparability)
    mm_df = valid[["kappa", "malignant", "patient_id"]].copy()
    mm_df["patient_id"] = mm_df["patient_id"].astype(str)
    md = smf.mixedlm("kappa ~ malignant", mm_df, groups=mm_df["patient_id"])
    mfit = md.fit(method="lbfgs")
    fe_coef = float(mfit.fe_params["malignant"])
    fe_se = float(mfit.bse_fe["malignant"])
    fe_p = float(mfit.pvalues["malignant"])
    ci = mfit.conf_int().loc["malignant"].values
    # random-effect variance (scalar for a (1|patient) random intercept)
    re_var = float(np.asarray(mfit.cov_re).ravel()[0])
    resid_var = float(mfit.scale)
    print(f"  MIXEDLM  malignant fixed effect = {fe_coef:+.4f} +- {fe_se:.4f}  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={fe_p:.3g}")
    print(f"  random-intercept variance = {re_var:.6f}  "
          f"residual variance = {resid_var:.6f}  "
          f"ICC = {re_var / (re_var + resid_var):.3f}")
    shrink_pct = 100.0 * fe_coef / naive_diff if naive_diff != 0 else float("nan")
    print(f"  shrinkage: mixed/naive = {shrink_pct:.1f}%")

    # optimizer robustness: refit with bfgs and powell, report max fe spread.
    # lbfgs emits a 'singular random-effects covariance' trajectory warning
    # on this dataset; confirm the fixed effect is optimizer-invariant.
    fe_alt = []
    for alt_method in ("bfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mfit_alt = smf.mixedlm(
                    "kappa ~ malignant", mm_df, groups=mm_df["patient_id"]
                ).fit(method=alt_method, maxiter=1000, disp=False)
                fe_alt.append(float(mfit_alt.fe_params["malignant"]))
        except Exception as e:
            print(f"  [robustness] {alt_method} failed: {e}")
    fe_spread = (max(fe_alt) - min(fe_alt)) if fe_alt else float("nan")
    print(f"  optimizer robustness: fe over {{bfgs,powell,cg}} = {fe_alt}  "
          f"spread = {fe_spread:.2e} (vs lbfgs fe={fe_coef:+.4f})")

    # save per-cell kappa for reproducibility / secondary analyses
    valid[["kappa", "malignant", "patient_id", "group"]].to_csv(
        HERE / "e1b_per_cell_kappa.csv", index=False
    )

    me_row = {
        "dataset": name,
        "n_cells": int(len(valid)),
        "n_patients_in_model": int(mm_df["patient_id"].nunique()),
        "naive_diff_mal_minus_non": naive_diff,
        "naive_welch_p": float(naive_p),
        "naive_cliff_delta": naive_delta,
        "mixed_fe_coef": fe_coef,
        "mixed_fe_se": fe_se,
        "mixed_fe_ci_lo": float(ci[0]),
        "mixed_fe_ci_hi": float(ci[1]),
        "mixed_fe_p": fe_p,
        "random_intercept_variance": re_var,
        "residual_variance": resid_var,
        "icc": re_var / (re_var + resid_var),
        "shrink_pct_of_naive": float(shrink_pct),
        "n_patients_in_table": int(len(pp)),
        "n_patients_same_dir_as_pooled": int(
            (np.sign(pp["diff_mal_minus_non"]) == np.sign(naive_diff)).sum()
            if len(pp) else 0
        ),
        "optimizer_fe_bfgs_powell_cg": fe_alt,
        "optimizer_fe_spread": float(fe_spread) if fe_alt else None,
    }
    return pp, me_row


def main():
    print("=" * 70)
    print("E1b -- Within-patient mixed-effects Ollivier-Ricci on Puram HNSCC")
    print(f"     seed={SEED}")
    print("=" * 70)

    X, meta, group_pair = load_puram()
    pp, me = analyse_dataset("Puram_HNSCC", X, meta, group_pair)

    # ---- write CSVs ----
    pp_path = HERE / "e1b_per_patient.csv"
    me_path = HERE / "e1b_mixed_effects.csv"
    pp.to_csv(pp_path, index=False)
    pd.DataFrame([me]).to_csv(me_path, index=False)
    print(f"\nWrote: {pp_path}")
    print(f"Wrote: {me_path}")

    # ---- results.json ----
    n_pos = int((pp["cliff_delta"] > 0).sum()) if len(pp) else 0
    n_neg = int((pp["cliff_delta"] < 0).sum()) if len(pp) else 0
    results = {
        "dataset": "Puram_HNSCC_GSE103322",
        "analysis": "within_patient_mixed_effects_E1b",
        "protocol": "PCA-50, kNN k=15, Ollivier-Ricci alpha=0.5, 4000 edges, "
                    "600 mal + 600 non stratified sample, MixedLM kappa~malignant+(1|patient)",
        "seed": SEED,
        "n_cells_with_valid_kappa": me["n_cells"],
        "n_patients_in_mixed_model": me["n_patients_in_model"],
        "n_patients_in_per_patient_table": me["n_patients_in_table"],
        "mixed_effects": {
            "fixed_effect_malignant": me["mixed_fe_coef"],
            "fixed_effect_se": me["mixed_fe_se"],
            "fixed_effect_ci_95": [me["mixed_fe_ci_lo"], me["mixed_fe_ci_hi"]],
            "fixed_effect_p": me["mixed_fe_p"],
            "random_intercept_variance": me["random_intercept_variance"],
            "residual_variance": me["residual_variance"],
            "icc": me["icc"],
            "optimizer_fe_bfgs_powell_cg": me["optimizer_fe_bfgs_powell_cg"],
            "optimizer_fe_spread": me["optimizer_fe_spread"],
        },
        "naive_pooled": {
            "diff_mal_minus_non": me["naive_diff_mal_minus_non"],
            "welch_p": me["naive_welch_p"],
            "cliff_delta": me["naive_cliff_delta"],
        },
        "shrink_pct_of_naive": me["shrink_pct_of_naive"],
        "per_patient_cliffs_delta": {
            r["patient_id"]: {
                "n_malignant": r["n_malignant"],
                "n_non_malignant": r["n_non_malignant"],
                "diff_mal_minus_non": r["diff_mal_minus_non"],
                "welch_p": r["welch_p"],
                "cliff_delta": r["cliff_delta"],
            }
            for _, r in pp.iterrows()
        },
        "per_patient_direction": {
            "n_delta_positive": n_pos,
            "n_delta_negative": n_neg,
            "n_total": int(len(pp)),
            "fraction_positive": n_pos / len(pp) if len(pp) else None,
        },
    }
    json_path = HERE / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote: {json_path}")

    # ---- headline summary ----
    print("\n--- HEADLINE ---")
    ds = me["dataset"]
    npats = me["n_patients_in_table"]
    nsame = me["n_patients_same_dir_as_pooled"]
    frac = nsame / npats if npats else float("nan")
    kill_dir = npats > 0 and frac < 0.5
    kill_shrink = abs(me["shrink_pct_of_naive"]) < 20.0
    verdict = "SIMPSON ARTIFACT" if (kill_dir or kill_shrink) else "SURVIVES"
    print(f"  {ds}:")
    print(f"    MixedLM fixed effect (malignant) = {me['mixed_fe_coef']:+.4f} "
          f"[{me['mixed_fe_ci_lo']:+.4f}, {me['mixed_fe_ci_hi']:+.4f}]  "
          f"p={me['mixed_fe_p']:.3g}")
    print(f"    random-intercept var = {me['random_intercept_variance']:.6f}  "
          f"ICC = {me['icc']:.3f}")
    print(f"    {nsame}/{npats} patients same direction as pooled ({frac:.0%})")
    print(f"    per-patient Cliff d: {n_pos} positive, {n_neg} negative "
          f"(of {npats})")
    print(f"    shrink = mixed/naive = {me['shrink_pct_of_naive']:.1f}%  "
          f"=> {verdict}")
    if me.get("optimizer_fe_spread") is not None:
        print(f"    optimizer robustness: fe spread over "
              f"{{bfgs,powell,cg}} = {me['optimizer_fe_spread']:.2e} "
              f"(values: {[round(x,5) for x in me['optimizer_fe_bfgs_powell_cg']]})")

    print("\n--- Cross-dataset comparison ---")
    print(f"  Tirosh melanoma : fe=+0.148  p=1e-79   (E1)")
    print(f"  Darmanis GBM    : fe=+0.071  p=6e-29   (E1)")
    print(f"  Puram HNSCC     : fe={me['mixed_fe_coef']:+.4f}  "
          f"p={me['mixed_fe_p']:.3g}   (E1b, this run)")


if __name__ == "__main__":
    main()
