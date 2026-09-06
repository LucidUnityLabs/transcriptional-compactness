"""
E1 — Within-patient stratified analysis of Ollivier-Ricci curvature.

Tests whether the malignant-vs-immune (Tirosh: malignant vs T-cells; Darmanis:
neoplastic vs immune) per-cell kappa difference is a within-patient effect or
a Simpson's-paradox cross-patient artifact.

Pipeline (per dataset):
  1) Load expression + per-cell patient/group labels.
  2) HVG-2000 -> mean-center -> PCA-50.
  3) Sample 600 mal + 600 nonmal stratified by patient where possible.
  4) Build kNN(k=15) Euclidean graph; compute Ollivier-Ricci kappa per edge
     with lazy walk alpha=0.5; aggregate to per-cell mean kappa.
  5) Per patient with >=30 cells in BOTH groups: mean kappa each, Welch t,
     Cliff's delta.
  6) MixedLM:  kappa ~ malignant + (1|patient_id), report fixed effect, SE,
     95% CI, p; compare to naive pooled (mean_kappa_mal - mean_kappa_imm)
     ignoring patient.

Outputs (in this directory):
  e1_per_patient.csv, e1_mixed_effects.csv, e1_summary.png
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
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
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


# ---------------- effect-size helpers ----------------
def cliffs_delta_correct(a, b):
    """Cliff's delta = P(a>b) - P(a<b). Positive => a stochastically larger."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    B = np.sort(b)
    # for each ai: #{b<ai}, #{b<=ai}
    n_less = np.searchsorted(B, a, side="left")          # b < a
    n_leq = np.searchsorted(B, a, side="right")          # b <= a
    n_greater = len(B) - n_leq                           # b > a
    gt = n_less.sum()    # pairs where a>b
    lt = n_greater.sum()  # pairs where a<b
    return float((gt - lt) / (len(a) * len(b)))


# ---------------- dataset loaders -> (X_pca, group, patient, idx_mal, idx_non) ----------------
def load_tirosh():
    """Tirosh 2016 melanoma. malignant=2, T cells = malignant==1 & nonmal_type==1.
    Patient ID is row 2 (1-19). Returns dataframe with cell metadata + Xpca.
    """
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("[Tirosh] loading ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    patient = pd.to_numeric(header.iloc[1, 1:], errors="coerce").values
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)  # genes x cells

    mask_mal = (malignant == 2)
    mask_T = (malignant == 1) & (nonmal_type == 1)
    keep = mask_mal | mask_T
    print(f"  total cells malignant={mask_mal.sum()}  T={mask_T.sum()}")

    # HVG over all kept cells (so subset doesn't bias selection)
    Xk = X[:, keep]
    var = Xk.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]

    pat = patient[keep]
    grp = np.where(mask_mal[keep], "Malignant", "T_cells")
    Xk = Xk[hvg]  # genes x cells (subset)

    # stratified sample: 600 + 600
    rng = np.random.default_rng(SEED)
    sel = stratified_sample(grp, pat, n_per_group=600, rng=rng)

    Xs = Xk[:, sel].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50, random_state=SEED).fit_transform(Xs)

    df = pd.DataFrame({
        "patient_id": pat[sel].astype(int),
        "group": grp[sel],
    })
    df["malignant"] = (df["group"] == "Malignant").astype(int)
    print(f"  sampled: {len(df)}  mal={df.malignant.sum()}  T={(1-df.malignant).sum()}")
    print(f"  patients in sample: {sorted(df.patient_id.unique())}")
    return Xpca, df, ("Malignant", "T_cells")


def load_darmanis():
    """Darmanis 2017 GBM. Neoplastic vs Immune cell. patient id from meta."""
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
    print(f"  total cells neoplastic={mask_neo.sum()}  immune={mask_imm.sum()}")

    Xk = Xlog[:, keep]
    var = Xk.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xk = Xk[hvg]

    pat = patient[keep]
    grp = np.where(mask_neo[keep], "Neoplastic", "Immune")

    rng = np.random.default_rng(SEED)
    sel = stratified_sample(grp, pat, n_per_group=600, rng=rng)

    Xs = Xk[:, sel].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50, random_state=SEED).fit_transform(Xs)

    df = pd.DataFrame({
        "patient_id": pat[sel],
        "group": grp[sel],
    })
    df["malignant"] = (df["group"] == "Neoplastic").astype(int)
    print(f"  sampled: {len(df)}  neo={df.malignant.sum()}  imm={(1-df.malignant).sum()}")
    print(f"  patients in sample: {sorted(df.patient_id.unique())}")
    return Xpca, df, ("Neoplastic", "Immune")


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
        # ideal share per patient
        per_p = n_per_group // len(unique_pats)
        chosen = []
        for p in unique_pats:
            ids = idx_lab[pats == p]
            take = min(len(ids), per_p)
            if take > 0:
                chosen.append(rng.choice(ids, size=take, replace=False))
        chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        # top up by random sampling without replacement from remaining
        remaining = np.setdiff1d(idx_lab, chosen)
        need = n_per_group - len(chosen)
        if need > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=min(need, len(remaining)),
                               replace=False)
            chosen = np.concatenate([chosen, extra])
        sel.append(chosen)
    return np.concatenate(sel)


# ---------------- analysis ----------------
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
            "n_immune": int(len(b)),
            "mean_kappa_mal": float(a.mean()),
            "mean_kappa_imm": float(b.mean()),
            "diff_mal_minus_imm": float(a.mean() - b.mean()),
            "welch_t": float(t),
            "welch_p": float(p),
            "cliff_delta": float(d),
        })
    pp = pd.DataFrame(rows)
    print(f"  per-patient table ({len(pp)} patients pass >=30/30):")
    if len(pp):
        print(pp.to_string(index=False))

    # mixed-effects
    mm_df = valid[["kappa", "malignant", "patient_id"]].copy()
    mm_df["patient_id"] = mm_df["patient_id"].astype(str)
    md = smf.mixedlm("kappa ~ malignant", mm_df, groups=mm_df["patient_id"])
    mfit = md.fit(method="lbfgs")
    fe_coef = float(mfit.fe_params["malignant"])
    fe_se = float(mfit.bse_fe["malignant"])
    fe_p = float(mfit.pvalues["malignant"])
    ci = mfit.conf_int().loc["malignant"].values
    print(f"  MIXEDLM  malignant fixed effect = {fe_coef:+.4f} +- {fe_se:.4f}  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={fe_p:.3g}")
    shrink_pct = 100.0 * fe_coef / naive_diff if naive_diff != 0 else np.nan
    print(f"  shrinkage: mixed/naive = {shrink_pct:.1f}%")

    me_row = {
        "dataset": name,
        "naive_diff_mal_minus_imm": naive_diff,
        "naive_welch_p": float(naive_p),
        "naive_cliff_delta": naive_delta,
        "mixed_fe_coef": fe_coef,
        "mixed_fe_se": fe_se,
        "mixed_fe_ci_lo": float(ci[0]),
        "mixed_fe_ci_hi": float(ci[1]),
        "mixed_fe_p": fe_p,
        "shrink_pct_of_naive": float(shrink_pct),
        "n_patients_in_table": int(len(pp)),
        "n_patients_same_dir_as_pooled": int(
            (np.sign(pp["diff_mal_minus_imm"]) == np.sign(naive_diff)).sum()
            if len(pp) else 0
        ),
    }
    return pp, me_row


def make_forest(per_patient_all, mixed_all, out_png):
    """Forest plot: per-patient Cliff's delta with 95% bootstrap CI from
    welch t-distribution proxy; pooled mixed-effects shown as a vertical band
    (we plot the standardized fixed effect alongside)."""
    datasets = sorted(per_patient_all["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 0.45 * max(
        len(per_patient_all[per_patient_all.dataset == d]) for d in datasets
    ) + 3), squeeze=False)
    for ax_i, ds in enumerate(datasets):
        ax = axes[0, ax_i]
        sub = per_patient_all[per_patient_all.dataset == ds].sort_values(
            "cliff_delta"
        ).reset_index(drop=True)
        y = np.arange(len(sub))
        ax.scatter(sub["cliff_delta"], y, color="C0", zorder=3,
                   s=40 + np.minimum(sub["n_malignant"] + sub["n_immune"], 600) / 5)
        for _, r in sub.iterrows():
            yy = sub.index[sub["patient_id"] == r["patient_id"]][0]
            # rough 95% CI on Cliff's delta via normal approx, sd ~ 1/sqrt(n)
            n_pair = r["n_malignant"] * r["n_immune"]
            se = (1 - r["cliff_delta"] ** 2) / np.sqrt(n_pair) * np.sqrt(
                r["n_malignant"] + r["n_immune"]
            )
            ax.hlines(yy, r["cliff_delta"] - 1.96 * se,
                      r["cliff_delta"] + 1.96 * se, color="C0", lw=1.5)
        # mixed-effects pooled (transform fe coef to a comparable Cliff's d
        # is non-trivial; we show the naive Cliff and a vertical band for fe)
        m_row = mixed_all.loc[mixed_all.dataset == ds].iloc[0]
        ax.axvline(m_row["naive_cliff_delta"], color="red", ls="--",
                   label=f"naive Cliff d={m_row['naive_cliff_delta']:+.2f}")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"P{p}" for p in sub["patient_id"]])
        ax.set_xlabel("Cliff's delta (malignant - non-malignant)")
        ax.set_title(
            f"{ds}\nMixedLM fe={m_row['mixed_fe_coef']:+.4f} "
            f"[{m_row['mixed_fe_ci_lo']:+.4f},{m_row['mixed_fe_ci_hi']:+.4f}] "
            f"shrink={m_row['shrink_pct_of_naive']:.0f}% of naive\n"
            f"naive diff={m_row['naive_diff_mal_minus_imm']:+.4f}"
        )
        ax.legend(loc="lower right")
        ax.set_xlim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()


def main():
    print("=" * 70)
    print("E1 — Within-patient stratified Ollivier-Ricci analysis")
    print(f"     seed={SEED}")
    print("=" * 70)

    all_pp = []
    all_me = []

    Xt, mt, gpt = load_tirosh()
    pp_t, me_t = analyse_dataset("Tirosh_melanoma", Xt, mt, gpt)
    all_pp.append(pp_t)
    all_me.append(me_t)

    Xd, md, gpd = load_darmanis()
    pp_d, me_d = analyse_dataset("Darmanis_GBM", Xd, md, gpd)
    all_pp.append(pp_d)
    all_me.append(me_d)

    pp_all = pd.concat(all_pp, ignore_index=True)
    me_all = pd.DataFrame(all_me)

    pp_path = HERE / "e1_per_patient.csv"
    me_path = HERE / "e1_mixed_effects.csv"
    png_path = HERE / "e1_summary.png"
    pp_all.to_csv(pp_path, index=False)
    me_all.to_csv(me_path, index=False)
    make_forest(pp_all, me_all, png_path)
    print(f"\nWrote: {pp_path}")
    print(f"Wrote: {me_path}")
    print(f"Wrote: {png_path}")

    # Headline summary
    print("\n--- HEADLINE ---")
    for _, row in me_all.iterrows():
        ds = row["dataset"]
        npats = row["n_patients_in_table"]
        nsame = row["n_patients_same_dir_as_pooled"]
        frac = nsame / npats if npats else float("nan")
        kill_dir = npats > 0 and frac < 0.5
        kill_shrink = abs(row["shrink_pct_of_naive"]) < 20.0
        verdict = "SIMPSON ARTIFACT" if (kill_dir or kill_shrink) else "SURVIVES"
        print(f"  {ds}: {nsame}/{npats} patients same direction "
              f"({frac:.0%}); shrink={row['shrink_pct_of_naive']:.0f}%  "
              f"=> {verdict}")


if __name__ == "__main__":
    main()
