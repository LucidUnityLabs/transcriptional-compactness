"""
T3: Bakry-Emery curvature-dimension condition CD(K, N) per cell on a kNN graph
of single-cell RNA-seq data.

Decomposes the Ollivier-Ricci kappa signal into:
  - K = Bakry-Emery curvature (negative => locally hyperbolic / spread)
  - N = effective dimension (smallest N for which CD(K, N) admits K > 0;
        proxy for dimension of accessible state space)

Reference:
  Cushing, Liu, Munch (2020) "Bakry-Emery curvature on graphs as an
  eigenvalue problem", Cal Var PDE.
  Lin, Yau (2010); Munch, Wojciechowski (2019).

For each vertex v on the unweighted symmetric kNN graph we form local
matrices A, B, c such that:
  Gamma(f)(v)   = f^T A f
  Gamma_2(f)(v) = f^T B f
  (Delta f)(v)  = c^T f
Then CD(K, N) at v <=> B - (1/N) c c^T - K A is PSD on the support of A.
The largest such K is the smallest generalized eigenvalue of (B - (1/N) c c^T, A)
on the range of A.  This is computed without an SDP.

The N=infty version is the classical case: K_BE(v; inf) = lambda_min(B, A).
N_eff(v): smallest N in a discrete grid for which K_BE(v; N) > 0.
"""

import sys
import time
import numpy as np
import pandas as pd
import networkx as nx
import ot
from scipy.linalg import eigh
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
DATA = OUT.parent.parent / "data"
SEED = 20260507
RNG = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# kNN graph
# ----------------------------------------------------------------------
def build_knn_graph(X, k=10):
    """Symmetric kNN graph (unweighted in adjacency; weights are Euclidean)."""
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


# ----------------------------------------------------------------------
# Local Bakry-Emery matrices
# ----------------------------------------------------------------------
def local_bakry_emery_matrices(G, v):
    """
    Build matrices A, B and vector c such that for any function f
    supported on B_2(v):
        Gamma(f)(v)  = f^T A f
        Gamma_2(f)(v)= f^T B f
        Delta f (v)  = c^T f
    using the *normalized* graph Laplacian
        Delta f(x) = (1/deg(x)) sum_{y~x} (f(y) - f(x)).

    Returns (A, B, c, nodes) where nodes is the ordered list of vertices in
    B_2(v); the first index (0) is v itself.
    """
    # Build B_2(v) ordering: v, then 1-ring, then strict 2-ring
    nbrs1 = sorted(G.neighbors(v))
    if len(nbrs1) == 0:
        return None
    ring2 = set()
    for u in nbrs1:
        for w in G.neighbors(u):
            if w != v and w not in nbrs1:
                ring2.add(w)
    ring2 = sorted(ring2)
    nodes = [v] + list(nbrs1) + list(ring2)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Construct local Laplacian matrix L acting on f restricted to B_2(v),
    # but only at vertices we can fully evaluate (need to know all neighbors
    # of x to evaluate Delta f(x); we trust 1-ring fully, but 2-ring has
    # neighbors outside B_2(v)).  For the Bakry-Emery local computation we
    # only need Delta f at v and at neighbors of v (1-ring), since:
    #   Gamma(f)(v) = (1/2)[Delta(f^2)(v) - 2 f(v) Delta f(v)]
    #               = (1/(2 deg v)) sum_{u~v} (f(u)-f(v))^2
    #   Gamma_2(f)(v) = (1/2) Delta(Gamma(f,f))(v) - Gamma(f, Delta f)(v)
    # Delta(Gamma(f))(v) = (1/deg v) sum_{u~v} [Gamma(f)(u) - Gamma(f)(v)],
    # which needs Gamma(f)(u) for u in 1-ring -> needs values of f on
    # neighbors of u -> 2-ring.
    # Gamma(f, Delta f)(v) = (1/(2 deg v)) sum_{u~v} (f(u)-f(v))(Delta f(u)-Delta f(v))
    # which needs Delta f(u) -> values of f on neighbors of u in 2-ring.

    # Build (Lf)(v) and (Lf)(u) for u in 1-ring as linear in f.
    # Lrows[x] = vector of length n s.t. Delta f (x) = Lrows[x] . f
    def laplacian_row(x):
        row = np.zeros(n)
        deg_x = G.degree(x)
        if deg_x == 0:
            return row
        row[idx[x]] -= 1.0  # -f(x)
        for y in G.neighbors(x):
            if y in idx:
                row[idx[y]] += 1.0 / deg_x
            else:
                # y is outside B_2(v): we *cannot* evaluate Delta f(x) exactly
                # for such x.  This only matters for x in 1-ring whose
                # neighbors include 3-ring vertices.  We approximate by
                # ignoring those terms (treat f on the cut boundary as the
                # local mean -> contribution vanishes in expectation).
                pass
        # rescale by 1/deg if any neighbors fell in idx:
        # actual normalization: each in-idx neighbor contributed 1/deg_x; -f(x)
        # contributed -1.  We need full sum_{y~x}(f(y)-f(x))/deg_x.
        # Fix: above we already added 1/deg_x per in-idx neighbor and -1 once.
        # The "-f(x)" should be summed once (1/deg_x * deg_x = 1) so that's
        # consistent IF all neighbors were in-idx.  When some are missing,
        # we lose those terms but keep the full -f(x).  To keep things
        # consistent, rebuild correctly:
        row[:] = 0.0
        row[idx[x]] = -1.0
        for y in G.neighbors(x):
            if y in idx:
                row[idx[y]] += 1.0 / deg_x
        return row

    Lv = laplacian_row(v)
    L_nbr = {u: laplacian_row(u) for u in nbrs1}

    # Gamma(f)(v) = (1/(2 deg v)) sum_{u~v} (f(u)-f(v))^2
    deg_v = G.degree(v)
    A = np.zeros((n, n))
    for u in nbrs1:
        e = np.zeros(n)
        e[idx[u]] = 1.0
        e[idx[v]] = -1.0
        A += np.outer(e, e)
    A = A / (2.0 * deg_v)

    # Gamma(f, g)(x) as bilinear: for each x, define G_x s.t.
    # Gamma(f,g)(x) = f^T G_x g (symmetric).
    # Gamma(f)(x) = (1/(2 deg x)) sum_{y~x} (f(y)-f(x))^2  (for x with all
    # neighbors in idx).
    def gamma_quad_at(x):
        if x not in idx:
            return None
        deg_x = G.degree(x)
        # only valid if all neighbors are in idx
        nbrs = list(G.neighbors(x))
        if any(y not in idx for y in nbrs):
            # boundary: skip / return zero (we don't need exact, this only
            # appears when x is in the 2-ring which we don't evaluate for B)
            return None
        M = np.zeros((n, n))
        for y in nbrs:
            e = np.zeros(n)
            e[idx[y]] = 1.0
            e[idx[x]] = -1.0
            M += np.outer(e, e)
        return M / (2.0 * deg_x)

    Gv_quad = A.copy()  # Gamma(f)(v)

    # Delta(Gamma(f))(v) = (1/deg v) sum_{u~v} [Gamma(f)(u) - Gamma(f)(v)]
    # Each Gamma(f)(u) is a quadratic form in f.
    DGamma = np.zeros((n, n))
    for u in nbrs1:
        Gu_quad = gamma_quad_at(u)
        if Gu_quad is None:
            # boundary 1-ring vertex with neighbors outside B_2(v)
            # -> approximate Gamma(f)(u) using only in-idx neighbors
            Gu_quad = np.zeros((n, n))
            deg_u = G.degree(u)
            for y in G.neighbors(u):
                if y in idx:
                    e = np.zeros(n)
                    e[idx[y]] = 1.0
                    e[idx[u]] = -1.0
                    Gu_quad += np.outer(e, e)
            Gu_quad = Gu_quad / (2.0 * deg_u)
        DGamma += (Gu_quad - Gv_quad)
    DGamma = DGamma / deg_v

    # Gamma(f, Delta f)(v) = (1/(2 deg v)) sum_{u~v} (f(u)-f(v))(Delta f(u)-Delta f(v))
    # bilinear form: (1/(2 deg v)) sum_u (e_u - e_v)(f) * (L_nbr[u] - Lv)(f)
    # quadratic part: symmetric outer-product
    GFLF = np.zeros((n, n))
    for u in nbrs1:
        e = np.zeros(n)
        e[idx[u]] = 1.0
        e[idx[v]] = -1.0
        L_diff = L_nbr[u] - Lv
        M = np.outer(e, L_diff)
        GFLF += 0.5 * (M + M.T)
    GFLF = GFLF / (2.0 * deg_v)

    B = 0.5 * DGamma - GFLF
    # symmetrize for numerical safety
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    return A, B, Lv, nodes


def bakry_emery_curvature_at(G, v, N_values):
    """
    For a vertex v, return dict N -> K_BE(v; N).

    CD(K, N) at v iff for all f in R^n (n = |B_2(v)|):
        f^T (B - (1/N) c c^T) f  >=  K f^T A f.
    A is PSD with non-trivial null space; B' := B - (1/N) c c^T is symmetric.
    The largest such K is determined as follows:

      1) If B' restricted to null(A) is not PSD, no K satisfies the
         inequality (CD(K, N) fails everywhere) -> sentinel large negative.
      2) Else, take the Schur complement of B' onto range(A) along null(A);
         K_BE(v; N) is the smallest generalized eigenvalue of (B'_eff, A_RR).

    Implementation note: the constant function 1 is in null(A) AND gives
    Gamma_2 = 0 and Delta f = 0, so it is automatically harmless.
    """
    res = local_bakry_emery_matrices(G, v)
    if res is None:
        return {N: np.nan for N in N_values}
    A, B, c, nodes = res
    n = A.shape[0]

    try:
        eigvals_A, eigvecs_A = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        return {N: np.nan for N in N_values}
    tol = 1e-10 * max(1.0, eigvals_A.max())
    rangeA_mask = eigvals_A > tol
    nullA_mask = ~rangeA_mask
    if not np.any(rangeA_mask):
        return {N: np.nan for N in N_values}
    U_R = eigvecs_A[:, rangeA_mask]
    U_N = eigvecs_A[:, nullA_mask]
    A_RR = U_R.T @ A @ U_R
    try:
        eA, vA = np.linalg.eigh(A_RR)
    except np.linalg.LinAlgError:
        return {N: np.nan for N in N_values}
    A_inv_sqrt = vA @ np.diag(1.0 / np.sqrt(np.maximum(eA, 1e-15))) @ vA.T

    out = {}
    for N in N_values:
        Bp = B if N == np.inf else (B - (1.0 / N) * np.outer(c, c))
        Bp = 0.5 * (Bp + Bp.T)

        if U_N.shape[1] > 0:
            Bp_NN = U_N.T @ Bp @ U_N
            Bp_NN = 0.5 * (Bp_NN + Bp_NN.T)
            try:
                eN_vals, eN_vecs = np.linalg.eigh(Bp_NN)
            except np.linalg.LinAlgError:
                out[N] = np.nan
                continue
            min_eN = eN_vals.min()
            if min_eN < -1e-7:
                out[N] = -1e6   # CD fails for any K
                continue
            Bp_RN = U_R.T @ Bp @ U_N
            # Detect: Bp_RN component along ker(Bp_NN) -> infeasible
            ker_mask = eN_vals < 1e-9
            if np.any(ker_mask):
                BpRN_proj = Bp_RN @ eN_vecs[:, ker_mask]
                if np.linalg.norm(BpRN_proj) > 1e-7:
                    out[N] = -1e6
                    continue
            # Schur complement (use pseudoinverse on PSD Bp_NN)
            Bp_NN_pinv = np.linalg.pinv(Bp_NN, rcond=1e-10, hermitian=True)
            Bp_RR = U_R.T @ Bp @ U_R
            Bp_eff = Bp_RR - Bp_RN @ Bp_NN_pinv @ Bp_RN.T
        else:
            Bp_eff = U_R.T @ Bp @ U_R

        Bp_eff = 0.5 * (Bp_eff + Bp_eff.T)
        M = A_inv_sqrt @ Bp_eff @ A_inv_sqrt
        M = 0.5 * (M + M.T)
        try:
            eigs = np.linalg.eigvalsh(M)
            out[N] = float(eigs.min())
        except np.linalg.LinAlgError:
            out[N] = np.nan
    return out


# ----------------------------------------------------------------------
# Effective dimension
# ----------------------------------------------------------------------
def effective_dimension(K_by_N):
    """
    Continuous effective dimension via the curvature-dimension correction.

    For the optimizer f* of CD(K, N=inf) at vertex v, the correction term
    (1/N)(Delta f*)^2 / Gamma(f*) measures how strongly the dimension
    constraint hurts curvature.  We approximate this by fitting

        K_BE(N) ~ K_BE(inf) - alpha / N

    to the (N, K_BE(N)) values for finite N, and define
        N_eff(v) := alpha / |K_BE(inf) - K_BE(2)| * 2
    Equivalently, the half-saturation N: smallest N* with K(N*) =
    K(2) + 0.5 (K(inf)-K(2)).  We compute it by linear interpolation in
    1/N.

    Returns N_eff in [2, inf).  NaN if curve is flat / non-monotone.
    """
    Ns_fin = sorted([n for n in K_by_N if n != np.inf])
    K_inf = K_by_N[np.inf]
    K_at = [K_by_N[n] for n in Ns_fin]
    if not all(np.isfinite([K_inf, *K_at])):
        return np.nan

    K_lo = K_at[0]                         # K at N=2
    if K_inf - K_lo < 1e-4:
        return 2.0                          # already saturated
    target = K_lo + 0.5 * (K_inf - K_lo)
    inv_Ns = [1.0 / n for n in Ns_fin] + [0.0]
    Ks = K_at + [K_inf]
    # K is monotone non-decreasing in N, i.e., non-increasing in 1/N.  We
    # find the largest 1/N for which K >= target by linear interpolation.
    # Walk from large N (small 1/N) down toward small N (large 1/N).
    Ks_arr = np.array(Ks)
    inv_arr = np.array(inv_Ns)
    order = np.argsort(inv_arr)            # ascending 1/N (descending N)
    inv_arr = inv_arr[order]; Ks_arr = Ks_arr[order]
    # find first crossing where K drops below target as 1/N increases
    for i in range(1, len(Ks_arr)):
        if Ks_arr[i] < target - 1e-9:
            # linear interpolate in 1/N between (inv_arr[i-1], Ks_arr[i-1]) and current
            x0, x1 = inv_arr[i - 1], inv_arr[i]
            y0, y1 = Ks_arr[i - 1], Ks_arr[i]
            if y1 == y0:
                return float(1.0 / x1)
            t = (target - y0) / (y1 - y0)
            inv_target = x0 + t * (x1 - x0)
            if inv_target <= 0:
                return np.inf
            return float(1.0 / inv_target)
    # never crossed -> N_eff = 2 (already at or below target at N=2)
    return float(Ns_fin[0])


# ----------------------------------------------------------------------
# Ollivier-Ricci (sanity check / for correlation)
# ----------------------------------------------------------------------
def ollivier_ricci_per_cell(G, alpha=0.5, n_edges=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    edges = list(G.edges())
    if n_edges is not None and n_edges < len(edges):
        idx = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in idx]
    edge_kappa = {}
    for (u, v) in edges:
        nu = list(G.neighbors(u))
        nv = list(G.neighbors(v))
        sup_u = [u] + nu
        sup_v = [v] + nv
        w_u = np.array([alpha] + [(1 - alpha) / len(nu)] * len(nu))
        w_v = np.array([alpha] + [(1 - alpha) / len(nv)] * len(nv))
        sub_nodes = list(set(sup_u + sup_v))
        sub = G.subgraph(sub_nodes)
        try:
            dists = {n: nx.single_source_dijkstra_path_length(sub, n) for n in sup_u}
        except Exception:
            edge_kappa[(u, v)] = np.nan
            continue
        C = np.zeros((len(sup_u), len(sup_v)))
        ok = True
        for i, a in enumerate(sup_u):
            for j, b in enumerate(sup_v):
                d = dists[a].get(b, np.inf)
                if not np.isfinite(d):
                    ok = False
                C[i, j] = d
        if not ok:
            edge_kappa[(u, v)] = np.nan
            continue
        W = ot.emd2(w_u, w_v, C)
        d_uv = G[u][v]["weight"]
        edge_kappa[(u, v)] = 1.0 - W / d_uv if d_uv > 0 else np.nan

    by_cell = {n: [] for n in G.nodes()}
    for (u, v), k in edge_kappa.items():
        if np.isnan(k):
            continue
        by_cell[u].append(k)
        by_cell[v].append(k)
    return {n: float(np.mean(vs)) if vs else np.nan for n, vs in by_cell.items()}


# ----------------------------------------------------------------------
# Dataset loaders (adapted from path1_hyperbolic/run_ricci.py, n_per=300)
# ----------------------------------------------------------------------
def load_tirosh_subset(n_per=300):
    EXPR = DATA / "GSE72056_melanoma.txt"
    print("  loading Tirosh ...")
    header = pd.read_csv(EXPR, sep="\t", nrows=4, header=None, low_memory=False)
    malignant = pd.to_numeric(header.iloc[2, 1:], errors="coerce").values
    nonmal_type = pd.to_numeric(header.iloc[3, 1:], errors="coerce").values
    expr = pd.read_csv(EXPR, sep="\t", skiprows=4, header=None, low_memory=False)
    X = expr.iloc[:, 1:].values.astype(np.float32)

    mask_mal = (malignant == 2)
    mask_T = (malignant == 1) & (nonmal_type == 1)

    var = X.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    X = X[hvg]

    rng = np.random.default_rng(0)
    idx_mal = rng.choice(np.where(mask_mal)[0], size=min(n_per, mask_mal.sum()), replace=False)
    idx_T = rng.choice(np.where(mask_T)[0], size=min(n_per, mask_T.sum()), replace=False)
    cells_idx = np.concatenate([idx_mal, idx_T])
    labels = np.array(["Malignant"] * len(idx_mal) + ["T cells"] * len(idx_T))

    Xs = X[:, cells_idx].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50).fit_transform(Xs)
    print(f"  Tirosh subset: {Xpca.shape[0]} cells, "
          f"{(labels=='Malignant').sum()} mal / {(labels=='T cells').sum()} T")
    return Xpca, labels


def load_darmanis_subset(n_per=300):
    import re
    EXPR = DATA / "GSE84465_GBM.csv"
    META = DATA / "GSE84465_meta.txt"
    print("  loading Darmanis ...")
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

    df = pd.read_csv(EXPR, sep=r"\s+", header=0, index_col=0, engine="c")
    df.columns = [c.strip('"') for c in df.columns]
    meta_idx = meta.set_index("cell_id").loc[df.columns]
    cell_type = meta_idx["cell type"].values

    Xlog = np.log1p(df.values.astype(np.float32))
    var = Xlog.var(axis=1)
    hvg = np.argsort(var)[::-1][:2000]
    Xlog = Xlog[hvg]

    mask_neo = (cell_type == "Neoplastic")
    mask_imm = (cell_type == "Immune cell")
    rng = np.random.default_rng(0)
    idx_neo = rng.choice(np.where(mask_neo)[0], size=min(n_per, mask_neo.sum()), replace=False)
    idx_imm = rng.choice(np.where(mask_imm)[0], size=min(n_per, mask_imm.sum()), replace=False)
    cells_idx = np.concatenate([idx_neo, idx_imm])
    labels = np.array(["Neoplastic"] * len(idx_neo) + ["Immune"] * len(idx_imm))

    Xs = Xlog[:, cells_idx].T.astype(np.float32)
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xpca = PCA(n_components=50).fit_transform(Xs)
    print(f"  Darmanis subset: {Xpca.shape[0]} cells, "
          f"{(labels=='Neoplastic').sum()} neo / {(labels=='Immune').sum()} imm")
    return Xpca, labels


# ----------------------------------------------------------------------
# Analyze a dataset
# ----------------------------------------------------------------------
def analyze(name, X, labels, k=10, n_or_edges=4000):
    print(f"\n[{name}]  building kNN graph (k={k})...")
    G = build_knn_graph(X, k=k)
    n = G.number_of_nodes()
    print(f"  graph: {n} nodes, {G.number_of_edges()} edges; "
          f"avg deg {2*G.number_of_edges()/n:.2f}")

    N_GRID = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0, 100.0, np.inf]
    K_inf = np.full(n, np.nan)
    K_2 = np.full(n, np.nan)
    K_5 = np.full(n, np.nan)
    K_gap = np.full(n, np.nan)
    N_eff = np.full(n, np.nan)
    print("  computing Bakry-Emery curvature per cell ...")
    t0 = time.time()
    for v in range(n):
        K_by_N = bakry_emery_curvature_at(G, v, N_GRID)
        K_inf[v] = K_by_N[np.inf]
        K_2[v] = K_by_N[2.0]
        K_5[v] = K_by_N[5.0]
        K_gap[v] = K_by_N[np.inf] - K_by_N[2.0]
        N_eff[v] = effective_dimension(K_by_N)
        if (v + 1) % 100 == 0:
            print(f"    {v+1}/{n}  ({time.time()-t0:.1f}s)")
    print(f"  BE done in {time.time()-t0:.1f}s")

    print(f"  computing Ollivier-Ricci on {n_or_edges} sampled edges ...")
    OR = ollivier_ricci_per_cell(G, alpha=0.5, n_edges=n_or_edges, rng=RNG)
    OR_arr = np.array([OR[i] for i in range(n)])

    df = pd.DataFrame({
        "cell_id": np.arange(n),
        "label": labels,
        "K_BE_inf": K_inf,
        "K_BE_2": K_2,
        "K_BE_5": K_5,
        "K_BE_gap": K_gap,    # K(inf) - K(2): "dimension penalty"
        "N_eff": N_eff,
        "OR_kappa": OR_arr,
    })
    df["dataset"] = name
    return df, G


def group_stats(df, key, group_a, group_b):
    a = df.loc[df["label"] == group_a, key].dropna().values
    b = df.loc[df["label"] == group_b, key].dropna().values
    # N_eff might be inf — convert to large finite for tests
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        a = np.where(np.isfinite(a), a, 1e9)
        b = np.where(np.isfinite(b), b, 1e9)
    if len(a) < 2 or len(b) < 2:
        return None
    t, pt = ttest_ind(a, b, equal_var=False)
    u, pu = mannwhitneyu(a, b)
    return dict(mean_a=float(np.mean(a)), std_a=float(np.std(a)), n_a=len(a),
                mean_b=float(np.mean(b)), std_b=float(np.std(b)), n_b=len(b),
                t=float(t), p_welch=float(pt), U=float(u), p_mwu=float(pu))


def main():
    print("=" * 60)
    print("T3: Bakry-Emery CD(K, N) per cell — decomposition of OR signal")
    print(f"seed={SEED}")
    print("=" * 60)

    Xt, lt = load_tirosh_subset(n_per=300)
    df_t, Gt = analyze("Tirosh melanoma", Xt, lt, k=10)

    Xd, ld = load_darmanis_subset(n_per=300)
    df_d, Gd = analyze("Darmanis GBM", Xd, ld, k=10)

    df = pd.concat([df_t, df_d], ignore_index=True)
    df.to_csv(OUT / "t3_be_per_cell.csv", index=False)
    print(f"\nSaved {OUT / 't3_be_per_cell.csv'}  ({len(df)} rows)")

    # -------- summary stats --------
    print("\n" + "=" * 60)
    print("Per-group summary")
    print("=" * 60)
    summary_rows = []
    for name, sub in [("Tirosh melanoma", df_t), ("Darmanis GBM", df_d)]:
        groups = list(sub["label"].unique())
        ga, gb = groups[0], groups[1]  # malignant first by construction
        for key in ["OR_kappa", "K_BE_inf", "K_BE_2", "K_BE_gap", "N_eff"]:
            s = group_stats(sub, key, ga, gb)
            if s is None:
                continue
            print(f"\n[{name}] {key}: {ga} vs {gb}")
            print(f"  {ga:14s}  mean={s['mean_a']:.4g}  std={s['std_a']:.4g}  n={s['n_a']}")
            print(f"  {gb:14s}  mean={s['mean_b']:.4g}  std={s['std_b']:.4g}  n={s['n_b']}")
            print(f"  Welch t={s['t']:+.3f}  p={s['p_welch']:.3g}    "
                  f"MWU U={s['U']:.1f}  p={s['p_mwu']:.3g}")
            summary_rows.append({"dataset": name, "metric": key,
                                 "group_a": ga, "mean_a": s["mean_a"], "n_a": s["n_a"],
                                 "group_b": gb, "mean_b": s["mean_b"], "n_b": s["n_b"],
                                 "t_welch": s["t"], "p_welch": s["p_welch"],
                                 "p_mwu": s["p_mwu"]})
        # correlation OR vs K_BE
        ok = sub[["OR_kappa", "K_BE_inf"]].dropna()
        if len(ok) > 5:
            r_p, p_p = pearsonr(ok["OR_kappa"], ok["K_BE_inf"])
            r_s, p_s = spearmanr(ok["OR_kappa"], ok["K_BE_inf"])
            print(f"\n[{name}] corr(OR, K_BE_inf): "
                  f"Pearson r={r_p:+.3f} p={p_p:.3g}, "
                  f"Spearman rho={r_s:+.3f} p={p_s:.3g}")
            summary_rows.append({"dataset": name, "metric": "corr_OR_KBE",
                                 "group_a": "Pearson_r", "mean_a": r_p,
                                 "group_b": "Spearman_rho", "mean_b": r_s,
                                 "p_welch": p_p, "p_mwu": p_s, "n_a": len(ok), "n_b": len(ok)})
    pd.DataFrame(summary_rows).to_csv(OUT / "t3_be_summary.csv", index=False)

    # -------- distribution plot --------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (name, sub) in enumerate([("Tirosh melanoma", df_t), ("Darmanis GBM", df_d)]):
        groups = list(sub["label"].unique())
        ga, gb = groups[0], groups[1]
        for col, key in enumerate(["OR_kappa", "K_BE_inf", "N_eff"]):
            ax = axes[row, col]
            a = sub.loc[sub["label"] == ga, key].dropna().values
            b = sub.loc[sub["label"] == gb, key].dropna().values
            if key == "N_eff":
                # show finite values; plot on log scale
                a_f = a[np.isfinite(a)]
                b_f = b[np.isfinite(b)]
                if len(a_f) and len(b_f):
                    bins = np.logspace(np.log10(max(min(a_f.min(), b_f.min()), 0.5)),
                                       np.log10(max(a_f.max(), b_f.max(), 2)),
                                       30)
                    ax.hist(a_f, bins=bins, alpha=0.55, label=ga, density=True)
                    ax.hist(b_f, bins=bins, alpha=0.55, label=gb, density=True)
                    ax.set_xscale("log")
            else:
                ax.hist(a, bins=30, alpha=0.55, label=ga, density=True)
                ax.hist(b, bins=30, alpha=0.55, label=gb, density=True)
                ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(key)
            ax.set_ylabel("density")
            ax.set_title(f"{name}: {key}")
            ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "t3_be_distributions.png", dpi=120)
    plt.close()
    print(f"\nSaved {OUT / 't3_be_distributions.png'}")

    # -------- correlation scatter --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, sub) in zip(axes, [("Tirosh melanoma", df_t), ("Darmanis GBM", df_d)]):
        groups = list(sub["label"].unique())
        for g, color in zip(groups, ["C3", "C0"]):
            s = sub[sub["label"] == g].dropna(subset=["OR_kappa", "K_BE_inf"])
            ax.scatter(s["OR_kappa"], s["K_BE_inf"], s=8, alpha=0.5,
                       label=g, color=color)
        ax.set_xlabel("Ollivier-Ricci kappa (per cell mean)")
        ax.set_ylabel("Bakry-Emery K (N=inf)")
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_title(name)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "t3_be_correlation.png", dpi=120)
    plt.close()
    print(f"Saved {OUT / 't3_be_correlation.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
