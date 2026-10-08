"""Corrected Ollivier-Ricci numerics (method version TC-1).

Fixes relative to the historical per-experiment implementations:

* Support-to-support distances are computed on the FULL graph, not on the
  induced support subgraph (audit C01).  The legacy restricted-subgraph
  metric can only overestimate transport costs on the affected edges: on
  the unit five-cycle it reports curvature 0 where the full-graph metric
  gives 0.25.
* kNN graphs are built identity-safely (audit C02): the self is removed by
  row identity, equal-distance candidates are tie-broken by a stable cell
  id, ``1 <= k < n`` is validated, and coincident coordinates are rejected
  instead of silently producing self-loops or zero-length edges.
* Edge lengths must be positive and finite; curvature is computed from the
  dimensionless cost ``C / d_G(u, v)`` so a global rescaling of edge
  lengths cannot change which edges are eligible (audit C03).  No absolute
  distance cutoffs.  Curvature is not clipped below at -1.
* Optimal transport is solved through :func:`transport`, which validates
  masses, checks solver status, plan nonnegativity, marginals and the
  objective gap, and raises on failure instead of emitting NaN (audit C04).
* Per-cell summaries retain degree, incident measured-edge counts and
  coverage, distinguish "not sampled" from "failed", and reject nonfinite
  measured values (audit C05).
* Cliff's delta placement variances are computed exactly via sorted
  searches; degenerate configurations return a non-estimable status rather
  than a fabricated 1e-6 standard error (audit C06).
* Meta-analysis helpers validate their inputs and never silently filter
  rows (audit C07); tiny tail probabilities are additionally stored in log
  space.
"""

from __future__ import annotations

import warnings
import hashlib
from pathlib import Path

IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
from dataclasses import dataclass, field
from math import lgamma

import numpy as np
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm
from scipy.stats import t as _t

#: Tolerance used for floating-point mass/marginal checks, scaled by the
#: cost magnitude at every use site (tiny mass errors are not harmless on
#: ill-conditioned costs).
MASS_TOL = 1e-9


class NumericsError(RuntimeError):
    """Fail-closed numerical error (never converted to NaN downstream)."""


class TransportError(NumericsError):
    """Optimal-transport solve failed validation."""


class DuplicateCoordinatesError(NumericsError):
    """kNN construction encountered coincident rows (zero-distance cells).

    The continuous-metric curvature control is not estimable on coincident
    coordinates: dropping zero-length edges silently changes the object
    under study.  Callers must report the affected analysis as
    non-estimable (with the duplicate count) rather than fabricate a
    comparison.  A mathematically specified zero-distance quotient with
    multiplicity-aware measures would be a NEW method version.
    """


# ---------------------------------------------------------------------------
# kNN graph construction (C02)
# ---------------------------------------------------------------------------
def knn_graph(X, k=15, block=512):
    """Exact identity-safe k-nearest-neighbour graph.

    * ``X`` is converted to a validated 2-D float64 array.
    * ``1 <= k < n`` is enforced.
    * The query row itself is removed by row identity (not by neighbour
      position), equal-distance candidates are tie-broken by ascending cell
      index (stable cell id), and the construction is independent of the
      neighbour ordering returned by any backend.
    * Coincident rows (pairwise distance exactly 0) raise
      :class:`DuplicateCoordinatesError`.

    Returns a ``networkx.Graph`` with node set ``range(n)`` and positive
    float ``weight`` attributes (the minimum observed candidate distance).
    """
    import networkx as nx

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n = X.shape[0]
    if not isinstance(k, (int, np.integer)) or not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < n (n={n}), got k={k!r}")

    # Exact blockwise distances with duplicate detection.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    n_dup = 0
    dup_seen = set()
    for start in range(0, n, block):
        stop = min(start + block, n)
        D = _pairwise_distances(X[start:stop], X)
        for r in range(start, stop):
            d_row = D[r - start]
            zero = np.flatnonzero(d_row == 0.0)
            zero = zero[zero != r]
            if zero.size:
                n_dup += int(zero.size)
                dup_seen.update(int(z) for z in zero)
                dup_seen.add(r)
            # Stable ordering: sort by (distance, index); drop self by id.
            order = np.lexsort((np.arange(n), d_row))
            neighbours = [int(j) for j in order if j != r][:k]
            for j in neighbours:
                w = float(d_row[j])
                if not np.isfinite(w) or w <= 0.0:
                    # Coincident rows raise below; negative/inf impossible
                    # for squared-Euclidean but guarded anyway.
                    if w == 0.0:
                        continue
                    raise NumericsError(
                        f"nonfinite/nonpositive distance {w} between cells "
                        f"{r} and {j}")
                if not G.has_edge(r, j) or G[r][j]["weight"] > w:
                    G.add_edge(r, j, weight=w)
    if n_dup:
        raise DuplicateCoordinatesError(
            f"{len(dup_seen)} of {n} cells share an exact coordinate with "
            f"another cell ({n_dup} zero-distance pairs); the "
            "continuous-metric control is not estimable — define a "
            "multiplicity-aware zero-distance quotient as a new method "
            "version or report this analysis as non-estimable")
    if any(u == v for u, v in G.edges()):
        raise NumericsError("self-loop constructed (impossible after "
                            "identity-safe removal)")
    return G


def _pairwise_distances(A, B):
    """Exact float64 squared-Euclidean pairwise distances."""
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    ab = A @ B.T
    d2 = aa[:, None] + bb[None, :] - 2.0 * ab
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


# ---------------------------------------------------------------------------
# Verified optimal transport (C04)
# ---------------------------------------------------------------------------
@dataclass
class TransportResult:
    cost: float
    plan: np.ndarray
    backend: str
    checks: dict = field(default_factory=dict)


def _solve_lp(w_u, w_v, C):
    """Exact EMD via scipy HiGHS (reference backend, no POT dependency)."""
    from scipy.optimize import linprog

    n, m = C.shape
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([w_u, w_v])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                  method="highs")
    if not res.success:
        raise TransportError(f"scipy LP transport failed: {res.message}")
    return float(res.fun), res.x.reshape(n, m), "scipy-highs"


def _solve_pot(w_u, w_v, C):
    """POT backend with status/warning inspection (mandatory in release CI;
    skipped automatically where POT is not installed)."""
    try:
        import ot
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TransportError("POT not importable") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            value, log = ot.emd2(w_u, w_v, C, log=True)
        except TypeError:
            value = ot.emd2(w_u, w_v, C)
            log = {}
    warn = log.get("warning") if isinstance(log, dict) else None
    if warn:
        raise TransportError(f"POT transport warning: {warn!r}")
    # POT returns the value only; recover the plan for the marginal checks
    # through emd (same solver state).
    plan = ot.emd(w_u, w_v, C)
    return float(value), plan, "POT"


def transport(w_u, w_v, C, backend="auto"):
    """Solve the discrete optimal-transport problem with full validation.

    Validates float64 contiguity, mass nonnegativity/finitess, total-mass
    agreement, solver status, plan nonnegativity, row/column marginals
    (tolerance scaled by the cost magnitude) and the objective gap.  Any
    violation raises :class:`TransportError`; results are never silently
    downgraded to NaN.

    ``backend="auto"`` prefers POT when importable and falls back to an
    exact scipy-HiGHS LP.  The LP fallback computes the same optimum; the
    POT backend additionally exercises the pinned production solver.
    """
    w_u = np.ascontiguousarray(np.asarray(w_u, dtype=np.float64))
    w_v = np.ascontiguousarray(np.asarray(w_v, dtype=np.float64))
    C = np.ascontiguousarray(np.asarray(C, dtype=np.float64))
    if w_u.ndim != 1 or w_v.ndim != 1 or C.ndim != 2:
        raise ValueError("w_u, w_v must be 1-D and C 2-D")
    if C.shape != (w_u.size, w_v.size):
        raise ValueError(f"C shape {C.shape} does not match mass lengths "
                         f"({w_u.size}, {w_v.size})")
    if not (np.all(np.isfinite(w_u)) and np.all(w_u >= 0)
            and np.all(np.isfinite(w_v)) and np.all(w_v >= 0)):
        raise TransportError("masses must be finite and nonnegative")
    if not np.all(np.isfinite(C)):
        raise TransportError("cost matrix must be finite")
    scale = float(np.abs(C).max())
    mass_tol = MASS_TOL * max(1.0, scale)
    su, sv = float(w_u.sum()), float(w_v.sum())
    if abs(su - sv) > mass_tol:
        raise TransportError(f"mass totals differ: {su!r} vs {sv!r}")
    if su == 0.0:
        raise TransportError("zero total mass")

    if backend == "auto":
        try:
            cost, plan, used = _solve_pot(w_u, w_v, C)
        except ImportError:
            cost, plan, used = _solve_lp(w_u, w_v, C)
    elif backend == "pot":
        cost, plan, used = _solve_pot(w_u, w_v, C)
    elif backend == "lp":
        cost, plan, used = _solve_lp(w_u, w_v, C)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    plan = np.ascontiguousarray(np.asarray(plan, dtype=np.float64))
    if not np.all(np.isfinite(plan)):
        raise TransportError("nonfinite transport plan")
    if plan.min() < -mass_tol:
        raise TransportError(f"negative plan entry {plan.min()!r}")
    row_res = np.abs(plan.sum(axis=1) - w_u).max()
    col_res = np.abs(plan.sum(axis=0) - w_v).max()
    if max(row_res, col_res) > mass_tol:
        raise TransportError(
            f"marginal violation (row {row_res:.3e}, col {col_res:.3e}, "
            f"tol {mass_tol:.3e})")
    objective_gap = abs(float(np.sum(plan * C)) - cost)
    if objective_gap > mass_tol * max(1.0, abs(cost)):
        raise TransportError(f"objective gap {objective_gap:.3e}")
    checks = {"row_residual": float(row_res), "col_residual": float(col_res),
              "objective_gap": float(objective_gap), "tol": mass_tol,
              "backend": used}
    return TransportResult(cost=float(cost), plan=plan, backend=used,
                           checks=checks)


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature, full-graph metric (C01, C03)
# ---------------------------------------------------------------------------
def _lazy_masses(G, u, v, alpha):
    nu = list(G.neighbors(u))
    nv = list(G.neighbors(v))
    if not nu or not nv:
        raise NumericsError(
            f"edge ({u}, {v}) has an isolated endpoint; the lazy-walk "
            "measure is undefined")
    w_u = np.array([alpha] + [(1.0 - alpha) / len(nu)] * len(nu),
                   dtype=np.float64)
    w_v = np.array([alpha] + [(1.0 - alpha) / len(nv)] * len(nv),
                   dtype=np.float64)
    return ([u] + nu, [v] + nv, w_u, w_v)


_DISTANCE_GRAPH_KEY = object()


class _NamedDistances:
    def __init__(self, distances, index):
        self.distances = distances
        self.index = index

    def get(self, node, default=None):
        i = self.index.get(node)
        if i is None or not np.isfinite(self.distances[i]):
            return default
        return float(self.distances[i])


def _dijkstra(G, source, cache, cache_limit=4096):
    """Exact full-graph Dijkstra with compact, bounded distance rows.

    A shared named-node crosswalk retains arbitrary graph identities.
    Float64 rows replace millions of Python node/float dictionary entries;
    the sparse graph is built once. This changes storage/backend, never
    restricts the metric to a support neighborhood.
    """
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra
    from collections import OrderedDict
    if _DISTANCE_GRAPH_KEY not in cache:
        nodes = list(G.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", dtype=np.float64, format="csr")
        if matrix.shape[0] > np.iinfo(np.int32).max or matrix.nnz > np.iinfo(np.int32).max:
            raise NumericsError("graph exceeds supported sparse index ABI")
        # csgraph uses the supported C-int index ABI on this SciPy build.
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)
        cache[_DISTANCE_GRAPH_KEY] = (matrix, index, OrderedDict())
    matrix, index, rows = cache[_DISTANCE_GRAPH_KEY]
    if source in rows:
        rows.move_to_end(source)
        return rows[source]
    distances = dijkstra(matrix, directed=False, indices=index[source])
    got = _NamedDistances(distances, index)
    rows[source] = got
    if len(rows) > cache_limit:
        rows.popitem(last=False)
    return got


def ricci_edges(G, alpha=0.5, n_edges=None, rng=None, backend="auto",
                verbose=False, selected_edges=None):
    """Ollivier-Ricci curvature on sampled edges, full-graph metric.

    For each sampled edge ``(u, v)`` the support measures (lazy random walk
    with self mass ``alpha``) are transported with costs given by shortest-
    path distances in the FULL graph ``G`` between support points, and the
    curvature is ``1 - W / d_G(u, v)`` using the same full-graph metric in
    the denominator.

    Edge weights must be positive and finite (validated up front; no
    absolute cutoffs — a global rescaling of all lengths leaves every
    curvature unchanged, unlike the legacy ``d_uv > 1e-6`` variant).  The
    curvature is deliberately NOT clipped at -1: very negative finite
    values are legitimate for this weighted metric; the upper bound is 1
    up to numerical tolerance.

    Returns a dict ``{(u, v): EdgeCurvature}`` with the transport QC.
    Failures raise; NaN results are never emitted.
    """
    for u, v, data in G.edges(data=True):
        w = data.get("weight")
        if w is None or not np.isfinite(w) or w <= 0:
            raise NumericsError(
                f"edge ({u}, {v}) has nonpositive/nonfinite weight {w!r}; "
                "require positive finite lengths (no absolute cutoffs — "
                "rescale-invariance is part of the method contract)")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")

    edges = list(G.edges())
    if selected_edges is not None:
        if n_edges is not None:
            raise ValueError('explicit edges and an edge budget are mutually exclusive')
        edges = list(selected_edges)
        identities = [frozenset(e) for e in edges]
        if len(set(identities)) != len(identities) or any(
                len(e) != 2 or not G.has_edge(*e) for e in edges):
            raise ValueError('explicit edges must be unique actual graph edges')
    if n_edges is not None:
        if (isinstance(n_edges, (bool, np.bool_))
                or not isinstance(n_edges, (int, np.integer))
                or n_edges < 1):
            raise ValueError(f"n_edges must be a positive int, got "
                             f"{n_edges!r}")
        if n_edges < len(edges):
            if rng is None:
                rng = np.random.default_rng(0)
            idx = rng.choice(len(edges), size=int(n_edges), replace=False)
            edges = [edges[i] for i in idx]

    dist_cache: dict = {}
    out = {}
    for ei, (u, v) in enumerate(edges):
        if verbose and (ei + 1) % 1000 == 0:
            print(f"    OR edge {ei + 1}/{len(edges)}", flush=True)
        sup_u, sup_v, w_u, w_v = _lazy_masses(G, u, v, alpha)
        d_uv = G[u][v]["weight"]
        dists_u = _dijkstra(G, u, dist_cache)
        d_G_uv = dists_u.get(v)
        if d_G_uv is None or not np.isfinite(d_G_uv) or d_G_uv <= 0:
            raise NumericsError(f"no positive path between {u} and {v}")
        C = np.empty((len(sup_u), len(sup_v)), dtype=np.float64)
        for i, a in enumerate(sup_u):
            da = _dijkstra(G, a, dist_cache)
            for j, b in enumerate(sup_v):
                dab = da.get(b)
                if dab is None or not np.isfinite(dab):
                    raise NumericsError(
                        f"support points {a} and {b} not connected")
                C[i, j] = dab
        # Dimensionless cost: global rescaling invariance (C03).
        C_n = C / d_G_uv
        res = transport(w_u, w_v, C_n, backend=backend)
        kappa = 1.0 - res.cost
        if not np.isfinite(kappa):
            raise TransportError(f"nonfinite curvature on edge ({u}, {v})")
        out[(u, v)] = EdgeCurvature(
            kappa=float(kappa), d_uv=float(d_uv), d_graph_uv=float(d_G_uv),
            transport_checks=res.checks, backend=res.backend)
    return out


@dataclass
class EdgeCurvature:
    kappa: float
    d_uv: float
    d_graph_uv: float
    transport_checks: dict
    backend: str


# ---------------------------------------------------------------------------
# Coverage-aware per-cell summaries (C05)
# ---------------------------------------------------------------------------
@dataclass
class CellCurvature:
    kappa: float
    degree: int
    n_incident_measured: int
    coverage_fraction: float
    status: str          # "measured" | "not_sampled"


def cell_curvature(G, edge_results):
    """Per-cell mean curvature over MEASURED incident edges, with coverage.

    ``edge_results`` maps ``(u, v) -> EdgeCurvature`` for the sampled edge
    draw.  Each cell record retains ``degree``, ``n_incident_measured``,
    ``coverage_fraction`` and a status that distinguishes ``not_sampled``
    (no incident edge was drawn) from failure (which raises upstream).
    Nonfinite measured values are rejected here as well (belt and braces:
    ``isfinite`` at every measured-value boundary, not just ``isnan``).
    """
    by_cell = {n: [] for n in G.nodes()}
    seen_edges = set()
    for (u, v), ec in edge_results.items():
        canon = (min(u, v), max(u, v))
        if canon in seen_edges:
            raise NumericsError(
                f"edge ({u}, {v}) recorded twice (also as its reverse) — "
                "duplicate measured-edge records would double-count every "
                "incident summary")
        seen_edges.add(canon)
        k = float(ec.kappa)
        if not np.isfinite(k):
            raise NumericsError(
                f"nonfinite measured curvature {k!r} on edge ({u}, {v})")
        by_cell[u].append(k)
        by_cell[v].append(k)
    out = {}
    for n in G.nodes():
        vals = by_cell[n]
        deg = G.degree(n)
        if vals:
            out[n] = CellCurvature(
                kappa=float(np.mean(vals)), degree=int(deg),
                n_incident_measured=len(vals),
                coverage_fraction=len(vals) / deg if deg else 0.0,
                status="measured")
        else:
            out[n] = CellCurvature(
                kappa=float("nan"), degree=int(deg),
                n_incident_measured=0, coverage_fraction=0.0,
                status="not_sampled")
    return out


# ---------------------------------------------------------------------------
# Cliff's delta with exact placement variances (C06)
# ---------------------------------------------------------------------------
@dataclass
class CliffResult:
    delta: float
    se: float | None
    estimable: bool
    status: str
    n1: int
    n2: int
    zeta10: float
    zeta01: float

    def as_dict(self):
        return {"delta": self.delta, "se": self.se,
                "se_estimable": bool(self.estimable), "status": self.status,
                "n1": self.n1, "n2": self.n2,
                "zeta10": self.zeta10, "zeta01": self.zeta01}


def cliff_placements(a, b):
    """Cliff's delta and U-statistic (Hoeffding projection) SE, exactly.

    ``g1(x) = E[sign(x - Y)]`` is computed for every ``a`` by a sorted
    search over ``b`` (no n1 x n2 dominance matrix), likewise ``g2``; the
    placement variances are their sample variances.  This is a first-order
    independence approximation, NOT a correction for graph-dependent cell
    measurements — downstream reporting must label it as such.

    Degenerate configurations (complete separation, all ties, n < 2) yield
    ``se=None`` with an explanatory status: substituting a tiny floor SE
    would fabricate inverse-variance precision (the legacy ``max(var,
    1e-12)`` gave SE 1e-6, i.e. nominal weight 1e12).  A point effect
    remains reportable descriptively.
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("cliff_placements inputs must be finite")
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        raise ValueError("empty group")

    bs = np.sort(b)
    as_ = np.sort(a)
    # #{b < x} and #{b > x} per a-value
    less_b = np.searchsorted(bs, a, side="left")
    greater_b = np.searchsorted(bs, a, side="right")
    g1 = (less_b - (n2 - greater_b)) / n2
    less_a = np.searchsorted(as_, b, side="left")
    greater_a = np.searchsorted(as_, b, side="right")
    g2 = (less_a - (n1 - greater_a)) / n1
    # delta = mean of the dominance matrix == mean g1 == -mean g2
    delta = float(np.mean(g1))
    if abs(delta + float(np.mean(g2))) > 1e-12:
        raise NumericsError("placement means disagree (internal error)")

    if n1 < 2 or n2 < 2:
        return CliffResult(delta, None, False,
                           "non_estimable_small_group", n1, n2,
                           float("nan"), float("nan"))
    z10 = float(np.var(g1, ddof=1))
    z01 = float(np.var(g2, ddof=1))
    if z10 <= 0.0 and z01 <= 0.0:
        status = ("non_estimable_degenerate_projection"
                  if abs(delta) >= 1.0 or delta == 0.0 else
                  "non_estimable_zero_projection_variance")
        return CliffResult(delta, None, False, status, n1, n2, z10, z01)
    var = z10 / n1 + z01 / n2
    if var <= 0.0:
        return CliffResult(delta, None, False,
                           "non_estimable_zero_projection_variance",
                           n1, n2, z10, z01)
    return CliffResult(delta, float(np.sqrt(var)), True, "ok",
                       n1, n2, z10, z01)


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis (C07)
# ---------------------------------------------------------------------------
def _logsf_abs(z):
    return float(_norm.logsf(abs(z))) + float(np.log(2.0))


def dl_meta(deltas, ses, label="", ci_critical=1.96):
    """Validating DerSimonian-Laird random-effects meta-analysis.

    * ``deltas``/``ses`` must be equal-length finite 1-D arrays with
      ``delta in [-1, 1]`` (Cliff's delta range) and strictly positive SE.
      Empty input and any invalid entry raise ``ValueError`` — rows are
      never silently filtered (filtering changes the pooled population).
    * Weight arithmetic is accumulated in extended precision and the
      heterogeneity ``Q`` uses the pair-product form
      ``Q = sum_{i<j} w_i w_j (d_i - d_j)^2 / sum w`` which avoids the
      cancellation of the squared-deviation form.
    * ``ci_critical=1.96`` is retained deliberately as the LEGACY critical
      value for arithmetic comparability with the committed results; it is
      recorded in the output.
    * Tail probabilities are stored both raw and in log space (``log_p``,
      ``log_pq``) so underflowed ``0.0`` values remain comparable.
    * A single study gets explicit ``None`` heterogeneity statistics.
    """
    d = np.ascontiguousarray(np.asarray(deltas, dtype=np.float64).ravel())
    s = np.ascontiguousarray(np.asarray(ses, dtype=np.float64).ravel())
    if d.ndim != 1 or s.ndim != 1 or d.shape != s.shape:
        raise ValueError(f"deltas/ses must be equal 1-D arrays, got "
                         f"{d.shape} and {s.shape}")
    k = d.size
    if k == 0:
        raise ValueError("empty meta-analysis input (refusing to pool)")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(s))):
        raise ValueError("nonfinite delta/se in meta-analysis input")
    if np.any(np.abs(d) > 1.0 + 1e-12):
        raise ValueError("delta outside [-1, 1] in meta-analysis input")
    if np.any(s <= 0.0):
        raise ValueError("nonpositive SE in meta-analysis input")

    xd = d.astype(np.longdouble)
    xs = s.astype(np.longdouble)
    w = np.ones_like(xs, dtype=np.longdouble) / (xs * xs)
    sw = w.sum()

    def _f(x):
        return float(x)

    d_FE = _f((w * xd).sum() / sw)
    base = {"label": label, "k": int(k), "ci_critical": ci_critical}
    if k == 1:
        return dict(base, delta=d_FE, se=_f(xs[0]),
                    ci_lo=d_FE - ci_critical * _f(xs[0]),
                    ci_hi=d_FE + ci_critical * _f(xs[0]),
                    z=_f(xd[0] / xs[0]),
                    p=_f(2.0 * _norm.sf(abs(float(xd[0] / xs[0])))),
                    log_p=_logsf_abs(_f(xd[0] / xs[0])),
                    Q=None, df=0, tau2=None, I2=None, pQ=None,
                    log_pq=None, delta_FE=d_FE,
                    heterogeneity_status="not_defined_single_study")
    i, j = np.triu_indices(k, 1)
    pair = (w[i] * w[j] * (xd[i] - xd[j]) ** 2).sum() / sw
    Q = _f(pair)
    df = k - 1
    c = _f(sw - (w * w).sum() / sw)
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    wstar = np.ones_like(xs, dtype=np.longdouble) / (xs * xs + tau2)
    swstar = wstar.sum()
    d_DL = _f((wstar * xd).sum() / swstar)
    se_DL = _f(np.sqrt(np.longdouble(1.0) / swstar))
    z = d_DL / se_DL
    p = _f(2.0 * _norm.sf(abs(z)))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    pQ = _f(_chi2.sf(Q, df))
    # Modified Hartung-Knapp SENSITIVITY (audit M04): a wider random-effects
    # interval under a Hartung-Knapp-type variance inflation.  It is NOT an
    # automatic replacement for the preregistered DL interval and does not
    # repair dependence between donors; it is reported alongside it.
    var_re = xs * xs + tau2
    raw_w = np.ones_like(xs, dtype=np.longdouble) / var_re
    hk_scale = max(np.longdouble(1.0),
                   (raw_w * (xd - np.longdouble(d_DL)) ** 2).sum() / df)
    hk_se = float(np.sqrt(hk_scale / raw_w.sum()))
    hk_crit = float(_t.ppf(0.975, df))
    modified_hk = {
        "se": hk_se, "df": int(df),
        "ci_lo": d_DL - hk_crit * hk_se,
        "ci_hi": d_DL + hk_crit * hk_se,
        "p": _f(2.0 * _t.sf(abs(d_DL / hk_se), df)),
        "note": "sensitivity only; not a replacement for the "
                "preregistered DL interval",
    }
    return dict(base, delta=d_DL, se=se_DL,
                ci_lo=d_DL - ci_critical * se_DL,
                ci_hi=d_DL + ci_critical * se_DL, z=float(z), p=p,
                log_p=_logsf_abs(float(z)), Q=Q, df=df, tau2=float(tau2),
                I2=float(I2), pQ=pQ,
                log_pq=float(_chi2.logsf(Q, df)), delta_FE=d_FE,
                modified_hk=modified_hk,
                heterogeneity_status="ok")


def bootstrap_dl(deltas, ses, B=1000, seed=0, ci=(2.5, 97.5)):
    """Patient-clustered bootstrap of the DL pooled delta.

    Every replicate is validated (a nonfinite replicate raises — failed
    replicates are never silently filtered before percentiles) and the
    exact seed, replicate count and quantile convention are recorded.
    """
    d = np.ascontiguousarray(np.asarray(deltas, dtype=np.float64).ravel())
    s = np.ascontiguousarray(np.asarray(ses, dtype=np.float64).ravel())
    if d.size != s.size:
        raise ValueError("deltas/ses length mismatch")
    if (isinstance(B, (bool, np.bool_))
            or not isinstance(B, (int, np.integer)) or int(B) < 1):
        raise ValueError(f"B must be a positive int, got {B!r}")
    k = d.size
    if k < 2:
        return {"B": 0, "status": "not_run_k_lt_2", "seed": int(seed),
                "quantile_convention": "linear (numpy percentile)"}
    rng = np.random.default_rng(seed)
    boot = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        idx = rng.integers(0, k, size=k)
        boot[b] = dl_meta(d[idx], s[idx])["delta"]
    if not np.all(np.isfinite(boot)):
        raise NumericsError("nonfinite bootstrap replicate")
    lo = float(np.percentile(boot, ci[0]))
    hi = float(np.percentile(boot, ci[1]))
    return {"B": int(B), "status": "ok", "seed": int(seed),
            "mean": float(np.mean(boot)), "ci_lo": min(lo, hi),
            "ci_hi": max(lo, hi),
            "quantile_convention": "linear (numpy percentile)",
            "ci_percentiles": [float(ci[0]), float(ci[1])],
            "n_failed_replicates": 0}


def lgamma_ratio(n, k):  # small exact helper kept for potential controls
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
