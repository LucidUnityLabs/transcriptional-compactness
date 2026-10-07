"""Numerics tests: analytic fixtures, verified transport, exact Cliff
placements, validating DL meta-analysis, bootstrap.

The DL tests recompute the pooling from the COMMITTED 35-row discovery
table (experiments/META_patient_level/per_patient_delta.csv) and must
recover the committed pooled numbers — the audit's independent
table-to-summary arithmetic check (report §1, S19/S20).  This validates
arithmetic only: it does NOT certify the upstream curvature values (see
test_counterexamples.py for why those change under the corrected metric).
"""

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import lib.numerics as numerics
from lib.numerics import (NumericsError, TransportError, bootstrap_dl,
                      cell_curvature, cliff_placements, dl_meta,
                      knn_graph, ricci_edges, transport)

REPO = Path(__file__).resolve().parent.parent
META_DIR = REPO / "experiments" / "META_patient_level"


# ---------------------------------------------------------------------------
# transport adapter (C04)
# ---------------------------------------------------------------------------
def test_transport_matches_bruteforce_lp():
    rng = np.random.default_rng(3)
    for _ in range(5):
        n, m = rng.integers(2, 6, size=2)
        a = rng.random(n)
        b = rng.random(m)
        a /= a.sum()
        b /= b.sum()
        C = rng.random((n, m)) * 10
        r = transport(a, b, C)
        # independent LP recomputation
        from scipy.optimize import linprog
        A_eq = np.zeros((n + m, n * m))
        for i in range(n):
            A_eq[i, i * m:(i + 1) * m] = 1.0
        for j in range(m):
            A_eq[n + j, j::m] = 1.0
        res = linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]),
                      bounds=(0, None), method="highs")
        assert r.cost == pytest.approx(res.fun, abs=1e-10)
        assert r.checks["row_residual"] < 1e-8
        assert r.checks["col_residual"] < 1e-8
        assert r.checks["objective_gap"] < 1e-8


def test_transport_rejects_mismatched_mass():
    with pytest.raises(TransportError):
        transport(np.array([0.6, 0.6]), np.array([0.5, 0.5]),
                  np.ones((2, 2)))
    with pytest.raises(TransportError):
        transport(np.array([-0.5, 1.5]), np.array([0.5, 0.5]),
                  np.ones((2, 2)))


def test_transport_rejects_nonfinite_cost():
    with pytest.raises(TransportError):
        transport(np.array([0.5, 0.5]), np.array([0.5, 0.5]),
                  np.array([[1.0, np.inf], [1.0, 1.0]]))


def test_transport_ill_conditioned_marginal_tolerance_scales():
    """Tiny mass errors are not harmless on ill-conditioned costs: the
    SAME absolute mass-total gap is judged at the cost magnitude."""
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5 + 5e-8])          # total mass off by 5e-8
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(TransportError):
        transport(a, b, C)                    # rejected at unit scale
    transport(a, b, C * 1e6)                  # accepted at 1e6 cost scale


# ---------------------------------------------------------------------------
# analytic OR fixtures
# ---------------------------------------------------------------------------
def _path_graph_uniform(n):
    G = nx.path_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def test_two_node_graph_curvature():
    """Two nodes joined by one edge: the lazy-walk measures coincide
    (both are 0.5*delta_0 + 0.5*delta_1), so W=0 and kappa=1."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.5)
    out = ricci_edges(G, alpha=0.5)
    (ec,) = out.values()
    assert ec.kappa == pytest.approx(1.0, abs=1e-12)


def test_triangle_symmetry():
    """Unit triangle: by symmetry W = 0.25 exactly, so kappa = 0.75.

    Hand derivation: supplies {0:1/2, 1:1/4, 2:1/4}, demands
    {1:1/2, 0:1/4, 2:1/4}, unit metric; the only unmet mass is 1/4 of
    demand at node 1 supplied from node 0 (or 2) at cost 1."""
    G = nx.cycle_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    out = ricci_edges(G, alpha=0.5)
    for ec in out.values():
        assert ec.kappa == pytest.approx(0.75, abs=1e-9)


def test_square_graph_curvature():
    """Unit square: W = 0.5 (a quarter-mass travels 0->1 and a
    quarter-mass travels 3->2, each at cost 1), so kappa = 0.5.

    With the triangle (0.75) and five-cycle (0.25) this forms the
    monotone family kappa(C_3)=0.75, kappa(C_4)=0.5, kappa(C_5)=0.25
    anchored by the audit's five-cycle counterexample."""
    G = nx.cycle_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    out = ricci_edges(G, alpha=0.5)
    for ec in out.values():
        assert ec.kappa == pytest.approx(0.5, abs=1e-9)


def test_upper_bound_is_one():
    """Curvature is bounded above by 1 (up to tolerance)."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 5))
    G = knn_graph(X, k=6)
    out = ricci_edges(G, alpha=0.5)
    assert all(ec.kappa <= 1.0 + 1e-9 for ec in out.values())


def test_ricci_rejects_nonpositive_weights():
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(1, 2, weight=1.0)
    with pytest.raises(NumericsError):
        ricci_edges(G, alpha=0.5)
    G2 = nx.Graph()
    G2.add_edge(0, 1, weight=-1.0)
    with pytest.raises(NumericsError):
        ricci_edges(G2, alpha=0.5)


def test_ricci_empty_graph_returns_empty():
    G3 = nx.Graph()
    G3.add_node(0)
    G3.add_node(1)
    assert ricci_edges(G3, alpha=0.5) == {}


def test_edge_sampling_deterministic():
    rng = np.random.default_rng(20260507)
    X = rng.normal(size=(50, 4))
    G = knn_graph(X, k=5)
    o1 = ricci_edges(G, alpha=0.5, n_edges=20, rng=np.random.default_rng(1))
    o2 = ricci_edges(G, alpha=0.5, n_edges=20, rng=np.random.default_rng(1))
    assert set(o1) == set(o2)
    assert all(o1[e].kappa == o2[e].kappa for e in o1)


# ---------------------------------------------------------------------------
# coverage-aware cell summaries (C05)
# ---------------------------------------------------------------------------
def test_cell_curvature_coverage_records():
    G = _path_graph_uniform(6)
    edges = list(G.edges())
    sub = {e: numerics.EdgeCurvature(0.25, 1.0, 1.0, {}, "test")
           for e in edges[:2]}
    cells = cell_curvature(G, sub)
    # node 0 touches only edge (0,1) -> covered; node 5 touches only
    # (4,5) -> not sampled
    assert cells[0].status == "measured"
    assert cells[0].n_incident_measured == 1
    assert cells[0].degree == 1
    assert cells[0].coverage_fraction == 1.0
    assert cells[5].status == "not_sampled"
    assert cells[5].kappa != cells[5].kappa  # NaN sentinel for not sampled


def test_cell_curvature_rejects_nonfinite_measured():
    G = _path_graph_uniform(4)
    bad = numerics.EdgeCurvature(float("inf"), 1.0, 1.0, {}, "test")
    with pytest.raises(NumericsError):
        cell_curvature(G, {(0, 1): bad})


def test_full_pipeline_coverage_fractions_sum():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(120, 6))
    G = knn_graph(X, k=8)
    out = ricci_edges(G, alpha=0.5, n_edges=100)
    cells = cell_curvature(G, out)
    n_inc = sum(c.n_incident_measured for c in cells.values())
    assert n_inc == 2 * len(out)  # each measured edge touches two cells
    sampled = [c for c in cells.values() if c.status == "measured"]
    assert sampled
    assert all(0.0 < c.coverage_fraction <= 1.0 for c in sampled)


# ---------------------------------------------------------------------------
# exact Cliff placements (C06)
# ---------------------------------------------------------------------------
def _dominance_matrix_delta_se(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    D = np.sign(a[:, None] - b[None, :])
    row, col = D.mean(axis=1), D.mean(axis=0)
    var = row.var(ddof=1) / len(a) + col.var(ddof=1) / len(b)
    return float(D.mean()), float(np.sqrt(var)) if var > 0 else None


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_cliff_matches_dominance_matrix_exact(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=37)
    b = rng.normal(size=53) + 0.3
    res = cliff_placements(a, b)
    d_ref, se_ref = _dominance_matrix_delta_se(a, b)
    assert res.delta == pytest.approx(d_ref, abs=1e-12)
    assert res.se == pytest.approx(se_ref, abs=1e-12)
    assert res.estimable


def test_cliff_with_ties_exact():
    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 2.0, 4.0])
    res = cliff_placements(a, b)
    d_ref, _ = _dominance_matrix_delta_se(a, b)
    assert res.delta == pytest.approx(d_ref, abs=1e-12)


def test_cliff_rejects_bad_input():
    with pytest.raises(ValueError):
        cliff_placements([], [1.0])
    with pytest.raises(ValueError):
        cliff_placements([np.nan], [1.0])


# ---------------------------------------------------------------------------
# DL meta-analysis (C07) — including the committed-table arithmetic check
# ---------------------------------------------------------------------------
def test_dl_validates_inputs():
    with pytest.raises(ValueError):
        dl_meta([], [])
    with pytest.raises(ValueError):
        dl_meta([np.nan, 0.5], [0.1, 0.1])
    with pytest.raises(ValueError):
        dl_meta([0.5, 1.5], [0.1, 0.1])       # delta outside [-1, 1]
    with pytest.raises(ValueError):
        dl_meta([0.5, 0.5], [0.1, 0.0])       # nonpositive SE
    with pytest.raises(ValueError):
        dl_meta(np.zeros((2, 1)), np.zeros(2))  # shape mismatch, no silent
    # broadcast would be silent under legacy shapes; corrected rejects


def test_dl_single_study_heterogeneity_missing():
    r = dl_meta([0.4], [0.2])
    assert r["k"] == 1
    assert r["Q"] is None and r["tau2"] is None and r["I2"] is None
    assert r["heterogeneity_status"] == "not_defined_single_study"
    assert r["delta"] == pytest.approx(0.4)


def test_dl_pair_product_form_matches_classic():
    rng = np.random.default_rng(9)
    d = rng.uniform(-0.8, 0.8, size=25)
    s = rng.uniform(0.05, 0.3, size=25)
    r = dl_meta(d, s)
    w = 1.0 / s ** 2
    dfe = (w * d).sum() / w.sum()
    Q_classic = (w * (d - dfe) ** 2).sum()
    assert r["Q"] == pytest.approx(Q_classic, rel=1e-10)


def test_dl_extreme_weights_extended_precision():
    """SEs spanning many orders of magnitude must not lose the small
    studies entirely to float cancellation."""
    d = np.array([0.5, 0.5, 0.5, 0.5])
    s = np.array([1e-8, 1.0, 1.0, 1.0])
    r = dl_meta(d, s)
    # the tiny-SE study dominates the FE estimate; FE == 0.5 exactly here
    assert r["delta_FE"] == pytest.approx(0.5, abs=1e-6)
    assert math.isfinite(r["delta"]) and math.isfinite(r["se"])


def test_dl_log_tail_probabilities_recorded():
    r = dl_meta([0.6] * 30, [0.01] * 30)
    assert r["p"] < 1e-300 or r["p"] == 0.0
    assert r["log_p"] < -690  # log(2) + logsf: ~ -705
    assert math.isfinite(r["log_p"])


def test_dl_reproduces_committed_discovery_overall():
    """S19 arithmetic check: recompute DL pooling from the COMMITTED
    35-row per-patient table and recover the committed results.json
    overall numbers (arithmetic equivalence, ~1e-9)."""
    import pandas as pd

    pp = pd.read_csv(META_DIR / "per_patient_delta.csv")
    committed = json.load(open(META_DIR / "results.json"))
    overall = committed["overall_patient_level"]
    assert len(pp) == committed["n_patients_total"] == 35
    r = dl_meta(pp["delta"].values, pp["se"].values,
                label="OVERALL_patient_level")
    assert r["delta"] == pytest.approx(overall["delta"], rel=1e-9)
    assert r["se"] == pytest.approx(overall["se"], rel=1e-9)
    assert r["ci_lo"] == pytest.approx(overall["ci_lo"], rel=1e-9)
    assert r["ci_hi"] == pytest.approx(overall["ci_hi"], rel=1e-9)
    assert r["p"] == pytest.approx(overall["p"], rel=1e-6)
    assert r["Q"] == pytest.approx(overall["Q"], rel=1e-9)
    assert r["tau2"] == pytest.approx(overall["tau2"], rel=1e-9)
    assert r["I2"] == pytest.approx(overall["I2"], rel=1e-9)
    assert r["pQ"] == pytest.approx(overall["pQ"], rel=1e-6)


def test_bootstrap_reproduces_committed_interval():
    """The committed bootstrap interval (seed 20260508 = SEED+1, B=1000)
    must be recovered by the validating bootstrap."""
    import pandas as pd

    pp = pd.read_csv(META_DIR / "per_patient_delta.csv")
    committed = json.load(open(META_DIR / "results.json"))
    boot = committed["overall_patient_level"]["bootstrap"]
    r = bootstrap_dl(pp["delta"].values, pp["se"].values, B=1000,
                     seed=20260508)
    assert r["B"] == boot["B"] == 1000
    assert r["mean"] == pytest.approx(boot["mean"], rel=1e-9)
    assert r["ci_lo"] == pytest.approx(boot["ci_lo"], rel=1e-9)
    assert r["ci_hi"] == pytest.approx(boot["ci_hi"], rel=1e-9)
    assert r["n_failed_replicates"] == 0


def test_bootstrap_rejects_silently_failed_replicates(monkeypatch):
    real = dl_meta

    def poisoned(deltas, ses, label="", **kw):
        out = real(deltas, ses, label=label, **kw)
        if out["k"] == 3 and not getattr(poisoned, "hit", False):
            poisoned.hit = True
            out = dict(out, delta=float("nan"))
        return out

    poisoned.hit = False
    monkeypatch.setattr(numerics, "dl_meta", poisoned)
    with pytest.raises(NumericsError):
        bootstrap_dl([0.1, 0.2, 0.3], [0.1, 0.1, 0.1], B=16, seed=0)
    monkeypatch.setattr(numerics, "dl_meta", real)


def test_bootstrap_k_lt_2_status():
    r = bootstrap_dl([0.3], [0.1], B=100, seed=0)
    assert r["B"] == 0 and r["status"] == "not_run_k_lt_2"


def test_one_sided_sf_convention_preserved():
    """The one-sided p-value convention norm.sf(z) is correct and kept
    (audit C07: do NOT replace it with half the two-sided p for negative
    effects — for z < 0, norm.sf(z) ~ 1 (not significant) while p/2 ~ 0
    would falsely indicate significance)."""
    from scipy.stats import norm

    r_neg = dl_meta([-0.6] * 12, [0.05] * 12)
    assert r_neg["z"] < 0
    assert norm.sf(r_neg["z"]) == pytest.approx(1.0)
    assert 0.5 * r_neg["p"] == pytest.approx(0.0, abs=1e-12)
    assert norm.sf(r_neg["z"]) != pytest.approx(0.5 * r_neg["p"])
    # for positive z the two conventions coincide
    r_pos = dl_meta([0.6] * 12, [0.05] * 12)
    assert norm.sf(r_pos["z"]) == pytest.approx(0.5 * r_pos["p"],
                                                rel=1e-6)


# ---------------------------------------------------------------------------
# input-contract validation (ported from the audit's reference suite)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [-0.1, 1.1, 0.0, 1.0,
                                   float("nan"), float("inf")])
def test_ricci_rejects_invalid_alpha(alpha):
    """alpha is the lazy (self) mass and must lie strictly inside (0, 1);
    the boundaries are rejected by the TC-1 contract (the reference
    implementation accepts and computes them — a deliberate divergence,
    fail-closed here)."""
    G = nx.cycle_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    with pytest.raises(ValueError):
        ricci_edges(G, alpha=alpha)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, np.int64(0)])
def test_ricci_rejects_bad_edge_budget(bad):
    """n_edges is a sampled-edge COUNT: positive int, never a bool (True
    would silently mean 'sample exactly one edge')."""
    G = nx.cycle_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    with pytest.raises(ValueError):
        ricci_edges(G, n_edges=bad)


def test_cell_curvature_rejects_reverse_duplicate_edge():
    """An edge recorded as both (u, v) and (v, u) would double-count every
    incident-cell summary — rejected outright."""
    G = _path_graph_uniform(4)
    ec = numerics.EdgeCurvature(0.25, 1.0, 1.0, {}, "test")
    with pytest.raises(NumericsError, match="recorded twice"):
        cell_curvature(G, {(0, 1): ec, (1, 0): ec})
    # the canonical single record is fine
    assert cell_curvature(G, {(0, 1): ec})[0].status == "measured"


# ---------------------------------------------------------------------------
# modified Hartung-Knapp sensitivity (M04)
# ---------------------------------------------------------------------------
def test_dl_modified_hk_is_a_recorded_sensitivity():
    """The reported uncertainty must match the estimand: HK-type variance
    inflation is reported ALONGSIDE the preregistered DL interval (wider
    or equal), never as a replacement for it."""
    # varied SEs with spread effects inflate the RE quadratic form past
    # its degrees of freedom -> a strictly wider sensitivity interval
    d = [0.1, 0.2, -0.3, 0.05, 0.4, -0.1]
    s = [0.05, 0.1, 0.4, 0.05, 0.3, 0.1]
    r = dl_meta(d, s)
    hk = r["modified_hk"]
    assert hk["df"] == r["df"] == 5
    assert hk["se"] > r["se"]
    assert hk["ci_lo"] < r["ci_lo"]
    assert hk["ci_hi"] > r["ci_hi"]
    assert "not a replacement" in hk["note"]
    # homogeneous effects: no inflation (scale pinned at 1) and the HK SE
    # degenerates to the DL SE
    homo = dl_meta([0.5] * 6, [0.2] * 6)
    assert homo["modified_hk"]["se"] == pytest.approx(homo["se"])
    assert homo["modified_hk"]["se"] == pytest.approx(
        float(np.sqrt(1.0 / (6 / (0.2 ** 2)))), rel=1e-9)


def test_dl_modified_hk_never_rewrites_the_preregistered_fields():
    """Whatever the heterogeneity, delta/se/ci (the preregistered DL
    fields) keep their DL values; the sensitivity lives only in its own
    block."""
    rng = np.random.default_rng(23)
    for _ in range(5):
        d = rng.uniform(-0.7, 0.7, size=7)
        s = rng.uniform(0.05, 0.4, size=7)
        r = dl_meta(d, s)
        bare = {k: v for k, v in r.items() if k != "modified_hk"}
        again = dl_meta(d, s)
        assert all(again[k] == v for k, v in bare.items())
        hk = again["modified_hk"]
        assert hk["se"] >= again["se"]


# ---------------------------------------------------------------------------
# bootstrap input validation (reference-suite port)
# ---------------------------------------------------------------------------
def test_bootstrap_validates_B_and_repeats():
    a = bootstrap_dl([0.1, 0.3, 0.5], [0.1, 0.2, 0.15], B=20, seed=1)
    b = bootstrap_dl([0.1, 0.3, 0.5], [0.1, 0.2, 0.15], B=20, seed=1)
    assert a == b  # seeded repeat is bit-identical
    assert a["B"] == 20 and a["seed"] == 1
    for bad in (0, -5, 1.5, True):
        with pytest.raises(ValueError):
            bootstrap_dl([0.1, 0.2], [0.1, 0.1], B=bad)


# ---------------------------------------------------------------------------
# Cliff's delta vs independent U-statistic oracles (reference-suite port)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [([1, 2, 2, 4], [2, 3, 4]),
                                 ([0, 0, 1], [0, 1, 1]),
                                 ([-2, 0, 3], [1, 2, 4, 5])])
def test_cliff_matches_mannwhitney_and_auc(a, b):
    """delta == 2U/(n1 n2) - 1 == 2*AUC - 1 == mean dominance sign —
    three independent oracles for the same U-statistic."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    res = cliff_placements(a, b)
    U = mannwhitneyu(a, b, method="asymptotic").statistic
    assert res.delta == pytest.approx(2 * U / (len(a) * len(b)) - 1)
    auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))],
                        np.r_[a, b])
    assert res.delta == pytest.approx(2 * auc - 1)
