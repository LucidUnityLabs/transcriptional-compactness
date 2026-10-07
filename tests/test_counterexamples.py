"""The two decisive audit counterexamples, fail-before/pass-after.

C01: the legacy ``ollivier_ricci_edges`` computes support distances on the
INDUCED SUPPORT SUBGRAPH.  On the unit five-cycle (realizable as the k=2
Euclidean kNN graph of a regular pentagon) every legacy curvature is 0;
the full-graph metric gives 0.25.

C02: the legacy ``build_knn_graph`` drops neighbour position zero instead
of removing the query row by identity.  With duplicate coordinates the
first returned neighbour can be the duplicate, leaving the SELF in the
neighbour list — a self-loop edge.

These tests pin BOTH sides: the legacy behaviour (documented defect —
these assertions fail against any future 'fix' of the historical scripts,
which are intentionally preserved) and the corrected behaviour in
``lib.numerics`` (the actual fix).
"""

import networkx as nx
import numpy as np
import pytest

import lib.numerics as numerics
from lib.numerics import DuplicateCoordinatesError


def cycle5():
    G = nx.cycle_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


# ---------------------------------------------------------------------------
# C01 — five-cycle restricted vs full-graph curvature
# ---------------------------------------------------------------------------
def test_c01_legacy_restricted_metric_reports_zero(legacy_e1):
    """FAIL-BEFORE documentation: legacy reports kappa=0 on the 5-cycle."""
    G = cycle5()
    legacy = legacy_e1.ollivier_ricci_edges(G, alpha=0.5, n_edges=None)
    assert len(legacy) == 5
    for e, k in legacy.items():
        assert k == pytest.approx(0.0, abs=1e-12), \
            "legacy five-cycle curvature changed — historical scripts " \
            "must stay byte-identical"


def test_c01_restricted_support_union_cannot_reach_the_shortcut(legacy_e1):
    """The support union of edge (0,1) is {0,1,2,4}; the 4-3-2 shortcut
    (length 2) lives outside it, so the local Dijkstra sees d(4,2)=3."""
    G = cycle5()
    sub = G.subgraph({0, 1, 2, 4})
    d_sub = nx.single_source_dijkstra_path_length(sub, 4)
    assert d_sub[2] == 3.0
    d_full = nx.single_source_dijkstra_path_length(G, 4)
    assert d_full[2] == 2.0


def test_c01_corrected_full_graph_metric_reports_quarter():
    G = cycle5()
    out = numerics.ricci_edges(G, alpha=0.5)
    assert set(out) == set(G.edges())
    for e, ec in out.items():
        assert ec.kappa == pytest.approx(0.25, abs=1e-12), (
            f"full-graph curvature on the unit five-cycle must be 0.25 "
            f"(edge {e}: {ec.kappa})")
        assert ec.d_graph_uv == pytest.approx(1.0)


def test_c01_regular_pentagon_knn_is_the_five_cycle(legacy_e1):
    """The counterexample input is a plain Euclidean kNN graph, not an
    exotic construction."""
    theta = 2 * np.pi * np.arange(5) / 5
    pent = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    Gp = legacy_e1.build_knn_graph(pent, k=2)
    assert Gp.number_of_edges() == 5
    cyc = next(f for f in nx.simple_cycles(Gp, length_bound=5) if len(f) == 5)
    assert len(cyc) == 5
    legacy = legacy_e1.ollivier_ricci_edges(Gp, alpha=0.5)
    vals = np.array(list(legacy.values()))
    assert np.allclose(vals, 0.0, atol=1e-9)

    out = numerics.ricci_edges(Gp, alpha=0.5)
    corrected = np.array([ec.kappa for ec in out.values()])
    # pentagon side lengths are equal, so every edge has curvature 0.25
    assert np.allclose(corrected, 0.25, atol=1e-9), corrected


def test_c01_scale_invariance_of_the_corrected_metric():
    """Multiplying all lengths by a positive constant leaves 1 - W/d
    unchanged (the legacy 1e-6 absolute cutoff violated this)."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    G1 = numerics.knn_graph(X, k=5)
    G2 = G1.copy()
    for u, v in G2.edges():
        G2[u][v]["weight"] = G1[u][v]["weight"] * 1e-9
    G3 = G1.copy()
    for u, v in G3.edges():
        G3[u][v]["weight"] = G1[u][v]["weight"] * 1e7
    k1 = {e: ec.kappa for e, ec in numerics.ricci_edges(G1).items()}
    k2 = {e: ec.kappa for e, ec in numerics.ricci_edges(G2).items()}
    k3 = {e: ec.kappa for e, ec in numerics.ricci_edges(G3).items()}
    for e in k1:
        assert k1[e] == pytest.approx(k2[e], abs=1e-9)
        assert k1[e] == pytest.approx(k3[e], abs=1e-9)


def test_c01_restricted_cost_never_smaller_componentwise():
    """With the same edge denominator, restricted costs are componentwise
    >= full-graph costs, so legacy curvatures are depressed edge-wise."""
    G = cycle5()
    u, v = 0, 1
    sup_u, sup_v, w_u, w_v = numerics._lazy_masses(G, u, v, 0.5)
    sub = G.subgraph(set(sup_u + sup_v))
    D_sub = {a: nx.single_source_dijkstra_path_length(sub, a)
             for a in sup_u}
    M_sub = np.array([[D_sub[a].get(b, np.inf) for b in sup_v]
                      for a in sup_u])
    D_full = {a: nx.single_source_dijkstra_path_length(G, a)
              for a in sup_u}
    M_full = np.array([[D_full[a][b] for b in sup_v] for a in sup_u])
    assert np.all(M_sub >= M_full)
    r_sub = numerics.transport(w_u, w_v, M_sub)
    r_full = numerics.transport(w_u, w_v, M_full)
    assert r_sub.cost >= r_full.cost
    assert r_sub.cost == pytest.approx(1.0)
    assert r_full.cost == pytest.approx(0.75)


def test_c01_negative_curvature_is_not_clipped():
    """A legitimately very negative curvature survives: -1 is NOT a valid
    universal lower bound for the weighted metric."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=100.0)
    G.add_edge(0, 2, weight=100.0)
    G.add_edge(0, 3, weight=1.0)
    G.add_edge(1, 4, weight=1.0)
    out = numerics.ricci_edges(G, alpha=0.5)
    kappas = [ec.kappa for ec in out.values()]
    assert min(kappas) < -1.0, "expected an unclipped curvature below -1"


# ---------------------------------------------------------------------------
# C02 — duplicate coordinates and the self-loop
# ---------------------------------------------------------------------------
DUP_X = np.array([[0.0], [0.0], [1.0], [2.0]])


def test_c02_legacy_neighbour_zero_is_not_always_self(legacy_e1):
    """FAIL-BEFORE documentation: with duplicates, kneighbors can return
    the duplicate at position 0 and the query itself later, so dropping
    position 0 keeps a self-loop."""
    G = legacy_e1.build_knn_graph(DUP_X, k=2)
    self_loops = [e for e in G.edges() if e[0] == e[1]]
    assert self_loops, (
        "legacy no longer produces the self-loop — historical scripts "
        "must stay byte-identical (this pins the documented defect)")


def test_c02_corrected_builder_rejects_coincident_coordinates():
    with pytest.raises(DuplicateCoordinatesError) as ei:
        numerics.knn_graph(DUP_X, k=2)
    assert "not estimable" in str(ei.value) or "non-estimable" in \
        str(ei.value).lower() or "not estimable" in str(ei.value).lower()


def test_c02_corrected_builder_identity_safe_no_self_loops():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 4))
    G = numerics.knn_graph(X, k=6)
    assert all(u != v for u, v in G.edges())
    assert all(d["weight"] > 0 for _, _, d in G.edges(data=True))
    # every node has AT LEAST k distinct neighbours (its own k nearest
    # others), and its neighbour set is exactly its k nearest others
    from scipy.spatial.distance import cdist

    D = cdist(X, X)
    for i in range(60):
        order = sorted((d, j) for j, d in enumerate(D[i]) if j != i)
        expected = {j for _, j in order[:6]}
        # the undirected graph contains i's own k nearest others, plus
        # any node that chose i as one of ITS k nearest
        assert expected <= set(G.neighbors(i))
        assert G.degree(i) >= 6
    # determinism on identical input
    G_repeat = numerics.knn_graph(X, k=6)
    assert sorted((u, v, round(d["weight"], 12))
                  for u, v, d in G.edges(data=True)) == \
        sorted((u, v, round(d["weight"], 12))
               for u, v, d in G_repeat.edges(data=True))
    # relabeling-invariance: the same points under a permutation produce
    # the same multiset of edge weights (construction is independent of
    # point ordering)
    perm = rng.permutation(60)
    G_perm = numerics.knn_graph(X[perm], k=6)
    assert sorted(round(d["weight"], 12)
                  for _, _, d in G.edges(data=True)) == \
        sorted(round(d["weight"], 12)
               for _, _, d in G_perm.edges(data=True))


def test_c02_k_validation():
    with pytest.raises(ValueError):
        numerics.knn_graph(np.zeros((3, 2)), k=0)
    with pytest.raises(ValueError):
        numerics.knn_graph(np.zeros((3, 2)), k=3)
    with pytest.raises(ValueError):
        numerics.knn_graph(np.zeros((3, 2, 2)), k=1)


# ---------------------------------------------------------------------------
# C06 — variance floor fabricates precision
# ---------------------------------------------------------------------------
def test_c06_legacy_se_floor(legacy_meta):
    """FAIL-BEFORE documentation: complete separation / all ties get
    SE=1e-6 (inverse-variance weight 1e12)."""
    d, se = legacy_meta.cliff_delta_se(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 0.4, 0.3, 0.2]))
    assert d == 1.0 and se == 1e-6
    d2, se2 = legacy_meta.cliff_delta_se(np.zeros(4), np.zeros(4))
    assert d2 == 0.0 and se2 == 1e-6


def test_c06_corrected_reports_non_estimable():
    res = numerics.cliff_placements(np.array([1., 2., 3., 4.]),
                                    np.array([.5, .4, .3, .2]))
    assert res.delta == 1.0
    assert res.se is None and not res.estimable
    assert "non_estimable" in res.status
    res2 = numerics.cliff_placements(np.zeros(4), np.zeros(4))
    assert res2.delta == 0.0 and res2.se is None and not res2.estimable
    res3 = numerics.cliff_placements([1.0], [2.0])
    assert res3.se is None and "small_group" in res3.status
