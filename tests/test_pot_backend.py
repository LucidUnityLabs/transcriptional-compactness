"""POT backend integration test.

Per the audit report: the real POT backend was unavailable in the audit
environment, so this test is SKIPPED wherever POT is not installed and is
MANDATORY in release CI (where requirements.txt installs POT==0.9.6.post1).
conftest.py applies the skip; CI installs POT so the marker always runs
there.

When POT is present, the verified-transport adapter must agree with the
independent scipy-HiGHS LP on representative problem families, including
the exact five-cycle costs, and the solver status/warning interface must
be exercised through numerics.ricci_edges.
"""

import networkx as nx
import numpy as np
import pytest

import lib.numerics as numerics

pytestmark = pytest.mark.pot_backend


def _cycle5():
    G = nx.cycle_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def test_pot_agrees_with_lp_reference():
    rng = np.random.default_rng(21)
    for trial in range(10):
        n, m = rng.integers(2, 7, size=2)
        a = rng.random(n)
        b = rng.random(m)
        a /= a.sum()
        b /= b.sum()
        C = rng.random((n, m)) * (10 ** rng.integers(0, 6))
        r_pot = numerics.transport(a, b, C, backend="pot")
        r_lp = numerics.transport(a, b, C, backend="lp")
        assert r_pot.cost == pytest.approx(r_lp.cost, rel=1e-9,
                                           abs=1e-12), trial
        assert r_pot.backend == "POT" and r_lp.backend == "scipy-highs"


def test_pot_five_cycle_costs():
    G = _cycle5()
    out = numerics.ricci_edges(G, alpha=0.5, backend="pot")
    for ec in out.values():
        assert ec.backend == "POT"
        assert ec.kappa == pytest.approx(0.25, abs=1e-9)
        assert ec.transport_checks["objective_gap"] < 1e-8
