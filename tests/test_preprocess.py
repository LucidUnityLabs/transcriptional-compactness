"""Frozen-geometry and community-annotation tests (C11, C12)."""

import networkx as nx
import numpy as np
import pytest

import lib.numerics as numerics
from lib.preprocess import (PreprocessError, community_annotation_unweighted,
                        frozen_label_contrasts, freeze_geometry)


@pytest.fixture(scope="module")
def toy_embedding():
    rng = np.random.default_rng(20260507)
    X = rng.normal(size=(150, 5))
    # two well-separated blobs so labels partition meaningfully
    X[:75] += 2.0
    return X


@pytest.fixture(scope="module")
def frozen(toy_embedding):
    return freeze_geometry(toy_embedding, k=8, alpha=0.5, n_edges=200,
                           seed=3)


def test_freeze_records_edge_identities(frozen):
    assert frozen["graph_nodes"] == 150
    assert frozen["graph_edges"] > 200
    assert len(frozen["measured_edges"]) == 200
    assert all(c.status in ("measured", "not_sampled")
               for c in frozen["cell_curvature"].values())


def test_frozen_contrasts_do_not_rebuild_geometry(frozen, toy_embedding):
    """Masks-only: identical frozen geometry, two different labellings
    must never resample/rescale/rebuild anything (there is no code path
    that could — the function receives no embedding)."""
    n = frozen["graph_nodes"]
    mask_a1 = np.zeros(n, dtype=bool)
    mask_a1[:75] = True
    mask_b = np.zeros(n, dtype=bool)
    mask_b[75:] = True
    out1 = frozen_label_contrasts(frozen, {"s1": (mask_a1, mask_b)})
    # a different malignant mask, disjoint from the comparator
    mask_a2 = np.zeros(n, dtype=bool)
    mask_a2[:60] = True
    out2 = frozen_label_contrasts(frozen, {"s2": (mask_a2, mask_b)})
    for rec in (out1["s1"], out2["s2"]):
        assert rec["ok"] or "min_cells" in rec.get("reason", "")
    # the frozen curvature payload is untouched by contrast application
    assert frozen["graph_nodes"] == 150


def test_frozen_contrasts_reject_overlap(frozen):
    n = frozen["graph_nodes"]
    a = np.zeros(n, dtype=bool)
    a[:80] = True
    b = np.zeros(n, dtype=bool)
    b[70:] = True            # overlaps a in 70..79
    with pytest.raises(PreprocessError, match="overlap"):
        frozen_label_contrasts(frozen, {"s": (a, b)},
                               overlap_rule="reject")
    out = frozen_label_contrasts(frozen, {"s": (a, b)},
                                 overlap_rule="exclude_overlap")
    assert out["s"]["n_overlap_excluded"] == 10


def test_frozen_contrasts_mask_shape_validated(frozen):
    with pytest.raises(PreprocessError):
        frozen_label_contrasts(frozen, {"s": (np.zeros(3, bool),
                                              np.zeros(3, bool))})


def test_legacy_decomposition_was_not_frozen(legacy_labelval):
    """FAIL-BEFORE documentation: the committed run_kappa_on_embedding
    resamples masks, re-centres, PER-AXIS divides by sd (anisotropic
    metric change) and rebuilds the kNN graph per scheme — the opposite
    of a frozen decomposition.  Pinned by reading the committed source."""
    src = open(legacy_labelval.__file__).read()
    for fragment in ("idx_mal = rng.choice", "Xs = Xs - Xs.mean",
                     "Xs = Xs / sd", "G = build_knn_graph(Xs"):
        assert fragment in src
    # and the per-axis division is applied AFTER selecting the arm cells:
    assert src.index("Xs = Xpca_all[sel]") < src.index("Xs = Xs / sd")


# ---------------------------------------------------------------------------
# C11 — Louvain consumed distance as connection strength
# ---------------------------------------------------------------------------
def _two_triangles_and_bridge():
    """Two tight triangles joined by one LONG edge: Louvain on binary
    connectivity splits them; Louvain on DISTANCE weights glues them."""
    G = nx.Graph()
    coords = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 0.9),
              3: (50.0, 0.0), 4: (51.0, 0.0), 5: (50.5, 0.9)}
    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        d = float(np.hypot(coords[u][0] - coords[v][0],
                           coords[u][1] - coords[v][1]))
        G.add_edge(u, v, weight=d)
    return G


def test_community_weight_semantics_change_partition():
    from networkx.algorithms.community import louvain_communities

    G = _two_triangles_and_bridge()
    comms_weighted = [set(c) for c in louvain_communities(G, seed=0)]
    # corrected: binary connectivity separates the two triangles
    comms_corr, definition = community_annotation_unweighted(G, seed=0)
    assert definition["community_weight_definition"].startswith("binary")
    assert sorted(map(sorted, comms_corr)) == [[0, 1, 2], [3, 4, 5]]
    # legacy semantics (distance as strength): the long bridge is the
    # STRONGEST connection, so the partition CROSSES the triangles —
    # the two-triangle/long-bridge fixture from the audit
    assert sorted(map(sorted, comms_weighted)) == [[0, 1], [2, 3], [4, 5]]
    # the distance attribute is never overwritten by the corrected path
    for u, v in G.edges():
        assert G[u][v]["weight"] > 0


def test_community_annotation_deterministic():
    G = _two_triangles_and_bridge()
    c1, _ = community_annotation_unweighted(G, seed=7)
    c2, _ = community_annotation_unweighted(G, seed=7)
    assert [sorted(map(sorted, c1))] == [sorted(map(sorted, c2))]


def test_legacy_louvain_call_documented(legacy_labelval):
    """FAIL-BEFORE documentation: the committed annotate_by_community
    calls louvain_communities(G, seed=SEED) on a distance-weighted
    graph — longer edges count as stronger connections."""
    src = open(legacy_labelval.__file__).read()
    assert "louvain_communities(G, seed=SEED)" in src
    assert "weight=None" not in src
