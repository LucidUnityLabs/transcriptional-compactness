"""Frozen-geometry label contrasts and corrected community annotation (TC-1).

Fixes relative to the historical LABEL_VALIDATION driver:

* The "fixed geometry" decomposition was not fixed (audit C12): it
  resampled each label mask, re-centred and PER-AXIS rescaled the selected
  coordinates (an anisotropic metric change), rebuilt the kNN graph and
  redrew the edge sample for every scheme.  :func:`freeze_geometry`
  freezes the cell universe, embedding, graph and measured edge
  IDENTITIES once; :func:`frozen_label_contrasts` then applies ONLY label
  masks (same cell order, no rescaling, no subsampling) and rejects
  mask overlap unless a specified ambiguity rule has been applied.
* Louvain consumed the distance attribute as connection strength, so
  longer edges counted as STRONGER connections (audit C11).  The
  corrected community annotation uses binary connectivity
  (``weight=None``) and records the community-weight definition; the
  distance attribute used by shortest paths is never overwritten.
"""

from __future__ import annotations

import numpy as np

from . import numerics


class PreprocessError(ValueError):
    pass


def freeze_geometry(X, k=15, alpha=0.5, n_edges=4000, seed=0,
                    method_note=""):
    """Compute the label-independent geometry ONCE.

    Builds the identity-safe kNN graph and measures the curvature of a
    fixed, recorded edge draw.  Returns the per-cell curvature records
    plus the exact measured edge identities, so that downstream label
    contrasts can never silently re-derive any of these objects.
    """
    G = numerics.knn_graph(X, k=k)
    rng = np.random.default_rng(seed)
    edges = numerics.ricci_edges(G, alpha=alpha, n_edges=n_edges, rng=rng)
    cells = numerics.cell_curvature(G, edges)
    return {
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "measured_edges": sorted(f"{u},{v}" for (u, v) in edges),
        "cell_curvature": cells,
        "k": int(k), "alpha": float(alpha), "seed": int(seed),
        "method_note": method_note,
    }


def frozen_label_contrasts(frozen, masks, contrast_names=(None, None),
                           min_cells=30, overlap_rule="reject"):
    """Apply ONLY label masks to a frozen geometry; no refitting allowed.

    ``frozen`` is the output of :func:`freeze_geometry`.  ``masks`` maps
    scheme name -> (mask_a, mask_b) boolean arrays aligned to the frozen
    cell order.  For every scheme the same frozen per-cell curvature
    values are contrasted between groups:

    * the two masks must not overlap (``overlap_rule="reject"`` raises;
      ``"exclude_overlap"`` drops overlapping cells from BOTH groups and
      records the count);
    * no resampling, no per-axis rescaling, no graph rebuild happens here
      — the function has no access to the embedding at all;
    * per-scheme arm-level inference uses :func:`numerics.cliff_placements`
      whose degenerate cases are reported as non-estimable.
    """
    cells = frozen["cell_curvature"]
    n = len(cells)
    out = {}
    for scheme, (mask_a, mask_b) in masks.items():
        ma = np.asarray(mask_a, dtype=bool)
        mb = np.asarray(mask_b, dtype=bool)
        if ma.shape != (n,) or mb.shape != (n,):
            raise PreprocessError(
                f"scheme {scheme!r}: masks must be boolean arrays of the "
                f"frozen cell count {n}, got {ma.shape} and {mb.shape}")
        n_overlap = int(np.sum(ma & mb))
        if n_overlap and overlap_rule == "reject":
            raise PreprocessError(
                f"scheme {scheme!r}: malignant and comparator masks overlap "
                f"in {n_overlap} cells; apply a reviewed ambiguity rule "
                "explicitly (overlap_rule='exclude_overlap') instead of "
                "letting a newly derived mask silently reassign cells")
        if n_overlap:
            keep = ~(ma & mb)
            ma, mb = ma & keep, mb & keep
        ka = np.array([cells[i].kappa for i in np.flatnonzero(ma)
                       if cells[i].status == "measured"], dtype=float)
        kb = np.array([cells[i].kappa for i in np.flatnonzero(mb)
                       if cells[i].status == "measured"], dtype=float)
        rec = {
            "n_mask_a": int(ma.sum()), "n_mask_b": int(mb.sum()),
            "n_overlap_excluded": n_overlap,
            "n_measured_a": int(ka.size), "n_measured_b": int(kb.size),
        }
        if ka.size < min_cells or kb.size < min_cells:
            rec.update(ok=False,
                       reason=f"fewer than {min_cells} measured cells in an "
                              f"arm ({ka.size}/{kb.size})")
            out[scheme] = rec
            continue
        res = numerics.cliff_placements(ka, kb)
        rec.update(ok=True,
                   mean_a=float(ka.mean()), mean_b=float(kb.mean()),
                   delta=res.delta, se=res.se,
                   se_estimable=bool(res.estimable), status=res.status)
        out[scheme] = rec
    return out


def community_annotation_unweighted(G, seed=0):
    """Louvain communities on BINARY kNN connectivity (corrected C11).

    The historical call ``louvain_communities(G, seed=SEED)`` treated the
    edge ``weight`` (Euclidean DISTANCE) as connection strength, so longer
    edges bound communities more strongly — a semantic mismatch with the
    stated design.  ``weight=None`` uses binary connectivity instead; the
    distance attribute is not modified.  Record the returned definition
    string with any results derived from these communities.
    """
    from networkx.algorithms.community import louvain_communities

    communities = [set(c) for c in louvain_communities(G, weight=None,
                                                       seed=seed)]
    return communities, {
        "community_weight_definition":
            "binary kNN connectivity (louvain weight=None)",
        "seed": int(seed),
    }
