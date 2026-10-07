"""Shared corrected numerical/validation library (method version TC-1).

This package centralizes the corrected implementations introduced by the
2026-10-06 correctness audit. The per-experiment ``run.py`` scripts of the
committed release are preserved unmodified as the historical analysis; the
corrected method is identified separately by METHOD_VERSION and consumed by
the ``run_corrected.py`` drivers.

Module map
----------
numerics      corrected Ollivier-Ricci curvature, kNN graph construction,
              verified optimal transport, coverage-aware per-cell summaries,
              Cliff's delta placements, DerSimonian-Laird meta-analysis.
bio_io        strict expression/matrix readers and label/axis identity gates.
preprocess    frozen-geometry label contrasts and corrected community
              annotation.
verification  complete scientific comparison gates and label verdicts.
cache         content-addressed, method-bound result cache.
acquisition   atomic verified downloads and SOFT-derived matrix plans.
"""

# Method identity for every corrected computation.  Bump on any semantic
# change to the library; caches and results bind to this string.
METHOD_VERSION = "TC-1"

# Human-readable description of what changed relative to the historical
# analysis (archived verbatim in the per-experiment run.py scripts).
METHOD_DIFFERENCE = (
    "full-graph support-to-support transport metric (legacy computed "
    "shortest paths on the induced support subgraph only); identity-safe "
    "kNN construction (legacy dropped neighbor position zero); verified "
    "transport solves that fail closed; exact Cliff placement variances "
    "with explicit non-estimable status (legacy floored variance at 1e-12); "
    "validating meta-analysis helpers (legacy silently filtered malformed "
    "inputs)"
)
