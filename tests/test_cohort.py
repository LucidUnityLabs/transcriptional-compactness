"""Cohort-normalization tests (R04): the library-size denominator is
computed over ALL raw genes BEFORE any gene filtering.

The legacy held-out paths filtered genes first, making the per-cell
denominator depend on the filter — a silent method change.  The
corrected contract: full raw library, explicit failure on empty
libraries (a silent ``0 -> 1`` substitution fabricates expression),
float64 end to end, and a facts record identifying the versioned change.
"""

import numpy as np
import pytest

from lib.cohort import log_cpm_full_library
from lib.numerics import NumericsError


def test_denominator_uses_unfiltered_library():
    """gene 0 is detected in both cells; gene 1 only in one.  With
    ``min_detected=2`` gene 1 is filtered AFTER the library sizes
    [10, 10] (over all three genes) are fixed."""
    counts = np.array([[9.0, 9.0],     # gene 0: in both cells
                       [1.0, 0.0],     # gene 1: one cell
                       [0.0, 1.0]])    # gene 2: one cell
    X_log, keep, facts = log_cpm_full_library(counts, min_detected=2,
                                              target_sum=100.0)
    assert keep.tolist() == [0]
    assert X_log.shape == (1, 2)
    # per-cell library = 9+1+0 = 10 and 9+0+1 = 10 (FULL raw library,
    # not the post-filter 9); CPM(100) of gene 0 = 90 in both cells
    assert X_log.ravel() == pytest.approx([np.log1p(90.0)] * 2)
    assert X_log.dtype == np.float64


def test_filter_dependence_is_impossible():
    """The corrected value must reflect the PRE-filter library (10 per
    cell): a filter-first denominator (the legacy order) would use 9 and
    produce log1p(100) — a silently different quantity."""
    counts = np.array([[9.0, 9.0],     # gene 0: in both cells
                       [1.0, 0.0],     # gene 1: one cell (filtered out)
                       [0.0, 1.0]])    # gene 2: one cell (filtered out)
    X_log, keep, _ = log_cpm_full_library(counts, min_detected=2,
                                          target_sum=100.0)
    assert keep.tolist() == [0]
    corrected = X_log[0, 0]
    assert corrected == pytest.approx(np.log1p(90.0))   # lib 9+1+0 = 10
    legacy_filter_first = np.log1p(9.0 / 9.0 * 100.0)   # lib 9 only
    assert corrected != pytest.approx(legacy_filter_first)


def test_facts_record_the_versioned_denominator_change():
    _, _, facts = log_cpm_full_library(np.eye(3), min_detected=1)
    assert facts["denominator"] == \
        "full raw library (computed before gene filtering)"
    assert facts["n_genes_full"] == 3
    assert facts["n_genes_kept"] == 3
    assert facts["min_detected_cells"] == 1
    assert facts["target_sum"] == 1e6
    assert facts["dtype"] == "float64"


def test_empty_library_is_an_explicit_failure():
    """Legacy substituted ``0 -> 1`` denominators; corrected REFUSES:
    excluding such cells is a caller QC decision that must be recorded."""
    counts = np.array([[1.0, 0.0], [0.0, 0.0]])  # cell 1 all-zero
    with pytest.raises(NumericsError, match="empty raw libraries"):
        log_cpm_full_library(counts, min_detected=1)


def test_negative_and_nonfinite_counts_rejected():
    with pytest.raises(NumericsError, match="nonnegative"):
        log_cpm_full_library(np.array([[1.0], [-1.0]]))
    with pytest.raises(NumericsError, match="finite"):
        log_cpm_full_library(np.array([[np.nan], [1.0]]))


def test_no_genes_pass_filter_is_a_failure():
    counts = np.array([[1.0, 0.0], [0.0, 1.0]])  # each gene in one cell
    with pytest.raises(NumericsError, match="no genes detected"):
        log_cpm_full_library(counts, min_detected=2)


def test_bad_shape_rejected():
    with pytest.raises(NumericsError, match="bad counts matrix"):
        log_cpm_full_library(np.zeros((0, 3)))
    with pytest.raises(NumericsError, match="bad counts matrix"):
        log_cpm_full_library(np.zeros((2, 2, 2)))
