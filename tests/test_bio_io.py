"""Strict reader and identity-gate tests (D01-D05).

Corruption suites: every malformed-input class the legacy readers silently
repaired (padded/truncated rows, filtered out-of-range coordinates,
nonfinite tokens, duplicate ids) must FAIL here with a message naming the
offending record.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.bio_io import (COUNT_SUM_LIMIT, DataValidationError,
                    align_labels, align_rna_axes, read_mtx_selected,
                    read_umi_tsv_selected, validate_label_identity,
                    validate_marker_scores)


def write_gzip(path, text):
    with gzip.open(path, "wt") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# UMI TSV reader (D01)
# ---------------------------------------------------------------------------
def _umi_file(tmp_path, rows, header="gene\tc1\tc2\tc3"):
    p = tmp_path / "umi.tsv.gz"
    return write_gzip(p, "\n".join([header] + rows) + "\n")


def test_umi_happy_path_selected_order(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2", "g2\t0\t3\t4"])
    counts, genes = read_umi_tsv_selected(p, ["c3", "c1"])
    assert genes == ["g1", "g2"]               # gene ids from the rows
    assert counts.shape == (2, 2)
    # requested order c3, c1 -> columns swapped accordingly
    assert counts[0].tolist() == [2.0, 1.0]
    assert counts[1].tolist() == [4.0, 0.0]


def test_umi_short_row_rejected_not_padded(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2", "g2\t0\t3"])
    with pytest.raises(DataValidationError, match="line 3"):
        read_umi_tsv_selected(p, ["c1"])


def test_umi_long_row_rejected_not_truncated(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2\t9"])
    with pytest.raises(DataValidationError, match="line 2"):
        read_umi_tsv_selected(p, ["c1"])


def test_umi_negative_token_rejected(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t-2\t2"])
    with pytest.raises(DataValidationError):
        read_umi_tsv_selected(p, ["c1"])


def test_umi_nonfinite_token_rejected(tmp_path):
    for tok in ("nan", "inf", "1.5", "1e3", ""):
        p = _umi_file(tmp_path, [f"g1\t1\t{tok}\t2"])
        with pytest.raises(DataValidationError):
            read_umi_tsv_selected(p, ["c1"])


def test_umi_token_in_UNSELECTED_column_still_validated(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\tnan\t2"])
    with pytest.raises(DataValidationError):
        read_umi_tsv_selected(p, ["c1"])  # c2 is unselected but validated


def test_umi_missing_requested_cell_rejected(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2"])
    with pytest.raises(DataValidationError, match="absent from header"):
        read_umi_tsv_selected(p, ["c1", "nope"])


def test_umi_duplicate_requested_rejected(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2"])
    with pytest.raises(DataValidationError, match="duplicate requested"):
        read_umi_tsv_selected(p, ["c1", "c1"])


def test_umi_duplicate_header_cell_rejected(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2"], header="gene\tc1\tc1\tc2")
    with pytest.raises(DataValidationError, match="duplicate cell ids"):
        read_umi_tsv_selected(p, ["c1"])


def test_umi_empty_gene_rejected(tmp_path):
    p = _umi_file(tmp_path, ["\t1\t0\t2"])   # empty gene id in a row
    with pytest.raises(DataValidationError, match="empty gene id"):
        read_umi_tsv_selected(p, ["c1"])


def test_umi_truncated_gzip_rejected(tmp_path):
    p = tmp_path / "trunc.tsv.gz"
    blob = ("gene\tc1\tc2\n" + "".join(f"g{i}\t1\t2\n" for i in range(500)))
    with gzip.open(p, "wt") as f:
        f.write(blob)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])  # corrupt the tail
    with pytest.raises(DataValidationError, match="gzip"):
        read_umi_tsv_selected(p, ["c1"])


def test_umi_blank_line_rejected(tmp_path):
    p = _umi_file(tmp_path, ["g1\t1\t0\t2", ""])
    with pytest.raises(DataValidationError, match="empty row"):
        read_umi_tsv_selected(p, ["c1"])


# ---------------------------------------------------------------------------
# Matrix Market reader (D02)
# ---------------------------------------------------------------------------
MTX_HDR = "%%MatrixMarket matrix coordinate integer general\n"


def _mtx_file(tmp_path, body, dims=(4, 3), nnz=None, header=MTX_HDR):
    if nnz is None:
        nnz = body.count("\n")
    text = header + f"% comment\n{dims[0]} {dims[1]} {nnz}\n" + body
    return write_gzip(tmp_path / "m.mtx.gz", text)


def test_mtx_happy_path_selected(tmp_path):
    body = "1 1 5\n2 2 7\n3 3 9\n"
    p = _mtx_file(tmp_path, body)
    sel = read_mtx_selected(p, [3, 1])
    assert sel.counts.shape == (2, 4)          # 2 selected cols x 4 rows
    assert sel.counts[0].tolist() == [0, 0, 9, 0]   # column 3 first
    assert sel.counts[1].tolist() == [5, 0, 0, 0]   # then column 1
    assert sel.n_entries_read == 3 == sel.n_entries_declared
    assert sel.n_duplicate_entries_summed == 0


def test_mtx_arbitrary_whitespace(tmp_path):
    p = write_gzip(tmp_path / "m.mtx.gz",
                   MTX_HDR + "3  2\t 2\n  1   1  4 \n 2  2  6\n")
    sel = read_mtx_selected(p, [1])
    assert sel.counts[0, 0] == 4.0


def test_mtx_duplicate_entries_summed(tmp_path):
    body = "1 2 5\n1 2 7\n3 1 1\n"
    p = _mtx_file(tmp_path, body)
    sel = read_mtx_selected(p, [2])
    assert sel.counts[0, 0] == 12.0
    assert sel.n_duplicate_entries_summed == 1


def test_mtx_2p53_guard(tmp_path):
    """A summed duplicate reaching 2^53 must not silently round in
    float64 and pass."""
    body = f"1 1 {2**53 - 1}\n1 1 1\n"
    p = _mtx_file(tmp_path, body)
    with pytest.raises(DataValidationError, match="2\\^53"):
        read_mtx_selected(p, [1])
    assert COUNT_SUM_LIMIT == 2 ** 53


def test_mtx_out_of_range_rejected_not_filtered(tmp_path):
    body = "1 1 5\n5 1 7\n"  # row 5 > declared 4
    p = _mtx_file(tmp_path, body)
    with pytest.raises(DataValidationError, match="out of bounds"):
        read_mtx_selected(p, [1])
    body2 = "0 1 5\n"        # zero-based coordinate
    p2 = _mtx_file(tmp_path, body2)
    with pytest.raises(DataValidationError, match="out of bounds"):
        read_mtx_selected(p2, [1])


def test_mtx_nnz_mismatch_rejected(tmp_path):
    body = "1 1 5\n2 2 6\n"
    p = _mtx_file(tmp_path, body, nnz=5)
    with pytest.raises(DataValidationError, match="nnz"):
        read_mtx_selected(p, [1])


def test_mtx_dims_line_not_a_data_row(tmp_path):
    """The legacy pandas reader parsed the dims line as a data row and
    heuristically removed it; the strict reader consumes it as a header."""
    body = "1 1 5\n"
    p = _mtx_file(tmp_path, body)
    sel = read_mtx_selected(p, [1])
    assert sel.counts[0, 0] == 5.0
    assert sel.n_entries_read == 1


def test_mtx_banner_and_field_rejected(tmp_path):
    for hdr in ("%%MatrixMarket matrix coordinate real general\n",
                "%%MatrixMarket matrix coordinate integer symmetric\n",
                "not a banner\n"):
        p = write_gzip(tmp_path / "m.mtx.gz",
                       hdr + "3 2 1\n1 1 4\n")
        with pytest.raises(DataValidationError):
            read_mtx_selected(p, [1])


def test_mtx_negative_count_rejected(tmp_path):
    p = _mtx_file(tmp_path, "1 1 -4\n")
    with pytest.raises(DataValidationError, match="negative"):
        read_mtx_selected(p, [1])


def test_mtx_requested_column_out_of_range(tmp_path):
    p = _mtx_file(tmp_path, "1 1 4\n")
    with pytest.raises(DataValidationError, match="out of range"):
        read_mtx_selected(p, [1, 99])
    with pytest.raises(DataValidationError, match="duplicate requested"):
        read_mtx_selected(p, [1, 1])


def test_mtx_empty_and_dup_entries(tmp_path):
    p = _mtx_file(tmp_path, "1 1 4\n")
    with pytest.raises(DataValidationError):
        read_mtx_selected(p, [])
    with pytest.raises(DataValidationError):
        read_mtx_selected(p, [])


# ---------------------------------------------------------------------------
# RNA axes (D04) and marker scores
# ---------------------------------------------------------------------------
def test_align_rna_axes_happy_and_reindex():
    perm, feats = align_rna_axes(["b", "a", "c"], ["g1", "g2"],
                                 ["a", "b", "c"])
    assert perm == [1, 0, 2]
    assert feats == ["g1", "g2"]


def test_align_rna_axes_rejects_set_mismatch():
    with pytest.raises(DataValidationError, match="do not match"):
        align_rna_axes(["a", "b"], ["g"], ["a", "x"])


def test_align_rna_axes_rejects_duplicates():
    with pytest.raises(DataValidationError, match="duplicate"):
        align_rna_axes(["a", "a"], ["g"], ["a", "b"])
    with pytest.raises(DataValidationError, match="duplicate"):
        align_rna_axes(["a", "b"], ["g", "g"], ["a", "b"])


def test_marker_scores_explicit_unknown_state():
    with pytest.raises(DataValidationError, match="no scores"):
        validate_marker_scores(np.array([]), 5, "panel")
    with pytest.raises(DataValidationError, match="no finite scores"):
        validate_marker_scores(np.array([np.nan] * 5), 5, "panel")
    with pytest.raises(DataValidationError, match="scores for"):
        validate_marker_scores(np.zeros(3), 5, "panel")


# ---------------------------------------------------------------------------
# Label identity gates (D03)
# ---------------------------------------------------------------------------
def _labels(rows):
    return pd.DataFrame(rows, columns=["barcode", "sample", "patient",
                                       "label"])


def test_label_identity_happy():
    df = _labels([("b1", "S1", "P1", "malignant"),
                  ("b2", "S1", "P1", "immune"),
                  ("b3", "S2", "P2", "immune")])
    rep = validate_label_identity(df)
    assert rep.n_conflicting_labels == 0


def test_label_identity_barcode_two_specimens_rejected():
    df = _labels([("b1", "S1", "P1", "immune"),
                  ("b1", "S2", "P1", "immune")])
    with pytest.raises(DataValidationError, match="multiple specimens"):
        validate_label_identity(df)


def test_label_identity_barcode_two_donors_rejected():
    df = _labels([("b1", "S1", "P1", "immune"),
                  ("b1", "S1", "P9", "immune")])
    with pytest.raises(DataValidationError, match="multiple donors"):
        validate_label_identity(df)


def test_label_identity_dual_label_rule():
    df = _labels([("b1", "S1", "P1", "malignant"),
                  ("b1", "S1", "P1", "immune"),
                  ("b2", "S1", "P1", "immune")])
    with pytest.raises(DataValidationError, match="conflicting labels"):
        validate_label_identity(df, dual_label_rule="reject")
    rep = validate_label_identity(df, dual_label_rule="exclude")
    assert rep.n_conflicting_labels == 1


def test_align_labels_exact_selected_match():
    df = _labels([("b1", "S1", "P1", "malignant"),
                  ("b2", "S1", "P1", "immune")])
    aligned, info = align_labels(df, ["b2", "b1", "b9"])
    assert aligned["barcode"].tolist() == ["b2", "b1", "b9"]
    assert aligned["label"].tolist()[:2] == ["immune", "malignant"]
    assert pd.isna(aligned["label"].iloc[2])
    assert info["n_matrix_cells_unlabeled"] == 1


def test_align_labels_missing_selected_rejected():
    """A SELECTED LABEL barcode missing from the matrix is an error
    (extra unlabeled matrix cells are not)."""
    df = _labels([("b1", "S1", "P1", "immune"),
                  ("bZ", "S1", "P1", "immune")])
    with pytest.raises(DataValidationError, match="absent from matrix"):
        align_labels(df, ["b1", "bX"])


def test_align_labels_duplicate_matrix_barcodes_rejected():
    df = _labels([("b1", "S1", "P1", "immune")])
    with pytest.raises(DataValidationError, match="duplicate matrix"):
        align_labels(df, ["b1", "b1"])


def test_zero_prefixed_donor_ids_survive():
    df = _labels([("b1", "S1", "007", "immune")])
    aligned, _ = align_labels(df, ["b1"])
    assert aligned["patient"].tolist() == ["007"]
    assert isinstance(aligned["patient"].iloc[0], str)
