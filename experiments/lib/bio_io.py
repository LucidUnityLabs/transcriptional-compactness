"""Strict expression/matrix readers and label/axis identity gates (TC-1).

Fixes relative to the historical loaders:

* UMI TSV reading is strict (audit D01): exact column count on every row,
  unique nonempty IDs, validation of ALL count tokens (including
  unselected columns), nonnegative-integer tokens only, complete
  requested-cell coverage, preserved requested order, gzip EOF/CRC
  verification.  Corruption raises with the offending row number — it is
  never padded, truncated or silently zeroed.
* Matrix Market reading is strict (audit D02): banner, declared
  dimensions/nnz, one-based bounds, nonnegative integer counts, duplicate
  coordinate entries summed explicitly (with a 2^53 total-count guard),
  exact requested-column order; out-of-range coordinates are errors, not
  filtered observations.
* Label/matrix identity is a gate, not an observation (audit D03):
  :func:`validate_label_identity` + :func:`align_labels` enforce unique
  resolved IDs, barcode/specimen/donor consistency and exact selected-cell
  alignment before any calculation.
* RNA axes are aligned by name (audit D04): :func:`align_rna_axes` requires
  unique IDs and matching axis sets between count-matrix Dimnames and
  metadata, and returns explicit reindexers.  A row-count assertion can
  never establish axis identity.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np

#: Conservative cap on summed duplicate-entry counts: float64 integers are
#: exact only to 2^53; at or above this bound summing could silently round.
COUNT_SUM_LIMIT = 2 ** 53


class DataValidationError(ValueError):
    """Strict reader/gate failure naming the offending record."""


def _open_maybe_gzip(path, mode="rt"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode)
    return open(p, mode)


def _read_all_lines(handle, path):
    """Read to EOF so gzip verifies CRC/truncation before parsing results
    are trusted; re-raise with the file name."""
    try:
        return handle.readlines()
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise DataValidationError(
            f"{path}: gzip stream failed before EOF (truncated or corrupt: "
            f"{exc})") from exc


# ---------------------------------------------------------------------------
# Strict UMI TSV reader (D01)
# ---------------------------------------------------------------------------
def read_umi_tsv_selected(path, cell_ids, expect_genes=None, detection_cell_ids=None):
    """Validate every count, stream to gzip EOF, retain requested columns.

    Optional detection_cell_ids records gene detection over a larger named
    universe without allocating that universe's dense matrix. Return order
    always follows cell_ids. All accepted counts are exact float64 integers.
    """
    import re
    req = [str(c) for c in cell_ids]
    if not req or len(set(req)) != len(req):
        raise DataValidationError("empty or duplicate requested cell ids")
    rows, genes, detected = [], [], []
    seen = set()
    try:
        with _open_maybe_gzip(path) as f:
            first = f.readline()
            if not first:
                raise DataValidationError(f"{path}: empty file")
            cells = first.rstrip("\n").split("\t")[1:]
            if not cells or any(not c for c in cells):
                raise DataValidationError(f"{path}: header has no cells or empty cell id")
            if len(set(cells)) != len(cells):
                raise DataValidationError(f"{path}: duplicate cell ids in header")
            pos = {c: i for i, c in enumerate(cells)}
            missing = set(req) - set(pos)
            if missing:
                raise DataValidationError(f"{path}: requested cells absent from header: {sorted(missing)[:5]}")
            keep = np.array([pos[c] for c in req])
            detection = None
            if detection_cell_ids is not None:
                ids = list(detection_cell_ids)
                if len(set(ids)) != len(ids) or set(ids) - set(pos):
                    raise DataValidationError("invalid detection cell universe")
                detection = np.array([pos[c] for c in ids])
            for row_no, raw in enumerate(f, 2):
                line = raw.rstrip("\n")
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise DataValidationError(f"{path}: empty row or missing counts at line {row_no}")
                gene, counts_text = parts
                if not gene:
                    raise DataValidationError(f"{path}: empty gene id at line {row_no}")
                if gene in seen:
                    raise DataValidationError(f"{path}: duplicate gene id at line {row_no}")
                seen.add(gene)
                if counts_text.count("\t") + 1 != len(cells):
                    raise DataValidationError(f"{path}: count columns mismatch at line {row_no}; expected {len(cells)}")
                if re.fullmatch(r"[+]?[0-9]+(?:\t[+]?[0-9]+)*", counts_text, flags=re.ASCII) is None:
                    raise DataValidationError(f"{path}: non-integer or empty count token at line {row_no}")
                values = np.fromstring(counts_text, dtype=np.float64, sep="\t")
                if len(values) != len(cells) or not np.all(np.isfinite(values)) or np.any(values >= COUNT_SUM_LIMIT):
                    raise DataValidationError(f"{path}: count outside exact float64 integer range at line {row_no}")
                rows.append(values[keep])
                genes.append(gene)
                if detection is not None:
                    detected.append(int(np.count_nonzero(values[detection])))
    except (OSError, EOFError) as exc:
        raise DataValidationError(f"{path}: gzip stream failed before EOF: {exc}") from exc
    if expect_genes is not None and len(genes) != expect_genes:
        raise DataValidationError(f"{path}: expected {expect_genes} gene rows, found {len(genes)}")
    out = np.asarray(rows, dtype=np.float64).reshape(len(genes), len(req))
    return (out, genes, np.asarray(detected)) if detection_cell_ids is not None else (out, genes)


# ---------------------------------------------------------------------------
# Strict Matrix Market reader (D02)
# ---------------------------------------------------------------------------
@dataclass
class MTXSelected:
    counts: np.ndarray        # dense float64 (n_requested_cols, n_rows)
    n_rows: int
    n_cols: int
    n_entries_declared: int
    n_entries_read: int
    n_duplicate_entries_summed: int
    gene_detection: np.ndarray | None = None


def read_mtx_selected(path, wanted_cols, banner_required=True, detection_cols=None, expected_n_cols=None):
    """Strict streaming Matrix Market coordinate reader, selected columns.

    ``wanted_cols`` are 1-based column indices in the exact order the
    caller requires.  The full coordinate stream is validated (banner
    format, declared dimensions and nnz, one-based bounds, nonnegative
    integer counts, arbitrary whitespace); duplicate coordinate entries are
    summed explicitly as sparse-coordinate entries, with the summed total
    capped below 2^53 so float64 accumulation cannot silently round.
    """
    wanted = [int(c) for c in wanted_cols]
    if not wanted and detection_cols is None:
        raise DataValidationError("empty requested column list")
    if len(set(wanted)) != len(wanted):
        raise DataValidationError("duplicate requested columns")

    with _open_maybe_gzip(path) as f:
        first = f.readline()
        if not first:
            raise DataValidationError(f"{path}: empty file")
        banner = first.strip()
        if banner_required:
            parts = banner.split()
            if len(parts) < 5 or parts[0] != "%%MatrixMarket" \
                    or parts[1] != "matrix" or parts[2] != "coordinate":
                raise DataValidationError(
                    f"{path}: bad MatrixMarket banner {banner!r}")
            if parts[3] != "integer":
                raise DataValidationError(
                    f"{path}: field {parts[3]!r} != 'integer' (the strict "
                    "reference reader requires integer counts)")
            if parts[4] != "general":
                raise DataValidationError(
                    f"{path}: symmetry {parts[4]!r} != 'general'")

        # skip comment lines
        dimension_line = f.readline()
        while dimension_line.lstrip().startswith("%"):
            dimension_line = f.readline()
        if not dimension_line:
            raise DataValidationError(f"{path}: missing dimensions line")
        dims = dimension_line.split()
        if len(dims) != 3:
            raise DataValidationError(
                f"{path}: dimensions line must be 'rows cols nnz', got "
                f"{dimension_line!r}")
        try:
            n_rows, n_cols, nnz = (int(x) for x in dims)
        except ValueError as exc:
            raise DataValidationError(
                f"{path}: non-integer dimensions {dims!r}") from exc
        if n_rows < 0 or n_cols < 0 or nnz < 0:
            raise DataValidationError(f"{path}: negative dimensions {dims!r}")

        if expected_n_cols is not None and n_cols != expected_n_cols:
            raise DataValidationError(
                f"{path}: MTX columns {n_cols} != complete barcode count {expected_n_cols}")

        for c in wanted:
            if not (1 <= c <= n_cols):
                raise DataValidationError(
                    f"{path}: requested column {c} out of range 1..{n_cols}")

        detection_set = set(detection_cols) if detection_cols is not None else set()
        if any(not 1 <= c <= n_cols for c in detection_set):
            raise DataValidationError("detection column out of range")
        detected = np.zeros(n_rows, dtype=np.int64) if detection_cols is not None else None
        seen_detection = set()
        pos = {c: i for i, c in enumerate(wanted)}
        # int64 holds exact integers far beyond the 2^53 float64-exactness
        # guard below, so duplicate-entry summing stays exact.
        acc = np.zeros((len(wanted), n_rows), dtype=np.int64)
        n_dup = 0
        n_read = 0
        for raw in f:
            line = raw.strip()
            if line == "":
                raise DataValidationError(
                    f"{path}: blank data line at entry {n_read + 1} (the "
                    "declared nnz must be matched exactly)")
            parts = line.split()
            if len(parts) != 3:
                raise DataValidationError(
                    f"{path}: data line {n_read + 1} must have 3 fields, got "
                    f"{line!r}")
            try:
                r, c, v = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError as exc:
                raise DataValidationError(
                    f"{path}: non-integer coordinate/count {line!r}") from exc
            if not (1 <= r <= n_rows) or not (1 <= c <= n_cols):
                raise DataValidationError(
                    f"{path}: coordinate ({r}, {c}) out of bounds "
                    f"1..{n_rows}, 1..{n_cols}")
            if v < 0:
                raise DataValidationError(
                    f"{path}: negative count {v} at ({r}, {c})")
            n_read += 1
            if n_read > nnz:
                raise DataValidationError(
                    f"{path}: more data lines than declared nnz={nnz}")
            if v > 0 and c in detection_set:
                coordinate = (c - 1) * n_rows + r - 1
                if coordinate not in seen_detection:
                    detected[r - 1] += 1
                    seen_detection.add(coordinate)
            j = pos.get(c)
            if j is None:
                continue
            cur = int(acc[j, r - 1])
            if cur != 0:
                n_dup += 1
            total = cur + v
            if total >= COUNT_SUM_LIMIT:
                raise DataValidationError(
                    f"{path}: summed duplicate count at ({r}, {c}) reaches "
                    f"2^53 ({total}); float64 cannot represent it exactly — "
                    "split the entry set or use exact integer storage")
            acc[j, r - 1] = total
        if n_read != nnz:
            raise DataValidationError(
                f"{path}: read {n_read} data lines, declared nnz={nnz}")

    counts = acc.astype(np.float64)
    return MTXSelected(counts=counts, n_rows=n_rows, n_cols=n_cols,
                       n_entries_declared=nnz, n_entries_read=n_read,
                       n_duplicate_entries_summed=n_dup, gene_detection=detected)


# ---------------------------------------------------------------------------
# RNA axis alignment (D04)
# ---------------------------------------------------------------------------
def align_rna_axes(count_dimnames_cells, feature_names, metadata_cells):
    """Align named RNA axes to metadata order; refuse shape-only matches.

    ``count_dimnames_cells``: cell ids from the count matrix Dimnames.
    ``feature_names``: feature ids from the count matrix.
    ``metadata_cells``: barcode order the caller will score against.

    Requires unique ids in both cell lists, unique feature ids, and SET
    equality between count-matrix cells and metadata cells; returns the
    permutation reindexing counts into metadata order.  A row-count match
    alone can never establish this (audit D04).
    """
    cm = [str(x) for x in count_dimnames_cells]
    md = [str(x) for x in metadata_cells]
    if len(set(cm)) != len(cm):
        raise DataValidationError("duplicate cell ids in count Dimnames")
    if len(set(md)) != len(md):
        raise DataValidationError("duplicate cell ids in metadata")
    feats = [str(x) for x in feature_names]
    if len(set(feats)) != len(feats):
        raise DataValidationError("duplicate feature ids")
    if set(cm) != set(md):
        only_counts = sorted(set(cm) - set(md))[:5]
        only_meta = sorted(set(md) - set(cm))[:5]
        raise DataValidationError(
            f"count-matrix cells do not match metadata cells "
            f"(counts-only {only_counts}, metadata-only {only_meta})")
    pos = {c: i for i, c in enumerate(cm)}
    return [pos[c] for c in md], feats


def validate_marker_scores(scores, n_cells, panel_name):
    """A missing marker panel or no finite scores must yield an explicit
    unknown/error state, not an empty-ranking index or -inf propagation."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise DataValidationError(
            f"marker panel {panel_name!r} produced no scores")
    if arr.shape[0] != n_cells:
        raise DataValidationError(
            f"marker panel {panel_name!r}: {arr.shape[0]} scores for "
            f"{n_cells} cells")
    if not np.any(np.isfinite(arr)):
        raise DataValidationError(
            f"marker panel {panel_name!r}: no finite scores (report an "
            "explicit unknown state instead of z-scoring -inf)")
    return arr


# ---------------------------------------------------------------------------
# Label identity gates (D03)
# ---------------------------------------------------------------------------
@dataclass
class LabelIdentityReport:
    n_rows: int
    n_unique_barcodes: int
    n_duplicate_barcode_rows: int
    n_barcodes_multiple_specimens: int
    n_barcodes_multiple_donors: int
    n_conflicting_labels: int
    problems: list


def validate_label_identity(labels, dual_label_rule="reject",
                            require_columns=("barcode", "sample", "patient",
                                             "label")):
    """Validate label-table identity structure before any computation.

    * required columns exist;
    * a barcode never maps to two specimens or two donors (rejects; prefix
      conventions are checked, not assumed);
    * duplicate (barcode, label-value) rows collapse; a barcode with TWO
      DISTINCT labels is a conflict handled by ``dual_label_rule``:
      ``"reject"`` raises (analysis must stop and the conflict be
      reported), ``"exclude"`` returns the conflict counts for an explicit
      exclusion rule applied by the caller.
    """
    df = labels
    for col in require_columns:
        if col not in df.columns:
            raise DataValidationError(f"label table missing column {col!r}")
    problems = []
    bc_spec = df.groupby("barcode")["sample"].nunique()
    n_multi_spec = int((bc_spec > 1).sum())
    if n_multi_spec:
        bad = bc_spec[bc_spec > 1].index[:5].tolist()
        problems.append(f"{n_multi_spec} barcodes in multiple specimens "
                        f"(e.g. {bad})")
    bc_don = df.groupby("barcode")["patient"].nunique()
    n_multi_don = int((bc_don > 1).sum())
    if n_multi_don:
        bad = bc_don[bc_don > 1].index[:5].tolist()
        problems.append(f"{n_multi_don} barcodes in multiple donors "
                        f"(e.g. {bad})")
    lab_nunique = df.groupby("barcode")["label"].nunique()
    n_conf = int((lab_nunique > 1).sum())
    if n_conf and dual_label_rule == "reject":
        bad = lab_nunique[lab_nunique > 1].index[:5].tolist()
        raise DataValidationError(
            f"{n_conf} barcodes carry conflicting labels (e.g. {bad}); "
            "apply an explicit reviewed dual-label rule (exclude/assign) "
            "before analysis — identity conflicts are not resolvable by "
            "silence")
    problems.append(f"{n_conf} dual-labeled barcodes"
                    if n_conf else "no dual-labeled barcodes")
    if n_multi_spec or n_multi_don:
        raise DataValidationError("; ".join(problems))
    return LabelIdentityReport(
        n_rows=len(df),
        n_unique_barcodes=int(df["barcode"].nunique()),
        n_duplicate_barcode_rows=int(len(df) - df["barcode"].nunique()),
        n_barcodes_multiple_specimens=n_multi_spec,
        n_barcodes_multiple_donors=n_multi_don,
        n_conflicting_labels=n_conf,
        problems=problems)


def align_labels(labels, matrix_barcodes, dual_label_rule="exclude"):
    """Exact one-to-one join of labels onto matrix cells.

    * matrix barcodes must be unique (validated BEFORE building a position
      dictionary);
    * every label sample must be checked (not only samples already
      selected);
    * the requirement is that SELECTED LABELED CELLS match exactly: extra
      unlabeled matrix cells are permitted (recorded), but a selected label
      barcode missing from the matrix is an error;
    * returns the label rows aligned to ``matrix_barcodes`` order with NaN
      rows for unlabeled matrix cells.
    """
    mb = [str(b) for b in matrix_barcodes]
    if len(set(mb)) != len(mb):
        from collections import Counter
        dup = [b for b, c in Counter(mb).items() if c > 1][:5]
        raise DataValidationError(f"duplicate matrix barcodes {dup}")
    rep = validate_label_identity(labels, dual_label_rule=dual_label_rule)
    if dual_label_rule == "exclude":
        keep = labels.groupby("barcode")["label"].transform(
            lambda x: x.nunique() == 1)
        labels = labels[keep]
    lab_unique = labels.drop_duplicates("barcode", keep="first")
    indexed = lab_unique.set_index("barcode")
    missing = sorted(set(indexed.index) - set(mb))
    if missing:
        raise DataValidationError(
            f"{len(missing)} label barcodes absent from matrix "
            f"(e.g. {missing[:5]}) — the selected labeled cells must "
            "match exactly")
    aligned = indexed.reindex(mb)
    n_unlabeled = int(aligned["label"].isna().sum())
    return aligned.reset_index(), {"n_matrix_cells_unlabeled": n_unlabeled,
                                   "identity_report": rep.__dict__}
