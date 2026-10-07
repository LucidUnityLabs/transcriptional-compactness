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
def read_umi_tsv_selected(path, cell_ids, expect_genes=None):
    """Stream a genes-x-cells UMI TSV strictly, retaining requested cells.

    File shape: the header row is ``<label>\\t<cell_1>\\t...\\t<cell_n>``
    and every data row is ``<gene>\\t<count_1>\\t...\\t<count_n>``.

    Contract: cell ids in the header are nonempty and unique; every row
    has EXACTLY the header column count; every token in EVERY count
    column (selected or not) is a nonnegative integer; gene ids are
    nonempty and unique; all requested cells exist exactly once; the
    returned matrix columns follow the REQUESTED order.

    Returns ``(counts, gene_names)`` where ``counts`` is a float64 dense
    array of shape (n_genes, n_requested).
    """
    cell_ids = [str(c) for c in cell_ids]
    if not cell_ids:
        raise DataValidationError("empty requested cell list")
    req = list(cell_ids)
    if len(set(req)) != len(req):
        raise DataValidationError("duplicate requested cell ids")

    with _open_maybe_gzip(path) as f:
        lines = _read_all_lines(f, path)

    if not lines:
        raise DataValidationError(f"{path}: empty file")
    header = lines[0].rstrip("\n").split("\t")
    if len(header) < 2:
        raise DataValidationError(f"{path}: header has no cell columns")
    file_cells = header[1:]
    if any(not c for c in file_cells):
        raise DataValidationError(f"{path}: empty cell id in header")
    if len(set(file_cells)) != len(file_cells):
        from collections import Counter
        dup = [c for c, n in Counter(file_cells).items() if n > 1]
        raise DataValidationError(
            f"{path}: duplicate cell ids in header: {sorted(dup)[:5]}")
    col_pos = {c: i for i, c in enumerate(file_cells)}
    missing = [c for c in req if c not in col_pos]
    if missing:
        raise DataValidationError(
            f"{path}: requested cells absent from header: {missing[:5]} "
            f"({len(missing)} of {len(req)} missing)")
    keep = [col_pos[c] for c in req]           # requested order, exact
    n_cols = len(file_cells)

    n_rows = len(lines) - 1
    if expect_genes is not None and n_rows != expect_genes:
        raise DataValidationError(
            f"{path}: expected {expect_genes} gene rows, found {n_rows}")
    out = np.zeros((n_rows, len(req)), dtype=np.float64)
    seen_genes = set()
    gene_order = []
    for row_no, raw in enumerate(lines[1:], start=2):
        line = raw.rstrip("\n")
        if line == "":
            raise DataValidationError(f"{path}: empty row at line {row_no}")
        parts = line.split("\t")
        if len(parts) != n_cols + 1:
            raise DataValidationError(
                f"{path}: line {row_no} has {len(parts) - 1} count "
                f"columns, expected {n_cols} (parsing is strict — no "
                "padding or truncation)")
        gene = parts[0]
        if not gene:
            raise DataValidationError(
                f"{path}: empty gene id at line {row_no}")
        if gene in seen_genes:
            raise DataValidationError(
                f"{path}: duplicate gene id {gene!r} at line {row_no}")
        seen_genes.add(gene)
        gene_order.append(gene)
        for tok in parts[1:]:
            if tok == "":
                raise DataValidationError(
                    f"{path}: empty count token at line {row_no}")
            # integer tokens only; reject '1.0', '1e3', '-1', 'nan'
            body = tok[1:] if tok[:1] == "+" else tok
            if not body.isdigit():
                raise DataValidationError(
                    f"{path}: non-integer count token {tok!r} at line "
                    f"{row_no} (lossless decimal parsers require an "
                    "explicit method version; add fixtures first)")
            if tok[:1] == "-":
                raise DataValidationError(
                    f"{path}: negative count {tok!r} at line {row_no}")
        out[row_no - 2, :] = [int(parts[1 + c]) for c in keep]

    return out, gene_order


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


def read_mtx_selected(path, wanted_cols, banner_required=True):
    """Strict streaming Matrix Market coordinate reader, selected columns.

    ``wanted_cols`` are 1-based column indices in the exact order the
    caller requires.  The full coordinate stream is validated (banner
    format, declared dimensions and nnz, one-based bounds, nonnegative
    integer counts, arbitrary whitespace); duplicate coordinate entries are
    summed explicitly as sparse-coordinate entries, with the summed total
    capped below 2^53 so float64 accumulation cannot silently round.
    """
    wanted = [int(c) for c in wanted_cols]
    if not wanted:
        raise DataValidationError("empty requested column list")
    if len(set(wanted)) != len(wanted):
        raise DataValidationError("duplicate requested columns")

    with _open_maybe_gzip(path) as f:
        lines = _read_all_lines(f, path)

    idx = 0
    if idx >= len(lines):
        raise DataValidationError(f"{path}: empty file")
    banner = lines[idx].strip()
    idx += 1
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
    while idx < len(lines) and lines[idx].lstrip().startswith("%"):
        idx += 1
    if idx >= len(lines):
        raise DataValidationError(f"{path}: missing dimensions line")
    dims = lines[idx].split()
    idx += 1
    if len(dims) != 3:
        raise DataValidationError(
            f"{path}: dimensions line must be 'rows cols nnz', got "
            f"{lines[idx - 1]!r}")
    try:
        n_rows, n_cols, nnz = (int(x) for x in dims)
    except ValueError as exc:
        raise DataValidationError(
            f"{path}: non-integer dimensions {dims!r}") from exc
    if n_rows < 0 or n_cols < 0 or nnz < 0:
        raise DataValidationError(f"{path}: negative dimensions {dims!r}")

    for c in wanted:
        if not (1 <= c <= n_cols):
            raise DataValidationError(
                f"{path}: requested column {c} out of range 1..{n_cols}")

    pos = {c: i for i, c in enumerate(wanted)}
    # int64 holds exact integers far beyond the 2^53 float64-exactness
    # guard below, so duplicate-entry summing stays exact.
    acc = np.zeros((len(wanted), n_rows), dtype=np.int64)
    n_dup = 0
    n_read = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
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
                       n_duplicate_entries_summed=n_dup)


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
