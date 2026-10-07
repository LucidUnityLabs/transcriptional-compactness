"""Recompute a committed meta-analysis summary from its per-patient table.

Independent table-to-summary arithmetic verification (audit section 8.4 /
release acceptance 8.8): never invokes the experiment's run.py, never
touches raw data. Example:

    python tools/verify_outputs.py \
      --table experiments/META_patient_level/per_patient_delta.csv \
      --result experiments/META_patient_level/results.json \
      --pointer /overall_patient_level

Verifies arithmetic ONLY: not raw data, labels, the graph metric, or
patient independence. Patient IDs are kept as STRINGS (zero-prefixed
donor IDs must never be parsed as numbers). For multi-arm tables, pass
--cohort (or the arm's own table) and the actual JSON pointer for that
arm — do not mix arms just because a CSV has the same column names.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from lib.runner import load_json_strict          # noqa: E402
from lib.verification import (VerificationError,  # noqa: E402
                              verify_patient_meta)


def resolve_pointer(document, pointer):
    if not pointer.startswith("/"):
        raise ValueError("pointer must start with /")
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        document = document[int(key)] if isinstance(document, list) \
            else document[key]
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pointer", default="/dl_meta")
    parser.add_argument("--cohort")
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    rows = []
    for record in records:
        if args.cohort and record.get("cohort") != args.cohort:
            continue
        donor = record["patient_id"]
        if "cohort" in record:
            donor = f"{record['cohort']}:{donor}"
        rows.append({"patient_id": donor,
                     "delta": float(record["delta"]),
                     "se": float(record["se"])})
    reported = resolve_pointer(load_json_strict(args.result), args.pointer)
    try:
        outcome = verify_patient_meta(rows, reported)
    except VerificationError as exc:
        raise SystemExit(f"ARITHMETIC_MISMATCH: {exc}") from exc
    print(outcome)
    print("Only table-to-summary arithmetic was verified, not raw data, "
          "labels, graph metric or independence.")


if __name__ == "__main__":
    main()
