"""HELDOUT_GSE161529 — CORRECTED acquisition stage (TC-1).

Separately identified corrected path; the committed ``download.py`` is
PRESERVED UNMODIFIED as the historical record.

Dependency graph fixed relative to download.py (audit D06-D08):
1. this stage requires, and does not bootstrap: the GEO family SOFT
   file (data/GSE161529/GSE161529_family.soft.gz), the label files
   (labels/cells_*.tsv.gz, produced by extract_labels.py from the
   figshare Seurat objects), and the shared features file;
2. sample->GSM mapping uses the supplementary URLs actually present in
   the SOFT file, requires the suffix EXACTLY, requires exactly one
   candidate per sample, and enforces injectivity (D08);
3. downloads are atomic, HTTPS-only, no implicit resume, size-ceiling
   guarded, gzip-verified to EOF, with optional SHA-256 against a
   reviewed lock (D07);
4. the manifest records the checks that were ACTUALLY performed and the
   stage exits nonzero on any required failure — there is no
   unconditional success boolean.

A reviewed data lock (authoritative URLs + hashes, committed) must exist
before release verification is allowed to pass; without it this stage is
ACQUISITION of candidates, not authentication.

Usage: python3 experiments/HELDOUT_GSE161529/download_corrected.py \
           [--lock data/GSE161529/lock.json]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from lib import acquisition  # noqa: E402
from lib.runner import run_or_block, write_results  # noqa: E402

DATA = HERE.parent.parent / "data"
SOFT = DATA / "GSE161529" / "GSE161529_family.soft.gz"
LABELS = DATA / "GSE161529" / "labels"
OUT = DATA / "GSE161529" / "samples"
FEATURES_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/"
                "GSE161529/suppl/GSE161529_features.tsv.gz")

#: Safety ceilings (NOT expected sizes and NOT a verified data lock).
MAX_BYTES = {"matrix": 20 << 30, "barcodes": 64 << 20, "features": 32 << 20}

REQUIRED_INPUTS = [
    ("family_soft", SOFT,
     "GEO GSE161529 family SOFT file (series GSE161529); the SOFT itself "
     "is a discovery input — acquire it from "
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161529/soft/ "
     "and record its hash in the reviewed lock"),
    ("labels_primary", LABELS / "cells_primary.tsv.gz",
     "labels produced by extract_labels.py from the figshare Seurat "
     "objects (figshare 10.6084/m9.figshare.17058077)"),
    ("labels_secondary", LABELS / "cells_secondary.tsv.gz",
     "secondary-arm labels from the same deposit"),
]


def label_samples():
    s = set()
    for lab in ("cells_primary.tsv.gz", "cells_secondary.tsv.gz"):
        import gzip

        with gzip.open(LABELS / lab, "rt") as f:
            f.readline()
            for line in f:
                s.add(line.split("\t")[1])
    return sorted(s)


def run():
    lock = {}
    lock_path = next((a for a in sys.argv[1:] if a.startswith("--lock=")),
                     None)
    if lock_path:
        with open(lock_path.split("=", 1)[1]) as f:
            lock = json.load(f).get("files", {})

    supp = acquisition.parse_soft_supplementary(SOFT)
    samples = label_samples()
    mapping = acquisition.matrix_sources(samples, supp)   # exact/injective

    downloads = []
    errors = []

    feats_dest = OUT / "GSE161529_features.tsv.gz"
    try:
        checks = acquisition.atomic_download(
            FEATURES_URL, feats_dest, sha256=lock.get(feats_dest.name),
            max_bytes=MAX_BYTES["features"])
        downloads.append(checks)
    except acquisition.AcquisitionError as exc:
        errors.append(str(exc))

    for sample in samples:
        rec = mapping[sample]
        stem = rec["stem"]
        base = rec["matrix_url"].rsplit("/", 1)[0]
        for kind, ceiling in (("matrix.mtx.gz", MAX_BYTES["matrix"]),
                              ("barcodes.tsv.gz", MAX_BYTES["barcodes"])):
            name = f"{stem}-{kind}"
            url = f"{base}/{name}"
            dest = OUT / name
            try:
                checks = acquisition.atomic_download(
                    url, dest, sha256=lock.get(name), max_bytes=ceiling)
                checks["sample"] = sample
                checks["gsm"] = rec["gsm"]
                downloads.append(checks)
            except acquisition.AcquisitionError as exc:
                errors.append(f"{sample} {name}: {exc}")

    ok = acquisition.write_success_manifest(OUT / "MANIFEST_CORRECTED.json",
                                            downloads, errors)
    manifest = {"mapping": mapping}
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump({"mapping": mapping,
                   "manifest": {"status": "ok" if ok else "failed",
                                "files": downloads, "errors": errors}},
                  f, indent=2, sort_keys=True)
    if errors:
        raise acquisition.AcquisitionError(
            f"{len(errors)} required downloads failed; see "
            f"{OUT / 'MANIFEST_CORRECTED.json'}")
    return {
        "method_version": "TC-1",
        "stage": "corrected acquisition (audit D06-D08)",
        "n_samples": len(samples),
        "mapping": mapping,
        "n_files_verified": len(downloads),
        "lock_used": bool(lock),
        "lock_note": "hashes were checked against the supplied lock"
                     if lock else
                     "NO reviewed lock supplied: files were verified for "
                     "transport integrity only — acquisition of a "
                     "candidate is not authentication (commit a reviewed "
                     "lock before release verification)",
    }


if __name__ == "__main__":
    run_or_block(
        "HELDOUT_GSE161529/download_corrected.py",
        REQUIRED_INPUTS,
        HERE / "results_acquisition_corrected.json",
        run,
        note="Corrected acquisition wiring (audit TC-1). Requires the "
             "family SOFT file and the extracted label files to already "
             "exist (extract_labels.py runs on the figshare Seurat "
             "objects) — the documented clean-start order is now "
             "explicit instead of impossible.",
    )
