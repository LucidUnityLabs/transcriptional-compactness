"""HELDOUT_GSE161529 — CORRECTED primary held-out validation (TC-1).

Separately identified corrected-method path for the breast cohort; the
committed ``run.py`` and ``run_sensitivity.py`` are PRESERVED UNMODIFIED
as the historical analysis (their numbers stay in results.json /
results_sensitivity.json).

Corrected relative to run.py / run_sensitivity.py:
* strict per-sample Matrix Market reader (audit D02: banner, declared
  dims/nnz, bounds, duplicate-entry summing, exact requested order; the
  legacy pandas reader heuristically dropped the dims row and FILTERED
  out-of-range coordinates);
* matched-cell/label identity enforced BEFORE computation (D03):
  validate_label_identity + align_labels replace the non-binding
  matched_cell_audit observation and the production ``assert``;
* full-graph Ollivier-Ricci metric (C01), identity-safe kNN (C02),
  verified transport (C04), coverage-aware summaries (C05), exact Cliff
  placements without variance floors (C06), validating DL/bootstrap
  (C07);
* the three dual-label handling variants run under the CORRECTED method
  with their own artifact; the corrected 'excluded' variant is NOT
  compared against the legacy numerical golden (that would compare two
  different computations — audit C09).  Legacy exact reproduction is the
  job of the preserved run_sensitivity.py gate;
* download integrity facts come from acquisition-time VERIFIED checks;
  no unconditional ``gzip_verified: true`` (D07);
* content-addressed method-bound caches (R01);
* reruns verify the FULL payload against the committed artifact
  (complete-field gate) unless ``--update``.

Secondary-arm labelling caveat (D05): the historical
``normal_epithelial`` label means 'epithelial cluster not called
malignant by the tumor-block rule'.  This driver renames the contrast
``epithelial_not_called_malignant`` and marks the preregistered
normal-luminal endpoint UNAVAILABLE pending a reviewed per-cell
annotation table with provenance (labels/cells_luminal_reviewed.tsv.gz).

Usage: python3 experiments/HELDOUT_GSE161529/run_corrected.py [--update]
Exits 2 with a blocked artifact if the GEO/figshare inputs are absent.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from lib import METHOD_VERSION, cohort, numerics  # noqa: E402
from lib.bio_io import (DataValidationError, align_labels,  # noqa: E402
                        read_mtx_selected, validate_label_identity)
from lib.cache import ResultCache, digest_of_inputs  # noqa: E402
from lib.runner import run_or_block  # noqa: E402

DATA = HERE.parent.parent / "data"
GEO = DATA / "GSE161529"
LAB_DIR = GEO / "labels"
SAMPLES_DIR = GEO / "samples"
FEATURES = SAMPLES_DIR / "GSE161529_features.tsv.gz"
MANIFEST = SAMPLES_DIR / "MANIFEST.json"

SEED = 20260507
VARIANT_ORDER = ("excluded", "dual_mal", "dual_immune")

REQUIRED_INPUTS = [
    ("labels_primary", LAB_DIR / "cells_primary.tsv.gz",
     "Chen et al. 2022 (Sci Data) companion labels, produced by "
     "extract_labels.py from the figshare Seurat objects "
     "(figshare deposit 10.6084/m9.figshare.17058077)"),
    ("labels_secondary", LAB_DIR / "cells_secondary.tsv.gz",
     "same companion deposit, secondary arm labels"),
    ("features", FEATURES,
     "GSE161529_features.tsv.gz from GEO series GSE161529 supplementary"),
    ("manifest", MANIFEST,
     "samples/MANIFEST.json written by the acquisition stage "
     "(download_corrected.py or the historical download.py) mapping "
     "label samples to GSM supplementary files"),
]

ARMS = {
    "primary": dict(labels_file="cells_primary.tsv.gz",
                    comparator_label="immune",
                    corrected_label="immune"),
    "secondary": dict(
        labels_file="cells_secondary.tsv.gz",
        comparator_label="normal_epithelial",
        corrected_label="epithelial_not_called_malignant"),
}


def load_features():
    genes = []
    with gzip.open(FEATURES, "rt") as f:
        for line in f:
            genes.append(line.split("\t")[1].rstrip("\n"))
    if len(set(genes)) != len(genes):
        raise DataValidationError("duplicate feature ids in features.tsv")
    return genes


def read_barcodes(stem):
    with gzip.open(SAMPLES_DIR / f"{stem}-barcodes.tsv.gz", "rt") as f:
        bcs = [line.strip() for line in f]
    if len(set(bcs)) != len(bcs):
        raise DataValidationError(f"duplicate barcodes for {stem}")
    return bcs


def resolve_labels(arm, variant):
    """Load labels and apply the dual-label rule EXPLICITLY (D03)."""
    cfg = ARMS[arm]
    lab = pd.read_csv(LAB_DIR / cfg["labels_file"], sep="\t",
                      dtype={"patient": str})
    lab = lab[lab["label"].isin(["malignant", cfg["comparator_label"]])]
    nlab = lab.groupby("barcode")["label"].nunique()
    conflicted = set(nlab[nlab > 1].index)
    is_dual = lab["barcode"].isin(conflicted)
    if variant == "excluded":
        lab = lab[~is_dual]
    elif variant == "dual_mal":
        lab = lab[~(is_dual & (lab["label"] != "malignant"))]
    elif variant == "dual_immune":
        lab = lab[~(is_dual & (lab["label"] == "malignant"))]
    else:
        raise ValueError(variant)
    rep = validate_label_identity(lab, dual_label_rule="exclude")
    lab = lab.drop_duplicates("barcode", keep="first")
    lab["label"] = lab["label"].map(
        {"malignant": "malignant",
         cfg["comparator_label"]: cfg["corrected_label"]})
    return lab.reset_index(drop=True), {
        "n_dual_labeled_barcodes": len(conflicted),
        "variant": variant,
        "identity_report": {k: v for k, v in rep.__dict__.items()
                            if k != "problems"},
    }


def load_counts_strict(labels, mapping):
    """Strict per-sample MTX loading for the selected labelled cells."""
    label_samples = set(labels["sample"].unique())
    unmapped = sorted(label_samples - set(mapping))
    if unmapped:
        raise DataValidationError(
            f"label samples absent from the acquisition mapping: "
            f"{unmapped} — every label-table sample must be checked "
            "(audit D03); an unjoinable sample is an explicit error, "
            "not a silent drop")
    gene_names = load_features()
    n_genes = len(gene_names)
    blocks = []
    cell_barcodes = []
    per_sample = {}
    wanted = {}
    for sample, sub in labels.groupby("sample"):
        wanted[sample] = set(sub["barcode"])
    for sample in sorted(wanted):
        stem = mapping[sample]["stem"]
        bcs = read_barcodes(stem)
        pos = {b: i for i, b in enumerate(bcs)}
        local_wanted = {b[len(sample) + 1:]: b for b in wanted[sample]}
        keep_local = {loc: pos[loc] + 1 for loc in local_wanted
                      if loc in pos}          # 1-based MTX columns
        n_absent = len(local_wanted) - len(keep_local)
        if n_absent:
            raise DataValidationError(
                f"{sample}: {n_absent} selected label barcodes absent "
                "from the matrix — selected labelled cells must match "
                "exactly")
        if not keep_local:
            continue
        cols = [keep_local[loc] for loc in
                sorted(keep_local, key=keep_local.get)]
        sel = read_mtx_selected(SAMPLES_DIR / f"{stem}-matrix.mtx.gz",
                                cols)
        if sel.counts.shape[1] != n_genes:
            raise DataValidationError(
                f"{stem}: MTX rows {sel.counts.shape[1]} != features "
                f"{n_genes}")
        blocks.append(sel.counts)
        cell_barcodes.extend(local_wanted[bcs[c - 1]] for c in cols)
        per_sample[sample] = {
            "n_cells": len(cols), "nnz_declared": sel.n_entries_declared,
            "nnz_read": sel.n_entries_read,
            "n_duplicate_entries_summed":
                sel.n_duplicate_entries_summed,
        }
    counts = np.concatenate(blocks, axis=0) if blocks else \
        np.zeros((0, n_genes))
    aligned, info = align_labels(labels, cell_barcodes,
                                 dual_label_rule="exclude")
    return counts, np.array(gene_names, dtype=object), \
        aligned, info, per_sample


def run_variant(arm, variant, mapping, cache_root):
    labels, identity = resolve_labels(arm, variant)
    counts, gene_names, aligned, align_info, per_sample = \
        load_counts_strict(labels, mapping)
    # log1p-CPM(1e6) float64; genes detected in >=5 cells
    expressed = (counts > 0).sum(axis=0)
    gene_keep = np.flatnonzero(expressed >= 5)
    counts = counts[:, gene_keep]
    libsize = counts.sum(axis=1, keepdims=True)
    libsize = np.where(libsize == 0, 1.0, libsize)
    X_log = np.log1p(counts / libsize * 1e6).T          # genes x cells

    patient = aligned["patient"].astype(str).values
    is_mal = (aligned["label"] == "malignant").values
    is_comp = (aligned["label"] ==
               ARMS[arm]["corrected_label"]).values

    label_files = sorted(p.name for p in LAB_DIR.glob("cells_*.tsv.gz"))
    digest_paths = [MANIFEST, FEATURES, LAB_DIR / ARMS[arm]["labels_file"]]
    digest_paths += [SAMPLES_DIR / f"{mapping[s]['stem']}-matrix.mtx.gz"
                     for s in sorted(mapping)]
    cache = ResultCache(cache_root, f"HELDOUT_GSE161529/{arm}/{variant}")
    inputs_digest = digest_of_inputs(digest_paths)
    kdf, facts = cohort.compute_cohort_kappa(
        X_log, patient, is_mal, is_comp,
        k=15, alpha=0.5, n_edges=4000, seed=SEED,
        min_per_group=10, cap_per_group=60, total_cap=3500,
        cache=cache, cache_inputs_digest=inputs_digest)
    tab, estimable, exclusions = cohort.per_patient_contrasts(kdf,
                                                              min_cells=10)
    meta = cohort.pool_cohort(tab, estimable,
                              label=f"{arm}/{variant}", seed=SEED, B=1000)
    n_pos = int((estimable["delta"] > 0).sum()) if len(estimable) else 0

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    return {
        "comparator": ARMS[arm]["corrected_label"],
        "variant": variant,
        "label_identity": _clean(identity),
        "label_alignment": _clean({k: v for k, v in align_info.items()
                                   if k != "identity_report"}),
        "per_sample_strict_load": _clean(per_sample),
        "n_cells_measured": int((kdf["status"] == "measured").sum()),
        "n_patients_passing_filter": int(len(estimable)),
        "exclusions": _clean(exclusions),
        "per_patient": {
            r.patient_id: _clean({k: getattr(r, k) for k in
                                  ("n_malignant", "n_comparator", "delta",
                                   "se", "se_estimable", "cliff_status")})
            for r in estimable.itertuples()},
        "per_patient_direction": {
            "n_delta_positive": n_pos,
            "n_delta_negative": int(len(estimable) - n_pos),
            "n_total": int(len(estimable))},
        "dl_meta": _clean(meta),
        "pooled_delta": _clean(meta["delta"]),
        "pooled_ci_95": [_clean(meta["ci_lo"]), _clean(meta["ci_hi"])],
        "one_sided_p_delta_gt_0": _clean(meta["one_sided_p_delta_gt_0"]),
        "pipeline_facts": _clean(facts),
        "label_files_seen": label_files,
    }


def run():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    mapping = manifest["mapping"]

    # D07: report the ACTUAL acquisition checks recorded in the manifest
    # (never an unconditional success boolean)
    download_facts = {
        "manifest_status": manifest.get("manifest", {}).get(
            "status", manifest.get("status", "unknown")),
        "n_files": len(manifest.get("manifest", {}).get(
            "files", manifest.get("files", []))),
        "n_errors": len(manifest.get("manifest", {}).get(
            "errors", manifest.get("errors", []))),
        "integrity": "checks recorded at acquisition time; see "
                     "samples/MANIFEST.json (no unconditional "
                     "gzip_verified flag)",
    }

    cache_root = HERE / "cache"
    arms_out = {}
    for arm in ("primary", "secondary"):
        arms_out[arm] = {
            v: run_variant(arm, v, mapping, cache_root)
            for v in VARIANT_ORDER}

    return {
        "method_version": METHOD_VERSION,
        "cohort": "GSE161529 (Pal et al. 2021 breast cancer atlas)",
        "analysis": "corrected primary held-out validation (TC-1); "
                    "historical path preserved in run.py/run_sensitivity.py",
        "seed": SEED,
        "variants": VARIANT_ORDER,
        "download": download_facts,
        "luminal_endpoint_status": {
            "status": "unavailable",
            "reason": "the preregistered normal-luminal comparator "
                      "requires a reviewed per-cell annotation table with "
                      "explicit normal/luminal status and provenance; "
                      "'epithelial cluster not called malignant' is not "
                      "positive luminal identity (audit D05). Provide "
                      "labels/cells_luminal_reviewed.tsv.gz and extend "
                      "ARMS to enable it.",
        },
        "comparators": arms_out,
    }


if __name__ == "__main__":
    run_or_block(
        "HELDOUT_GSE161529/run_corrected.py",
        REQUIRED_INPUTS,
        HERE / "results_corrected.json",
        run,
        note="Corrected-method rerun wiring (audit TC-1). Labels are "
             "produced by extract_labels.py from the figshare Seurat "
             "objects; per-sample matrices by the acquisition stage; "
             "features from GEO. None are redistributed — see "
             "REPRODUCING.md.",
    )
