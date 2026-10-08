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
Exits 2 with an attempt-only blocked diagnostic if inputs are absent.
Accepted CSV/JSON/cache sets are immutable and read through lib.publication;
the historical flat files are retained and never replaced.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
IMPORTED_DRIVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path.insert(0, str(HERE.parent))

from lib import METHOD_VERSION, cohort, numerics  # noqa: E402
from lib.bio_io import (DataValidationError, align_labels,  # noqa: E402
                        read_mtx_selected, validate_label_identity)
from lib.cache import ResultCache, digest_of_inputs  # noqa: E402
from lib.runner import run_or_block  # noqa: E402
from lib.verification import direction_counts, protocol_decision

DATA = HERE.parent.parent / "data"
GEO = DATA / "GSE161529"
LAB_DIR = GEO / "labels"
SAMPLES_DIR = GEO / "samples"
FEATURES = SAMPLES_DIR / "GSE161529_features.tsv.gz"
MANIFEST = SAMPLES_DIR / "MANIFEST.json"
SOURCE_EXCLUSIONS = DATA.parent / 'config/GSE161529_source_exclusions.json'
SOURCE_COVERAGE = DATA.parent / 'config/GSE161529_raw_label_coverage.json'

SEED = 20260507
VARIANT_ORDER = ("excluded", "dual_mal", "dual_immune")

REQUIRED_INPUTS = [
    ('whole_raw_label_coverage', SOURCE_COVERAGE,
     'tools/verify_breast_coverage.py --write after source acquisitions and full-label extraction'),
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
            genes.append(line.split("\t")[0].rstrip("\n"))
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
    source_exclusions = []
    if SOURCE_EXCLUSIONS.exists():
        exclusions = json.loads(SOURCE_EXCLUSIONS.read_text())['samples']
        for sample, evidence in exclusions.items():
            actual = hashlib.sha256((LAB_DIR / cfg['labels_file']).read_bytes()).hexdigest()
            if actual != evidence['label_file_sha256'][cfg['labels_file']]:
                raise DataValidationError('source exclusion label content changed; review required')
            raw = SAMPLES_DIR / evidence['raw_barcode_file']
            if hashlib.sha256(raw.read_bytes()).hexdigest() != evidence['raw_barcode_sha256']:
                raise DataValidationError('source exclusion raw barcode content changed; review required')
            removed = lab[lab['sample'] == sample]
            if set(removed.patient.astype(str)) != {evidence['donor']}:
                raise DataValidationError('source exclusion donor identity differs')
            source_exclusions.append({'sample':sample,'donor':evidence['donor'],'n_labels_removed':len(removed),'reason':evidence['reason'],'required_to_restore':evidence['required_to_restore']})
            lab = lab[lab['sample'] != sample]
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
        'source_exclusions': source_exclusions,
        "n_dual_labeled_barcodes": len(conflicted),
        "variant": variant,
        "identity_report": {k: v for k, v in rep.__dict__.items()
                            if k != "problems"},
    }


def load_counts_strict(labels, mapping, detection_labels=None):
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
    detected = np.zeros(n_genes, dtype=np.int64)
    cell_barcodes = []
    per_sample = {}
    wanted = {}
    for sample, sub in labels.groupby("sample"):
        wanted[sample] = set(sub["barcode"])
    sample_universe = set(wanted) | (set(detection_labels["sample"]) if detection_labels is not None else set())
    for sample in sorted(sample_universe):
        stem = mapping[sample]["stem"]
        bcs = read_barcodes(stem)
        pos = {b: i for i, b in enumerate(bcs)}
        local_wanted = {b[len(sample) + 1:]: b for b in wanted.get(sample, set())}
        keep_local = {loc: pos[loc] + 1 for loc in local_wanted
                      if loc in pos}          # 1-based MTX columns
        n_absent = len(local_wanted) - len(keep_local)
        if n_absent:
            raise DataValidationError(
                f"{sample}: {n_absent} selected label barcodes absent "
                "from the matrix — selected labelled cells must match "
                "exactly")
        if not keep_local and detection_labels is None:
            continue
        cols = [keep_local[loc] for loc in
                sorted(keep_local, key=keep_local.get)]
        detection_cols = None
        if detection_labels is not None:
            universe = detection_labels[detection_labels["sample"] == sample]["barcode"]
            local = [b[len(sample) + 1:] for b in universe]
            if set(local) - set(pos):
                raise DataValidationError(f"{sample}: detection-universe barcodes absent from matrix")
            detection_cols = [pos[b] + 1 for b in local]
        sel = read_mtx_selected(SAMPLES_DIR / f"{stem}-matrix.mtx.gz",
                                cols, detection_cols=detection_cols, expected_n_cols=len(bcs))
        if sel.n_cols != len(bcs):
            raise DataValidationError(f"{stem}: MTX columns != complete barcode count")
        if detection_cols is not None:
            detected += sel.gene_detection
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
    if detection_labels is not None:
        info["gene_detection_full_label_universe"] = detected.tolist()
    return counts, np.array(gene_names, dtype=object), \
        aligned, info, per_sample


def run_variant(arm, variant, mapping, cache_root, output_dir):
    labels, identity = resolve_labels(arm, variant)
    selected, sampling_audit = cohort.stratified_sample_minima_first(
        labels["patient"].astype(str).values, (labels["label"] == "malignant").values,
        (labels["label"] == ARMS[arm]["corrected_label"]).values,
        min_per_group=10, cap_per_group=60, total_cap=3500, seed=SEED)
    selected_labels = labels.iloc[selected].reset_index(drop=True)
    counts, gene_names, aligned, align_info, per_sample = \
        load_counts_strict(selected_labels, mapping, detection_labels=labels)
    # log1p-CPM(1e6) float64; genes detected in >=5 cells
    full_library = counts.sum(axis=1)
    detection = np.asarray(align_info.pop("gene_detection_full_label_universe"))
    gene_keep = np.flatnonzero(detection >= 5)
    if np.any(full_library <= 0) or not len(gene_keep):
        raise DataValidationError("empty library or no eligible detected genes")
    X_log = np.log1p(counts[:, gene_keep].T / full_library * 1e6)
    normalization_facts = {"denominator": "complete raw library before gene filtering",
        "gene_filter_universe": "all arm/variant labelled cells; streamed and coordinate-deduplicated",
        "n_genes_kept": len(gene_keep), "preload_sampling": sampling_audit}

    patient = aligned["patient"].astype(str).values
    is_mal = (aligned["label"] == "malignant").values
    is_comp = (aligned["label"] ==
               ARMS[arm]["corrected_label"]).values

    label_files = sorted(p.name for p in LAB_DIR.glob("cells_*.tsv.gz"))
    digest_paths = [MANIFEST, FEATURES, LAB_DIR / ARMS[arm]["labels_file"]]
    if SOURCE_EXCLUSIONS.exists(): digest_paths.append(SOURCE_EXCLUSIONS)
    if SOURCE_COVERAGE.exists(): digest_paths.append(SOURCE_COVERAGE)
    if (LAB_DIR / 'LABELS_MANIFEST_CORRECTED.json').exists():
        digest_paths += [LAB_DIR / 'LABELS_MANIFEST_CORRECTED.json',
                         GEO / 'GSE161529_family.soft.gz',
                         DATA.parent / 'config/GSE161529_sample_aliases.json']
    digest_paths += [LAB_DIR / 'LABELS_MANIFEST_CORRECTED_V2.json']
    digest_paths += [SAMPLES_DIR / f"{mapping[s]['stem']}-matrix.mtx.gz"
                     for s in sorted(mapping)]
    digest_paths += [SAMPLES_DIR / f"{mapping[s]['stem']}-barcodes.tsv.gz"
                     for s in sorted(mapping)]
    from lib.publication import AttemptCache
    cache = AttemptCache(cache_root, f"HELDOUT_GSE161529/{arm}/{variant}")
    inputs_digest = digest_of_inputs(digest_paths)
    kdf, facts = cohort.compute_cohort_kappa(
        X_log, patient, is_mal, is_comp,
        k=15, alpha=0.5, n_edges=4000, seed=SEED,
        min_per_group=10, cap_per_group=60, total_cap=3500,
        cache=cache, cache_inputs_digest=inputs_digest)
    tab, estimable, exclusions = cohort.per_patient_contrasts(kdf,
                                                              min_cells=10)
    cache.receipt(Path(output_dir)/f'cache_{arm}_{variant}.json')
    table_path = Path(output_dir) / f'per_patient_{arm}_{variant}_corrected.csv'
    temporary_table = table_path.with_suffix('.csv.tmp')
    estimable.to_csv(temporary_table, index=False)
    temporary_table.replace(table_path)
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
        "selected_input_cell_ids": aligned['barcode'].astype(str).tolist(),
        "selected_input_cell_identity_sha256": hashlib.sha256(json.dumps(aligned['barcode'].astype(str).tolist(), separators=(',',':')).encode()).hexdigest(),
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
        "per_patient_direction": direction_counts(estimable["delta"].values),
        "dl_meta": _clean(meta),
        "pooled_delta": _clean(meta["delta"]),
        "pooled_ci_95": [_clean(meta["ci_lo"]), _clean(meta["ci_hi"])],
        "one_sided_p_delta_gt_0": _clean(meta["one_sided_p_delta_gt_0"]),
        "pipeline_facts": _clean({k: v for k, v in facts.items() if k != "cache_hit_reason"}),
        "normalization": normalization_facts,
        "label_files_seen": label_files,
    }


def run(arms=('primary', 'secondary'), variants=VARIANT_ORDER, *, attempt_dir=None):
    if arms and attempt_dir is None:
        raise ValueError('numerical run requires an isolated publication attempt directory')
    from lib.runner import load_json_strict
    source_coverage = load_json_strict(SOURCE_COVERAGE)
    if source_coverage['status'] != 'all eligible specimens have complete exact barcode coverage':
        raise ValueError('whole breast source coverage gate failed')
    if source_coverage['exclusions_sha256'] != hashlib.sha256(SOURCE_EXCLUSIONS.read_bytes()).hexdigest():
        raise ValueError('breast source exclusion receipt differs from coverage manifest')
    for filename, expected in source_coverage['label_file_sha256'].items():
        if hashlib.sha256((LAB_DIR / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('whole-source coverage label content changed')
    if hashlib.sha256((LAB_DIR / 'LABELS_MANIFEST_CORRECTED.json').read_bytes()).hexdigest() != source_coverage['label_extraction_manifest_sha256']:
        raise ValueError('whole-source coverage extraction receipt changed')
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
    if download_facts['manifest_status'] != 'ok' or download_facts['n_errors']:
        raise ValueError('breast acquisition manifest is incomplete or failed')
    provenance_path = LAB_DIR / 'LABELS_MANIFEST_CORRECTED_V2.json'
    patient_evidence = {'verified': False, 'reason': 'source-backed label extraction receipt absent'}
    if not provenance_path.exists():
        raise ValueError('versioned complete-axes label manifest required')
    if provenance_path.exists():
        from lib.runner import load_json_strict
        import re
        provenance = load_json_strict(provenance_path)
        from lib.axes_provenance import verify_label_axes
        verify_label_axes(GEO, provenance, DATA.parent / 'tools/extract_breast_axes.R')
        if provenance.get('status') != 'ok' or provenance.get('method_version') != METHOD_VERSION:
            raise ValueError('breast label provenance receipt is incomplete')
        expected_objects = {'ERTotal','HER2','TNBC','ERTotalSub','HER2Sub','TNBCSub'}
        if {e['object'] for e in provenance['objects']} != expected_objects:
            raise ValueError('breast label source-object coverage is incomplete')
        crosses = provenance.get('barcode_crosswalks', [])
        expected_crosses = {('ERTotal','ERTotalSub'),('HER2','HER2Sub'),('TNBC','TNBCSub')}
        if {(e['total'],e['subset']) for e in crosses} != expected_crosses or len(crosses) != 3 or any(e['n_epithelial_subset_conflicts'] or e['n_subset_cells_absent_from_total'] for e in crosses):
            raise ValueError('breast complete Total/Sub crosswalk gates failed')
        if set(provenance['uncompressed_sha256']) != {'cells_primary.tsv.gz','cells_secondary.tsv.gz'}:
            raise ValueError('breast label hash coverage is incomplete')
        soft_text = gzip.decompress((GEO / 'GSE161529_family.soft.gz').read_bytes()).decode()
        donors = {}
        for block in soft_text.split('^SAMPLE = ')[1:]:
            donor = re.search(r'!Sample_characteristics_ch1 = patient: (.+)', block)
            if donor: donors[block.splitlines()[0].strip()] = donor.group(1).strip()
        for filename, expected in provenance['uncompressed_sha256'].items():
            h = hashlib.sha256()
            with gzip.open(LAB_DIR / filename, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f'breast source-derived label content changed: {filename}')
            labelled = pd.read_csv(LAB_DIR / filename, sep='\t', usecols=['sample','patient'], dtype=str).drop_duplicates()
            for row in labelled.itertuples():
                if donors.get(mapping[row.sample]['gsm']) != row.patient:
                    raise ValueError(f'breast donor identity differs from actual GEO source: {row.sample}')
        patient_evidence = {'verified': True, 'source': provenance['donor_source'],
                            'label_manifest_sha256': hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                            'scope': 'exact source GEO donor join; inferred labels and shared-graph dependence remain limitations'}

    cache_root = Path(attempt_dir) / "cache" if attempt_dir is not None else None
    arms_out = {}
    for arm in arms:
        arms_out[arm] = {
            v: run_variant(arm, v, mapping, cache_root, attempt_dir)
            for v in variants}

    return {
        "method_version": METHOD_VERSION,
        "cohort": "GSE161529 (Pal et al. 2021 breast cancer atlas)",
        "analysis": "corrected primary held-out validation (TC-1); "
                    "historical path preserved in run.py/run_sensitivity.py",
        "seed": SEED,
        "variants": variants,
        "download": download_facts,
        'whole_source_coverage': source_coverage,
        "loaded_driver_sha256": IMPORTED_DRIVER_SHA256,
        "execution_selection": {'arms': list(arms), 'variants': list(variants), 'cell_cap': 3500, 'measured_edge_budget': 4000},
        'source_coverage_status': 'partial: whole ER_0001 source identity mismatch excluded' if SOURCE_EXCLUSIONS.exists() else 'complete acquisition-mapped source coverage',
        "patient_unit_verified": patient_evidence['verified'],
        "patient_unit_evidence": patient_evidence,
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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['all', *ARMS], default='all')
    parser.add_argument('--variant', choices=['all', *VARIANT_ORDER], default='all')
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()
    arms = tuple(ARMS) if args.arm == 'all' else (args.arm,)
    variants = VARIANT_ORDER if args.variant == 'all' else (args.variant,)
    suffix = '' if args.arm == args.variant == 'all' else f'_{args.arm}_{args.variant}'
    from lib.publication import run_breast_attempt
    run_breast_attempt(
        "HELDOUT_GSE161529/run_corrected.py", REQUIRED_INPUTS,
        HERE / f"results{suffix}_corrected.json",
        lambda attempt: run(arms, variants, attempt_dir=attempt), arms, variants,
        argv=sys.argv[1:], seed_cache=HERE/'cache')
