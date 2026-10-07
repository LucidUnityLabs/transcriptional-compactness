"""HELDOUT_GSE131907 — CORRECTED secondary held-out validation (TC-1).

Separately identified corrected-method path for the lung cohort.  The
committed ``run.py`` is PRESERVED UNMODIFIED as the historical analysis
(local-subgraph metric, position-zero self-drop, permissive UMI reader);
this driver is the corrected method and every artifact it writes binds to
``lib.METHOD_VERSION``.

Corrected relative to run.py:
* strict UMI reader (audit D01: no padding/truncation/silent zeroing);
* label/matrix identity gates before computation (D03);
* full-graph Ollivier-Ricci metric (C01), identity-safe kNN (C02),
  verified transport that fails closed (C04), coverage-aware per-cell
  summaries (C05), exact Cliff placements without variance floors (C06),
  validating DL/bootstrap (C07);
* the unit of inference is labelled SPECIMEN-level unless a reviewed
  specimen->donor map is supplied: ``Sample`` is a proxy (58 specimens /
  44 patients), so the corrected artifact records
  ``patient_unit_verified: false`` and labels the inference
  specimen-level/exploratory (C08) instead of claiming patient-level
  validation;
* content-addressed, method-bound cache (R01): the legacy
  ``cache/HELDOUT_GSE131907_kappa.npz`` filename-keyed cache is never
  consulted here and can never serve this computation;
* reruns verify the FULL payload against the committed artifact
  (complete-field gate, C09-style) unless ``--update``.

Usage: python3 experiments/HELDOUT_GSE161529/../HELDOUT_GSE131907/run_corrected.py [--update]
Exits 2 with a blocked artifact if the GEO inputs are absent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from lib import METHOD_VERSION, cohort, numerics  # noqa: E402
from lib.bio_io import (DataValidationError, align_labels,  # noqa: E402
                        read_umi_tsv_selected,
                        validate_label_identity)
from lib.cache import ResultCache, digest_of_inputs  # noqa: E402
from lib.runner import run_or_block  # noqa: E402

DATA = HERE.parent.parent / "data"
KIM_DIR = DATA / "GSE131907_kim_nsclc"
KIM_MTX = KIM_DIR / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
KIM_ANN = KIM_DIR / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"

SEED = 20260507
MAL_SUBTYPES = {"tS1", "tS2", "tS3", "Malignant cells"}
IMMUNE_REFINED = {"T/NK cells", "Myeloid cells", "B lymphocytes",
                  "MAST cells"}

REQUIRED_INPUTS = [
    ("umi_matrix", KIM_MTX,
     "GSE131907 (Kim et al. 2020 metastatic LUAD atlas) raw UMI matrix "
     "genes x cells TSV.gz; GEO series GSE131907 supplementary file "
     "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"),
    ("annotation", KIM_ANN,
     "GSE131907 original-author cell annotation "
     "GSE131907_Lung_Cancer_cell_annotation.txt.gz (same GEO series)"),
]

#: Reviewed specimen->donor map (C08).  Absent until the authoritative
#: metadata is acquired and reviewed; then it must be committed as a
#: machine-readable file {specimen: donor} with source evidence.
DONOR_MAP_PATH = DATA / "GSE131907_kim_nsclc" / "specimen_donor_map.tsv"


def load_annotation():
    ann = pd.read_csv(KIM_ANN, sep="\t", index_col=0, dtype=str)
    if ann.index.duplicated().any():
        dup = ann.index[ann.index.duplicated()][:5].tolist()
        raise DataValidationError(
            f"duplicate cell barcodes in annotation: {dup}")
    is_mal = ((ann["Cell_type.refined"] == "Epithelial cells")
              & ann["Cell_subtype"].isin(MAL_SUBTYPES)).values
    is_comp = ann["Cell_type.refined"].isin(IMMUNE_REFINED).values
    labels = pd.DataFrame({
        "barcode": ann.index.astype(str),
        "sample": ann["Sample"].astype(str).values,
        "patient": ann["Sample"].astype(str).values,  # specimen proxy
        "label": np.where(is_mal, "malignant",
                          np.where(is_comp, "immune", "other"))})
    return labels[labels["label"] != "other"].reset_index(drop=True), ann


def run(cache_root=None):
    labels, _ann = load_annotation()
    # D03: identity structure is a gate, not an observation
    validate_label_identity(labels, dual_label_rule="reject")

    barcodes = labels["barcode"].tolist()
    counts, gene_names = read_umi_tsv_selected(KIM_MTX, barcodes)
    if counts.shape[1] != len(barcodes):
        raise DataValidationError("strict reader returned wrong width")
    # log1p-CPM(1e6) in float64; gene filter >=5 cells (declared protocol)
    expressed = (counts > 0).sum(axis=1)
    gene_keep = np.flatnonzero(expressed >= 5)
    counts = counts[gene_keep]
    gene_names = [gene_names[i] for i in gene_keep]
    lib = counts.sum(axis=0, keepdims=True)
    lib = np.where(lib == 0, 1.0, lib)
    X_log = np.log1p(counts / lib * 1e6)

    specimen = labels["sample"].values
    is_mal = (labels["label"] == "malignant").values
    is_comp = (labels["label"] == "immune").values

    donor_map = None
    donor_verified, donor_evidence = cohort.donor_unit_flag(donor_map)
    unit = "specimen (Sample proxy — specimen-level/exploratory inference)"
    if donor_verified:
        unit = "patient (verified specimen->donor map applied)"

    cache = ResultCache(cache_root or HERE / "cache",
                        "HELDOUT_GSE131907/corrected")
    inputs_digest = digest_of_inputs([KIM_MTX, KIM_ANN])
    kdf, facts = cohort.compute_cohort_kappa(
        X_log, specimen, is_mal, is_comp,
        k=15, alpha=0.5, n_edges=4000, seed=SEED,
        min_per_group=10, cap_per_group=60, total_cap=3500,
        cache=cache, cache_inputs_digest=inputs_digest)

    tab, estimable, exclusions = cohort.per_patient_contrasts(kdf,
                                                              min_cells=10)
    meta = cohort.pool_cohort(tab, estimable, label="GSE131907_corrected",
                              seed=SEED, B=1000)

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

    measured = kdf[kdf["status"] == "measured"]
    n_pos = int((estimable["delta"] > 0).sum()) if len(estimable) else 0
    return {
        "method_version": METHOD_VERSION,
        "metric": facts["metric"],
        "cohort": "GSE131907 (Kim et al. 2020 metastatic LUAD atlas)",
        "analysis": "corrected secondary held-out validation (TC-1); "
                    "historical path preserved in run.py",
        "seed": SEED,
        "unit_of_inference": unit,
        "patient_unit_verified": donor_verified,
        "patient_unit_evidence": donor_evidence,
        "donor_map_required": str(DONOR_MAP_PATH),
        "design": {
            "knn_k": 15, "or_alpha": 0.5, "n_edges": 4000,
            "min_cells_per_group": 10,
            "sampling": "minimum-budget-first stratified, audited "
                        "exclusions (R05)",
            "normalization": "log1p-CPM(1e6) float64; genes detected in "
                             ">=5 cells",
            "pipeline_facts": _clean(facts),
        },
        "n_cells_measured": int(len(measured)),
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
        "decision_rule_inputs": {
            "pooled_delta": _clean(meta["delta"]),
            "one_sided_p": _clean(meta["one_sided_p_delta_gt_0"]),
            "rule": "SUPPORTED if pooled delta > 0 and one-sided p < 0.05",
        },
    }


if __name__ == "__main__":
    run_or_block(
        "HELDOUT_GSE131907/run_corrected.py",
        REQUIRED_INPUTS,
        HERE / "results_corrected.json",
        run,
        note="Corrected-method rerun wiring (audit TC-1). GEO inputs are "
             "not redistributed with the repository; place them under "
             "data/GSE131907_kim_nsclc/ per REPRODUCING.md and rerun.",
    )
