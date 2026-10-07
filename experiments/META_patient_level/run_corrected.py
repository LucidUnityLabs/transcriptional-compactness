"""META_patient_level — CORRECTED patient-level meta-analysis (TC-1).

Separately identified corrected-method path; the committed ``run.py`` is
PRESERVED UNMODIFIED as the historical analysis (results.json,
per_patient_delta.csv stay the committed record).

Corrected relative to run.py:
* full-graph Ollivier-Ricci metric (C01), identity-safe kNN (C02),
  verified transport (C04), coverage-aware summaries (C05);
* exact Cliff placement variances with explicit non-estimable status —
  NO 1e-12 variance floor (C06); non-estimable donors are excluded from
  inverse-variance pooling through a recorded exclusion list;
* validating DL meta/bootstrap — malformed rows raise instead of being
  silently filtered; log-space tail probabilities recorded (C07);
* content-addressed method-bound caches; the legacy
  ``cache/<cohort>_v2_kappa.npz`` filename-keyed caches are never
  consulted and can never serve this computation (R01);
* cohort loaders are reused from the historical script ONLY as GEO
  parsers; every downstream numerical stage is the corrected library.
  An ``ot`` stub is installed when POT is absent so the historical module
  imports without the production solver (its loader functions do not use
  it; the corrected pipeline never calls the legacy OR routine);
* reruns verify the FULL payload against the committed artifact
  (complete-field gate) unless ``--update``.

NOTE (report C01): the corrected pooled delta is EXPECTED to differ from
the committed +0.51 because the transport metric changed.  The committed
numbers are historical; do not force agreement, and do not weaken
tolerances to manufacture it.

Usage: python3 experiments/META_patient_level/run_corrected.py [--update]
Exits 2 with a blocked artifact if the GEO inputs are absent.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from lib import METHOD_VERSION, cohort  # noqa: E402
from lib.cache import ResultCache, digest_of_inputs  # noqa: E402
from lib.runner import run_or_block  # noqa: E402

DATA = HERE.parent.parent / "data"
SEED = 20260507

#: (cohort, loader name, required input path, provenance)
COHORTS = [
    ("Tirosh_melanoma", "load_tirosh", DATA / "GSE72056_melanoma.txt",
     "GEO GSE72056 (Tirosh 2016 melanoma) processed expression matrix"),
    ("Darmanis_GBM", "load_darmanis", DATA / "GSE84465_GBM.csv",
     "GEO GSE84465 (Darmanis GBM) expression; companion meta file "
     "GSE84465_meta.txt also required"),
    ("Puram_HNSCC", "load_puram", DATA / "GSE103322_HNSCC.txt",
     "GEO GSE103322 (Puram HNSCC) processed expression matrix"),
    ("Peng_PDAC", "load_pdac",
     DATA / "GSE111672_PDAC-A-indrop-filtered-expMat.txt.gz",
     "GEO GSE111672 (Peng/Moncada PDAC) PDAC-A and PDAC-B matrices"),
    ("Chen_prostate", "load_chen", DATA / "GSE176031_chen",
     "GEO GSE176031 (Chen prostate) per-sample DGE matrices directory"),
    ("Olalekan_ovarian", "load_olalekan", DATA / "GSE147082",
     "GEO GSE147082 (Olalekan ovarian HGSOC) per-patient CSV directory"),
]

REQUIRED_INPUTS = [(name, path, prov) for name, _l, path, prov in COHORTS]


def _load_legacy_module():
    """Import the historical run.py for its GEO loader functions only."""
    try:
        import ot  # noqa: F401
    except ImportError:
        stub = types.ModuleType("ot")

        def _unused(*a, **k):
            raise RuntimeError(
                "legacy ot.emd2 called in corrected driver — the "
                "corrected pipeline must use lib.numerics")

        stub.emd2 = _unused
        sys.modules.setdefault("ot", stub)
    spec = importlib.util.spec_from_file_location(
        "meta_legacy_loaders", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run():
    legacy = _load_legacy_module()
    cache_root = HERE / "cache"
    cohort_out = {}
    patient_tables = []
    for cohort_name, loader_name, input_path, provenance in COHORTS:
        loader = getattr(legacy, loader_name)
        X_log, patient, is_mal, is_comp = loader()
        inputs_digest = digest_of_inputs([input_path])
        cache = ResultCache(cache_root, f"META/{cohort_name}/corrected")
        kdf, facts = cohort.compute_cohort_kappa(
            X_log, patient, is_mal, is_comp,
            k=15, alpha=0.5, n_edges=6000, seed=SEED,
            min_per_group=10, cap_per_group=60, total_cap=3500,
            cache=cache, cache_inputs_digest=inputs_digest)
        tab, estimable, exclusions = cohort.per_patient_contrasts(
            kdf, min_cells=10)
        meta = cohort.pool_cohort(tab, estimable, label=cohort_name,
                                  seed=SEED, B=1000)
        tab = tab.copy()
        tab["cohort"] = cohort_name
        patient_tables.append(tab)
        cohort_out[cohort_name] = {
            "provenance": provenance,
            "n_patients_estimable": int(len(estimable)),
            "exclusions": exclusions,
            "dl_meta": meta,
            "pipeline_facts": {k: facts[k] for k in
                               ("graph_nodes", "graph_edges",
                                "or_edges_computed")},
        }
    all_pp = pd.concat([t for t in patient_tables if len(t)],
                       ignore_index=True)
    estimable_all = all_pp[all_pp.get("se_estimable", False) == True]  # noqa: E712
    overall = cohort.pool_cohort(
        all_pp, estimable_all, label="OVERALL_patient_level_corrected",
        seed=SEED + 1, B=1000)
    cohort_deltas = [cohort_out[c]["dl_meta"] for c, _l, _p, _prov in COHORTS
                     if cohort_out[c]["dl_meta"]["k"] > 0]
    cohort_as_study = cohort.numerics.dl_meta(
        [m["delta"] for m in cohort_deltas],
        [m["se"] for m in cohort_deltas],
        label="OVERALL_cohort_level_corrected")

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

    all_pp.to_csv(HERE / "per_patient_delta_corrected.csv", index=False)
    return {
        "method_version": METHOD_VERSION,
        "analysis": "corrected patient-level meta-analysis (TC-1); "
                    "historical path preserved in run.py",
        "seed": SEED,
        "note_on_divergence": "the corrected pooled delta is EXPECTED to "
                              "differ from the committed historical "
                              "results.json (different transport metric); "
                              "manuscript numbers must be regenerated from "
                              "this artifact after review",
        "n_patients_total": int(len(estimable_all)),
        "per_cohort": _clean(cohort_out),
        "overall_patient_level": _clean(overall),
        "overall_cohort_level": _clean(cohort_as_study),
    }


if __name__ == "__main__":
    run_or_block(
        "META_patient_level/run_corrected.py",
        REQUIRED_INPUTS,
        HERE / "results_corrected.json",
        run,
        note="Corrected-method rerun wiring (audit TC-1). Discovery GEO "
             "cohorts are not redistributed; see REPRODUCING.md data "
             "layout. Per-cohort blocked detail is listed in "
             "missing_inputs.",
    )
