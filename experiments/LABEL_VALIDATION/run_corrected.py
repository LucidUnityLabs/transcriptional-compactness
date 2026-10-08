"""LABEL_VALIDATION — CORRECTED annotation-scheme sensitivity (TC-1).

Separately identified corrected-method path; the committed ``run.py`` is
PRESERVED UNMODIFIED as the historical analysis (results.json,
label_validation_summary.txt stay the committed record).

Corrected relative to run.py:
* community annotation uses BINARY kNN connectivity (``weight=None``) —
  the historical call fed Euclidean DISTANCE into Louvain's weight
  argument, so longer edges bound communities more strongly (C11);
* the "fixed geometry" decomposition actually freezes geometry
  (C12): one cell universe, one embedding, one graph, one measured-edge
  draw via ``preprocess.freeze_geometry``; only label masks are applied
  afterwards (``frozen_label_contrasts``), with no per-arm resampling,
  no per-axis rescaling, no graph rebuild, and explicit overlap
  rejection;
* the robustness verdict requires the COMPLETE scheme set and validates
  every effect; failed schemes produce a structured inconclusive verdict
  instead of being filtered out of a recovered list (C10);
* ``published_delta`` literals are recorded as DESCRIPTION ONLY: they are
  not treated as a reproduction gate.  A real baseline gate must read the
  corresponding full-precision prior artifact and verify data/method
  equivalence (the corrected META artifact provides that once data are
  present);
* full-graph OR metric, verified transport, exact Cliff placements as
  everywhere else (C01-C07).

Usage: python3 experiments/LABEL_VALIDATION/run_corrected.py [--update]
Exits 2 with a blocked artifact if the GEO inputs are absent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from lib import METHOD_VERSION, cohort, numerics  # noqa: E402
from lib.cache import ResultCache, digest_of_inputs  # noqa: E402
from lib.preprocess import (community_annotation_unweighted,  # noqa: E402
                            freeze_geometry, frozen_label_contrasts)
from lib.runner import run_or_block  # noqa: E402
from lib.verification import label_verdict  # noqa: E402

DATA = HERE.parent.parent / "data"
CHEN_DIR = DATA / "GSE176031_chen"
OLALEKAN_DIR = DATA / "GSE147082"
SEED = 20260508
SCHEMES = ["original", "alt1_broad", "alt2_cluster"]

#: historical headline values, recorded as DESCRIPTION ONLY (audit C10:
#: rounded literals are not a reproducibility certificate)
PUBLISHED_DESCRIPTION = {
    "prostate_chen_GSE176031": 0.21,
    "ovarian_olalekan_GSE147082": 0.48,
}

REQUIRED_INPUTS = [
    ("chen_expression", CHEN_DIR,
     "GEO GSE176031 (Chen 2021 prostate) per-sample DGE matrices"),
    ("olalekan_expression", OLALEKAN_DIR,
     "GEO GSE147082 (Olalekan 2021 ovarian) per-patient CSV matrices"),
]


def run():
    # The historical annotation schemes (marker panels) are reproduced
    # exactly as data-selection rules; all downstream numerics are the
    # corrected library.  Community detection uses binary connectivity.
    import importlib.util as _ilu
    import types

    try:
        import ot  # noqa: F401
    except ImportError:
        stub = types.ModuleType("ot")
        stub.emd2 = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("corrected driver must not call legacy OT"))
        sys.modules.setdefault("ot", stub)
    spec = _ilu.spec_from_file_location("lv_legacy", HERE / "run.py")
    legacy = _ilu.module_from_spec(spec)
    import ast
    source = (HERE / "run.py").read_text().replace("np.float32", "np.float64")
    source = source.replace("X_counts = X_counts[:, keep]\n    cell_pids", "X_counts = X_counts[:, keep]\n    full_library = X_counts.sum(axis=0)\n    cell_pids")
    source = source.replace("libsize = X_counts.sum(axis=0)\n    libsize[libsize == 0] = 1", "libsize = full_library")
    source = source.replace('if patient is None:\n            continue', 'if patient is None or "org" in f.lower():\n            continue')
    exec(compile(ast.parse(source), str(HERE / "run.py"), "exec"), legacy.__dict__)
    # The annotators resolve these globals at call time: wire the
    # corrected routines before ANY scheme is evaluated (C11, R04).
    legacy.build_knn_graph = numerics.knn_graph
    legacy.louvain_communities = lambda G, seed: community_annotation_unweighted(G, seed)[0]

    out = {}
    for cohort_key, loader_name, annotator_name, data_dir in (
            ("prostate_chen_GSE176031", "load_chen", "annotate_chen",
             CHEN_DIR),
            ("ovarian_olalekan_GSE147082", "load_olalekan",
             "annotate_olalekan", OLALEKAN_DIR)):
        loader = getattr(legacy, loader_name)
        annotator = getattr(legacy, annotator_name)
        expr_log, meta = loader()

        scheme_results = {}
        masks = {}
        for scheme in SCHEMES:
            mal_mask, comp_mask, calls = annotator(expr_log, meta, scheme)
            masks[scheme] = (mal_mask, comp_mask)

        # ---- whole-pipeline contrasts per scheme (own embedding each,
        #      as in the historical design; accurately named as such)
        for scheme in SCHEMES:
            mal_mask, comp_mask = masks[scheme]
            X = expr_log.values.T.astype(np.float64)
            pat = np.array(["_"] * X.shape[0], dtype=object)
            is_mal = np.asarray(mal_mask, dtype=bool)
            is_comp = np.asarray(comp_mask, dtype=bool)
            # pooled (not patient-stratified) contrast, mirroring the
            # historical 600+600 subsample design but through the
            # corrected sampler with a single pseudo-patient stratum
            cache = ResultCache(
                HERE / "cache", f"LABEL_VALIDATION/{cohort_key}/{scheme}")
            inputs_digest = digest_of_inputs([data_dir])
            kdf, facts = cohort.compute_cohort_kappa(
                X.T, pat, is_mal, is_comp,
                k=15, alpha=0.5, n_edges=4000, seed=SEED,
                min_per_group=30, cap_per_group=600, total_cap=1200,
                cache=cache, cache_inputs_digest=inputs_digest)
            a = kdf.loc[kdf.is_mal & (kdf.status == "measured"),
                        "kappa"].values
            b = kdf.loc[~kdf.is_mal & (kdf.status == "measured"),
                        "kappa"].values
            res = numerics.cliff_placements(a, b)
            from scipy.stats import ttest_ind

            tt = ttest_ind(a, b, equal_var=False)
            scheme_results[scheme] = {
                "ok": bool(a.size >= 5 and b.size >= 5),
                "n_mal": int(a.size), "n_comp": int(b.size),
                "delta": float(res.delta),
                "se": res.se, "se_estimable": bool(res.estimable),
                "cliff_status": res.status,
                "p_welch": float(tt.pvalue),
                "pipeline_facts": {k: facts[k] for k in
                                   ("graph_nodes", "graph_edges",
                                    "or_edges_computed")},
            }

        # ---- corrected decomposition: geometry frozen ONCE (C12)
        X_all, pca_facts = cohort.hvg_pca(expr_log.values.astype(np.float64))
        pat_all = np.array(["all"] * X_all.shape[0], dtype=object)
        frozen = freeze_geometry(X_all, k=15, alpha=0.5, n_edges=4000,
                                 seed=SEED,
                                 method_note="single global embedding; "
                                             "labels applied afterwards "
                                             "only")
        frozen_masks = {s: masks[s] for s in SCHEMES}
        decomp = frozen_label_contrasts(frozen, frozen_masks,
                                        overlap_rule="exclude_overlap")

        # ---- complete-set, non-vacuous verdict (C10)
        verdict = label_verdict(scheme_results, SCHEMES)

        out[cohort_key] = {
            "published_delta_description_only":
                PUBLISHED_DESCRIPTION[cohort_key],
            "schemes": scheme_results,
            "frozen_geometry_decomposition": decomp,
            "frozen_geometry_pca": pca_facts,
            "verdict": verdict,
        }

    return {
        "method_version": METHOD_VERSION,
        "analysis": "corrected annotation-scheme sensitivity (TC-1); "
                    "historical path preserved in run.py",
        "seed": SEED,
        "community_weight_definition":
            "binary kNN connectivity (louvain weight=None; C11 fix)",
        "cohorts": out,
    }


if __name__ == "__main__":
    run_or_block(
        "LABEL_VALIDATION/run_corrected.py",
        REQUIRED_INPUTS,
        HERE / "results_corrected.json",
        run,
        note="Corrected-method rerun wiring (audit TC-1). GEO cohorts are "
             "not redistributed; see REPRODUCING.md.",
    )
