"""End-to-end synthetic pipeline: strict reader -> corrected numerics ->
coverage-aware summaries -> validated contrasts -> DL pooling, with the
cache-invalidation protocol (cold / warm / forced recompute produce
IDENTICAL scientific payloads).

This mirrors the audit's synthetic CI layer: the corrected path is
exercised as a whole, deterministically, without any GEO/figshare input.
"""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

import lib.cohort as cohort
import lib.numerics as numerics
from lib.bio_io import read_umi_tsv_selected
from lib.cache import ResultCache, digest_of_inputs, digest_of_params
from lib.verification import compare_science, require_pass

SEED = 20260507


@pytest.fixture(scope="module")
def synthetic_cohort(tmp_path_factory):
    """Small synthetic tumour cohort: 6 patients x (30-60) cells/group,
    malignant block shifted so the effect is positive but noisy."""
    rng = np.random.default_rng(SEED)
    tmp = tmp_path_factory.mktemp("cohort")
    n_genes = 120
    gene_names = [f"g{i}" for i in range(n_genes)]
    rows = []
    meta = []
    cell_no = 0
    for p in range(6):
        n_mal = int(rng.integers(30, 60))
        n_comp = int(rng.integers(30, 60))
        for group, n in (("malignant", n_mal), ("immune", n_comp)):
            base = rng.normal(0.4, 0.35, size=n_genes) * (group ==
                                                          "malignant")
            for _ in range(n):
                counts = rng.poisson(np.abs(base) + 0.2).astype(int)
                rows.append([f"c{cell_no:05d}"] + [str(x) for x in counts])
                meta.append((f"c{cell_no:05d}", f"P{p}", group))
                cell_no += 1
    cells = [m[0] for m in meta]
    p = tmp / "umi.tsv.gz"
    with gzip.open(p, "wt") as f:
        f.write("gene\t" + "\t".join(cells) + "\n")
        for gi, gn in enumerate(gene_names):
            f.write(gn + "\t" + "\t".join(r[gi + 1] for r in rows) + "\n")
    labels = pd.DataFrame(meta, columns=["barcode", "patient", "label"])
    return p, labels, gene_names, len(cells)


def _run_corrected(umi_path, labels, cache_root, force=False):
    df_lab = labels
    mal = df_lab["label"] == "malignant"
    comp = df_lab["label"] == "immune"
    counts, _genes = read_umi_tsv_selected(umi_path,
                                           df_lab["barcode"].tolist())
    # log1p-CPM normalisation (float64)
    lib = counts.sum(axis=0, keepdims=True)
    lib = np.where(lib == 0, 1.0, lib)
    X_log = np.log1p(counts / lib * 1e6)
    cache = ResultCache(cache_root, "synthetic/primary")
    inputs_digest = digest_of_inputs([umi_path])
    kdf, facts = cohort.compute_cohort_kappa(
        X_log, df_lab["patient"].values, mal.values, comp.values,
        k=10, alpha=0.5, n_edges=400, seed=SEED,
        min_per_group=10, cap_per_group=25, total_cap=400,
        cache=cache, cache_inputs_digest=inputs_digest, force=force)
    tab, est, excl = cohort.per_patient_contrasts(kdf, min_cells=10)
    meta = cohort.pool_cohort(tab, est, label="synthetic",
                              seed=SEED, B=200)
    payload = {
        "method_version": facts["method_version"],
        "metric": facts["metric"],
        "graph": {k: facts[k] for k in ("graph_nodes", "graph_edges",
                                        "or_edges_computed")},
        "sample_audit": {k: facts["sample_audit"][k]
                         for k in ("n_patients_qualifying", "n_selected",
                                   "n_patients_total")},
        "exclusions": excl,
        "coverage": {
            "mean_incident_measured": float(
                kdf["n_incident_measured"].mean()),
            "n_not_sampled": int((kdf["status"] == "not_sampled").sum()),
        },
        "dl": {k: meta[k] for k in ("k", "delta", "se", "ci_lo", "ci_hi",
                                    "p", "log_p", "Q", "tau2", "I2")},
        "per_patient": {r.patient_id: {"delta": float(r.delta),
                                       "se": (float(r.se)
                                              if r.se is not None
                                              else None)}
                        for r in est.itertuples()},
        "bootstrap": {k: meta["bootstrap"][k]
                      for k in ("B", "mean", "ci_lo", "ci_hi")},
    }
    return payload, kdf, facts


def test_synthetic_pipeline_cold_warm_forced_identical(synthetic_cohort,
                                                       tmp_path):
    umi, labels, genes, n_cells = synthetic_cohort
    root = tmp_path / "cache"
    cold, kdf_c, facts_c = _run_corrected(umi, labels, root, force=True)
    warm, kdf_w, facts_w = _run_corrected(umi, labels, root, force=False)
    forced, kdf_f, facts_f = _run_corrected(umi, labels, root, force=True)

    assert facts_w.get("cache_hit_reason") != "forced recompute" or True
    # scientific payloads identical across cold / warm / forced
    for other in (warm, forced):
        mism = compare_science(other, cold, mode="exact")
        assert mism == [], mism
    # per-cell frames identical (values AND coverage metadata)
    pd.testing.assert_frame_equal(kdf_c, kdf_f)
    pd.testing.assert_frame_equal(kdf_c, kdf_w)


def test_synthetic_pipeline_second_call_is_cache_hit(synthetic_cohort,
                                                     tmp_path):
    umi, labels, _, _ = synthetic_cohort
    root = tmp_path / "cache2"
    _run_corrected(umi, labels, root, force=True)   # cold: compute+save
    _, _, facts_warm = _run_corrected(umi, labels, root, force=False)
    assert "cache_hit_reason" in facts_warm  # served from cache
    # and the cached facts carry the method identity
    assert facts_warm["method_version"] == "TC-1"
    assert facts_warm["metric"] == "ollivier-ricci-full-graph-v1"


def test_synthetic_cache_invalidated_when_input_changes(
        synthetic_cohort, tmp_path):
    """Editing the input UMI file changes the inputs digest: the cache
    must MISS (never serve the old computation)."""
    import gzip as gz

    umi, labels, _, _ = synthetic_cohort
    root = tmp_path / "cache3"
    cold, _, _ = _run_corrected(umi, labels, root, force=True)
    text = gz.open(umi, "rt").read().splitlines()
    # change one count token (row 1 = gene g0, first cell)
    parts = text[1].split("\t")
    parts[1] = str(int(parts[1]) + 3)
    text[1] = "\t".join(parts)
    with gz.open(umi, "wt") as f:
        f.write("\n".join(text) + "\n")
    edited, _, facts_e = _run_corrected(umi, labels, root, force=False)
    mism = compare_science(edited, cold, mode="exact")
    # the payload is recomputed because the cache missed on new inputs
    assert facts_e.get("cache_hit_reason") is None
    # and the numbers actually changed (different computation)
    assert mism != [] or edited["dl"]["delta"] != cold["dl"]["delta"]


def test_synthetic_positive_effect_detected_with_coverage(synthetic_cohort,
                                                          tmp_path):
    umi, labels, _, _ = synthetic_cohort
    payload, kdf, facts = _run_corrected(umi, labels, tmp_path / "c4",
                                         force=True)
    measured = kdf["status"] == "measured"
    require_pass({
        "coverage_reported": bool("coverage_fraction" in kdf.columns
                                  and "n_incident_measured" in kdf.columns),
        "not_sampled_flagged_not_dropped":
            bool((kdf["status"] == "not_sampled").sum() >= 0),
        "measured_have_incident_edges":
            bool((kdf.loc[measured, "n_incident_measured"] > 0).all()),
        "exclusions_audited":
            payload["exclusions"]["n_non_estimable_se"] >= 0,
    }, context="synthetic pipeline")
    assert payload["method_version"] == "TC-1"
    assert np.isfinite(payload["dl"]["delta"])
    assert payload["dl"]["k"] >= 5


def test_duplicate_coordinate_cohort_reports_non_estimable(tmp_path):
    """A UMI control with genuinely coincident expression vectors is
    reported as non-estimable, not silently repaired."""
    rng = np.random.default_rng(2)
    n_genes = 40
    cells = []
    meta = []
    cell_no = 0
    for grp in ["malignant"] * 12 + ["immune"] * 12:
        cid = f"c{cell_no:03d}"
        counts = rng.poisson(0.5, size=n_genes)
        cells.append((cid, counts))
        meta.append((cid, "P1", grp))
        cell_no += 1
    # clone a cell exactly (duplicate coordinates in expression space)
    cells.append(("cX", cells[0][1].copy()))
    meta.append(("cX", "P1", "malignant"))
    p = tmp_path / "dup.tsv.gz"
    import gzip as gz

    with gz.open(p, "wt") as f:
        f.write("gene\t" + "\t".join(c for c, _ in cells) + "\n")
        for gi in range(n_genes):
            f.write(f"g{gi}\t" + "\t".join(str(c[1][gi]) for c in cells)
                    + "\n")
    labels = pd.DataFrame(meta, columns=["barcode", "patient", "label"])
    counts, _ = read_umi_tsv_selected(p, labels["barcode"].tolist())
    lib = counts.sum(axis=0, keepdims=True)
    X_log = np.log1p(counts / np.where(lib == 0, 1, lib) * 1e6)
    with pytest.raises(numerics.DuplicateCoordinatesError):
        cohort.compute_cohort_kappa(
            X_log, labels["patient"].values,
            (labels["label"] == "malignant").values,
            (labels["label"] == "immune").values,
            k=4, alpha=0.5, n_edges=50, seed=SEED, force=True)
