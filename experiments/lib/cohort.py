"""Corrected cohort analysis pipeline (TC-1).

Shared driver layer used by every ``run_corrected.py``: validated
normalisation, float64 PCA with an explicit deterministic solver, named
RNG streams, minimum-budget-first stratified sampling with audited
exclusions, curvature with coverage, and per-patient Cliff contrasts with
explicit non-estimable handling.

Index-level audit notes implemented here:

* R04 — float64 everywhere, explicit SVD solver, stable HVG axis order
  (variance descending with index tie-break), named RNG streams per
  stage.
* R05 — the legacy sampler capped per patient and THEN downsampled
  globally, which could starve exactly the patients that had qualified;
  the corrected sampler budgets the per-group MINIMUM for every
  qualifying patient first, then distributes the remaining budget, and
  records every exclusion with its reason.
* C08 — every payload records ``patient_unit_verified`` ONLY when the
  caller supplies an actual verified specimen->donor mapping; grouping a
  column is not verification.
"""

from __future__ import annotations

import json as _json
import hashlib
import importlib.metadata
import platform
from pathlib import Path

import numpy as np

IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
import pandas as pd
from scipy.stats import norm as _norm

from . import METHOD_VERSION, numerics
from .cache import ResultCache, digest_of_params


def rng_stream(seed, stage_name):
    """Named deterministic RNG stream (distinct, reproducible per stage)."""
    entropy = [int(seed)] + [ord(c) for c in stage_name]
    return np.random.default_rng(entropy)


def log_cpm_full_library(counts_genes_cells, min_detected=5,
                         target_sum=1e6):
    """log1p-CPM normalization with the R04 denominator contract.

    * the per-cell library size is computed over ALL raw genes BEFORE any
      gene filtering (the legacy held-out paths filtered genes first, so
      the denominator depended on the filter — a silent method change);
    * empty (all-zero) or nonfinite libraries are EXPLICIT failures:
      excluding such cells is a QC decision the caller must make and
      record, never a silent ``0 -> 1`` substitution;
    * everything is float64 end to end.

    Returns ``(X_log, kept_gene_indices, facts)`` where ``X_log`` is
    genes x cells log1p-CPM.
    """
    counts = np.ascontiguousarray(
        np.asarray(counts_genes_cells, dtype=np.float64))
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise numerics.NumericsError(
            f"bad counts matrix {counts.shape}")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise numerics.NumericsError(
            "counts must be finite and nonnegative")
    lib = counts.sum(axis=0)
    n_empty = int((lib <= 0).sum())
    if n_empty:
        raise numerics.NumericsError(
            f"{n_empty} of {lib.size} cells have empty raw libraries — "
            "apply an explicit upstream QC exclusion and record it; a "
            "silent 0->1 denominator substitution fabricates expression")
    expressed = (counts > 0).sum(axis=1)
    keep = np.flatnonzero(expressed >= int(min_detected))
    if keep.size == 0:
        raise numerics.NumericsError(
            f"no genes detected in >= {min_detected} cells")
    kept = counts[keep]
    X_log = np.log1p(kept / lib * float(target_sum))
    if not np.all(np.isfinite(X_log)):
        raise numerics.NumericsError("nonfinite normalized expression")
    facts = {
        "denominator": "full raw library (computed before gene filtering)",
        "n_genes_full": int(counts.shape[0]),
        "n_genes_kept": int(keep.size),
        "min_detected_cells": int(min_detected),
        "target_sum": float(target_sum),
        "dtype": "float64",
    }
    return np.ascontiguousarray(X_log), keep, facts


def hvg_pca(X_log, n_hvg=2000, n_pc=50):
    """HVG selection + PCA on validated float64 input.

    X_log: genes x cells log-normalised matrix (float64).  Returns
    (cells x n_pc float64, facts-dict).  Explicit full-SVD solver: the
    result is deterministic for a given input, with stable HVG axis
    ordering (variance descending, index tie-break).
    """
    from sklearn.decomposition import PCA

    X = np.ascontiguousarray(np.asarray(X_log, dtype=np.float64))
    if X.ndim != 2 or min(X.shape) < 2:
        raise numerics.NumericsError(f"bad expression matrix {X.shape}")
    if not np.all(np.isfinite(X)):
        raise numerics.NumericsError("nonfinite entries in expression "
                                     "matrix")
    var = X.var(axis=1)
    n_h = int(min(n_hvg, X.shape[0]))
    hvg = np.lexsort((np.arange(X.shape[0]), -var))[:n_h]
    Xh = X[hvg]
    Xh = Xh - Xh.mean(axis=1, keepdims=True)
    n_c = int(min(n_pc, Xh.shape[1] - 1, Xh.shape[0]))
    if n_c < 1:
        raise numerics.NumericsError("no PCA components available")
    Xp = PCA(n_components=n_c, svd_solver="full").fit_transform(Xh.T)
    facts = {"n_hvg": n_h, "n_pca": int(n_c),
             "hvg_axis_order": "variance desc, index tie-break",
             "svd_solver": "full", "dtype": "float64"}
    return np.ascontiguousarray(Xp), facts


def stratified_sample_minima_first(patient, is_g1, is_g2,
                                   min_per_group=10, cap_per_group=60,
                                   total_cap=3500, seed=0):
    """Minimum-budget-first stratified sampling with audited exclusions.

    1. every patient qualifying (>= min_per_group in BOTH groups) is
       budgeted min(cap, available) per group FIRST;
    2. if the total exceeds total_cap, the surplus above the qualifying
       minima is reduced fairly, never below a qualifying patient's
       minimum;
    3. non-qualifying patients are excluded with recorded reasons;
    4. no cell is selected twice (overlap rejected by construction).

    Returns (selected_indices, audit_dict).
    """
    rng = rng_stream(seed, "stratified-sample")
    patient = np.asarray(patient)
    is_g1 = np.asarray(is_g1, dtype=bool)
    is_g2 = np.asarray(is_g2, dtype=bool)
    if np.any(is_g1 & is_g2):
        raise numerics.NumericsError("group masks overlap — a cell cannot "
                                     "be in both contrast groups")
    chosen = []
    exclusions = {}
    budgeted = {}
    for pid in np.unique(patient):
        m = patient == pid
        a = np.flatnonzero(m & is_g1)
        b = np.flatnonzero(m & is_g2)
        if a.size < min_per_group or b.size < min_per_group:
            exclusions[str(pid)] = {
                "reason": "below_min_per_group",
                "n_g1": int(a.size), "n_g2": int(b.size)}
            continue
        take_a = int(min(cap_per_group, a.size))
        take_b = int(min(cap_per_group, b.size))
        budgeted[pid] = (a, b, take_a, take_b)
    # first pass: minima
    picks = []
    for pid, (a, b, ta, tb) in budgeted.items():
        picks.append(rng.permutation(a)[:ta])
        picks.append(rng.permutation(b)[:tb])
    sel = np.concatenate(picks) if picks else np.array([], dtype=int)
    # second pass: fair reduction of surplus above qualifying minima
    if sel.size > total_cap:
        n_qual = len(budgeted)
        floor = 2 * n_qual * min_per_group
        if total_cap < floor:
            raise numerics.NumericsError(
                f"total_cap {total_cap} cannot honour the qualifying "
                f"minima ({floor} cells over {n_qual} patients); raise "
                "the cap or lower the minima explicitly")
        surplus = sel.size - total_cap
        # drop surplus cells uniformly from the surplus region (cells
        # above each patient's minimum), never from any minimum slot
        keep_mask = np.ones(sel.size, dtype=bool)
        # protect minimum slots: first min_per_group per group per patient
        protected = set()
        i = 0
        for pid, (a, b, ta, tb) in budgeted.items():
            for take in (ta, tb):
                protected.update(range(i, i + min_per_group))
                i += take
        unprotected = np.array([j for j in range(sel.size)
                                if j not in protected], dtype=int)
        drop = rng.choice(unprotected, size=surplus, replace=False)
        keep_mask[drop] = False
        sel = sel[keep_mask]
    audit = {
        "n_patients_total": int(np.unique(patient).size),
        "n_patients_qualifying": int(len(budgeted)),
        "exclusions": exclusions,
        "n_selected": int(sel.size),
        "min_per_group": int(min_per_group),
        "cap_per_group": int(cap_per_group),
        "total_cap": int(total_cap),
        "rng": "named stream 'stratified-sample'",
    }
    return np.sort(sel), audit


def compute_cohort_kappa(X_log, patient, is_mal, is_comp, *,
                         k=15, alpha=0.5, n_edges=4000, seed=0,
                         min_per_group=10, cap_per_group=60, total_cap=3500,
                         cache=None, cache_inputs_digest=None, force=False):
    """Corrected kappa computation for one cohort (no legacy fallbacks).

    Returns (per-cell DataFrame, facts).  When ``cache`` is provided the
    computed payload is bound to (METHOD_VERSION, inputs digest,
    parameters); a mismatched cache is never served.
    """
    params = {"k": k, "alpha": alpha, "n_edges": n_edges, "seed": seed,
              "min_per_group": min_per_group,
              "cap_per_group": cap_per_group, "total_cap": total_cap}
    from . import cache as cache_module
    params["loaded_implementation_sha256"] = {
        "cohort": IMPORTED_SOURCE_SHA256,
        "numerics": numerics.IMPORTED_SOURCE_SHA256,
        "cache": cache_module.IMPORTED_SOURCE_SHA256,
    }
    # Bind the actual scientific input as well as raw-source bytes. A
    # changed driver normalization, gene filter or label mask must not
    # reuse a curvature cache merely because source files are unchanged.
    identity = hashlib.sha256(_json.dumps({"shape": list(np.shape(X_log)),
        "patient": [str(p) for p in patient]}, sort_keys=True).encode())
    for row in np.asarray(X_log):
        identity.update(np.ascontiguousarray(row, dtype=np.float64).tobytes())
    identity.update(np.asarray(is_mal, dtype=np.bool_).tobytes())
    identity.update(np.asarray(is_comp, dtype=np.bool_).tobytes())
    params["scientific_input_sha256"] = identity.hexdigest()
    params["environment"] = {p: importlib.metadata.version(p) for p in
                             ("numpy", "scipy", "scikit-learn", "networkx", "POT")}
    params["python"] = platform.python_version()
    import ot.lp.emd_wrap
    params["pot_binary_sha256"] = hashlib.sha256(
        Path(ot.lp.emd_wrap.__file__).read_bytes()).hexdigest()
    pdg = digest_of_params(params)
    key = None
    if cache is not None and cache_inputs_digest is not None:
        key = cache.key(cache_inputs_digest, pdg)
        arrays, why = cache.load(key, cache_inputs_digest, pdg,
                                 force=force)
        if arrays is not None:
            df = pd.DataFrame({"patient": arrays["patient"],
                               "is_mal": arrays["is_mal"],
                               "kappa": arrays["kappa"],
                               "degree": arrays["degree"],
                               "n_incident_measured":
                                   arrays["n_incident_measured"],
                               "coverage_fraction":
                                   arrays["coverage_fraction"],
                               "status": arrays["status"]})
            facts = _json.loads(str(arrays["facts"][0]))
            facts["cache_hit_reason"] = why
            return df, facts

    sel, sample_audit = stratified_sample_minima_first(
        patient, is_mal, is_comp, min_per_group=min_per_group,
        cap_per_group=cap_per_group, total_cap=total_cap, seed=seed)
    Xp, pca_facts = hvg_pca(X_log[:, sel])
    G = numerics.knn_graph(Xp, k=k)
    edges = numerics.ricci_edges(
        G, alpha=alpha, n_edges=n_edges,
        rng=rng_stream(seed, "edge-draw"))
    cells = numerics.cell_curvature(G, edges)
    n = G.number_of_nodes()
    pat_s = np.asarray(patient)[sel]
    mal_s = np.asarray(is_mal, dtype=bool)[sel]
    df = pd.DataFrame({
        "patient": pat_s, "is_mal": mal_s,
        "kappa": [cells[i].kappa for i in range(n)],
        "degree": [cells[i].degree for i in range(n)],
        "n_incident_measured": [cells[i].n_incident_measured
                                for i in range(n)],
        "coverage_fraction": [cells[i].coverage_fraction for i in range(n)],
        "status": [cells[i].status for i in range(n)]})
    facts = {
        "method_version": METHOD_VERSION,
        "metric": "ollivier-ricci-full-graph-v1",
        "sample_audit": sample_audit,
        "pca": pca_facts,
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "or_edges_computed": int(len(edges)),
        "measured_edge_ids": sorted(f"{u},{v}" for u, v in edges),
        "transport_backend": (next(iter(edges.values())).backend
                              if edges else None),
    }
    if cache is not None and key is not None:
        cache.save(key, {
            "patient": np.asarray(df["patient"].values, dtype="U"),
            "is_mal": df["is_mal"].values.astype(bool),
            "kappa": df["kappa"].values.astype(np.float64),
            "degree": df["degree"].values.astype(np.int64),
            "n_incident_measured":
                df["n_incident_measured"].values.astype(np.int64),
            "coverage_fraction":
                df["coverage_fraction"].values.astype(np.float64),
            "status": np.asarray(df["status"].values, dtype="U"),
            "facts": np.array([_json.dumps(facts, sort_keys=True)]),
        }, facts={"inputs_digest": cache_inputs_digest,
                  "params_digest": pdg})
    return df, facts


def json_loads(s):
    import json

    return json.loads(s)


def per_patient_contrasts(df, min_cells=10):
    """Per-patient Cliff contrasts with explicit non-estimable handling.

    Patients with non-estimable projection SE stay in the table
    (descriptive point effect) but are EXCLUDED from inverse-variance
    pooling via an explicit, recorded exclusion list — never silently.
    """
    rows = []
    measured = df[df["status"] == "measured"]
    for pid, sub in measured.groupby("patient", sort=True):
        a = sub.loc[sub.is_mal, "kappa"].values
        b = sub.loc[~sub.is_mal, "kappa"].values
        if a.size < min_cells or b.size < min_cells:
            rows.append({"patient_id": str(pid), "n_malignant": int(a.size),
                         "n_comparator": int(b.size), "ok": False,
                         "reason": "below_min_cells"})
            continue
        res = numerics.cliff_placements(a, b)
        rows.append({"patient_id": str(pid), "ok": True,
                     "n_malignant": int(a.size),
                     "n_comparator": int(b.size),
                     "mean_kappa_mal": float(a.mean()),
                     "mean_kappa_comp": float(b.mean()),
                     "delta": res.delta, "se": res.se,
                     "se_estimable": bool(res.estimable),
                     "cliff_status": res.status})
    tab = pd.DataFrame(rows, columns=[
        "patient_id", "ok", "n_malignant", "n_comparator", "reason",
        "mean_kappa_mal", "mean_kappa_comp", "delta", "se",
        "se_estimable", "cliff_status"])
    estimable = tab[(tab.get("ok", False) == True) &  # noqa: E712
                    (tab.get("se_estimable", False) == True)]  # noqa: E712
    excluded = tab[(tab.get("ok", True) != False) &
                   (tab.get("se_estimable", True) != True)]
    exclusions = {
        "n_below_min_cells": int((tab.get("ok", True) == False).sum()),
        "n_non_estimable_se": int(len(excluded)),
        "non_estimable_patients": excluded["patient_id"].tolist()
        if len(excluded) else [],
        "note": "non-estimable projection SEs (degenerate separations) "
                "are excluded from inverse-variance pooling EXPLICITLY "
                "and reported descriptively — no variance floor is "
                "applied",
    }
    return tab, estimable, exclusions


def pool_cohort(tab, estimable, label="", seed=0, B=1000):
    """DL pooling + validating bootstrap over estimable patients."""
    if estimable.empty:
        return {"k": 0, "status": "INSUFFICIENT_PATIENTS", "label": label,
                "delta": None, "se": None, "ci_lo": None, "ci_hi": None,
                "p": None, "z": None, "one_sided_p_delta_gt_0": None,
                "bootstrap": {"B": 0, "status": "insufficient_patients"}}
    meta = numerics.dl_meta(estimable["delta"].values,
                            estimable["se"].values, label=label)
    boot = numerics.bootstrap_dl(estimable["delta"].values,
                                 estimable["se"].values, B=B,
                                 seed=seed)
    meta["one_sided_p_delta_gt_0"] = float(_norm.sf(meta["z"]))
    meta["bootstrap"] = boot
    return meta


def donor_unit_flag(mapping=None):
    """C08: patient-level claims require a VERIFIED specimen->donor map.

    Returns (flag, evidence): flag is True only when ``mapping`` is a
    non-empty reviewed {specimen: donor} mapping supplied by the caller;
    a nonempty ID column alone never sets it.
    """
    if mapping:
        return True, {"n_specimens": len(mapping),
                      "source": "caller-supplied reviewed map"}
    return False, {"source": "none — grouping a column is not donor "
                             "verification"}
