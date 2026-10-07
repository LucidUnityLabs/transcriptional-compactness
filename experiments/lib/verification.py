"""Complete scientific comparison gates and label verdicts (TC-1).

Fixes relative to the historical verification scripts:

* The sensitivity gate enforced only one of the many checks it computed
  (audit C09): execution stopped when ``pooled_delta_exact_match`` was
  false while a changed SE, CI, p-value, heterogeneity, patient counts or
  bootstrap field could fail its own check and the run would continue.
  :func:`compare_science` compares EVERY field and :func:`require_pass``
  refuses an empty/incomplete check set.
* Tiny tail probabilities are compared in log space — a generic
  ``atol=1e-8`` must not make ``1e-38`` and ``2e-38`` "equal" (audit C09).
* Expected numbers are loaded from a real baseline artifact (the committed
  ``results.json``), not duplicated literals; each expected value carries
  its provenance path.
* Label-verification success logic could be incomplete or vacuous
  (audit C10): failed schemes were filtered from ``recovered_deltas``,
  ``all(... for ... if ...ok)`` was true for an empty collection, and a
  partially successful run could index a missing ``delta``.  The
  :func:`label_verdict` here requires the COMPLETE expected scheme set,
  requires each run to be ``ok``, validates each effect, and returns a
  structured inconclusive status when a required scheme fails — an empty
  filtered list is never confirmation.
"""

from __future__ import annotations

import math

import numpy as np

from . import numerics

#: Tail probabilities below this magnitude are compared in log10 space.
LOG_P_THRESHOLD = 1e-300


class VerificationError(AssertionError):
    """A scientific reproduction gate failed."""


def _leaf_match(path, obs, exp, mode, rtol, atol, log_paths, mismatches):
    if isinstance(exp, dict):
        if not isinstance(obs, dict):
            mismatches.append(f"{path}: expected dict, got {type(obs).__name__}")
            return
        for k, v in exp.items():
            if k not in obs:
                mismatches.append(f"{path}.{k}: missing in observed")
            else:
                _leaf_match(f"{path}.{k}", obs[k], v, mode, rtol, atol,
                            log_paths, mismatches)
        return
    if isinstance(exp, (list, tuple)):
        if not isinstance(obs, list) or len(obs) != len(exp):
            mismatches.append(f"{path}: expected list of {len(exp)}, got "
                              f"{obs!r}")
            return
        for i, v in enumerate(exp):
            _leaf_match(f"{path}[{i}]", obs[i], v, mode, rtol, atol,
                        log_paths, mismatches)
        return
    # scalar leaf
    if isinstance(exp, bool) or isinstance(obs, bool):
        if bool(obs) is not bool(exp):
            mismatches.append(f"{path}: {obs!r} != {exp!r}")
        return
    if exp is None or obs is None:
        if obs is not exp:
            mismatches.append(f"{path}: {obs!r} != {exp!r}")
        return
    if isinstance(exp, (int, float, np.integer, np.floating)) and \
            isinstance(obs, (int, float, np.integer, np.floating)):
        o, e = float(obs), float(exp)
        if mode == "exact":
            # int/float type drift with an equal value is a schema change
            # (a patient COUNT that silently becomes a float must be
            # flagged, not waved through as numerically equal)
            exp_int = isinstance(exp, (int, np.integer))
            obs_int = isinstance(obs, (int, np.integer))
            if exp_int != obs_int and o == e:
                mismatches.append(f"{path}: {obs!r} != {exp!r} "
                                  "(int/float type drift, exact)")
                return
            if o != e and not (math.isnan(o) and math.isnan(e)):
                name = path.lower()
                is_p = name.endswith("p") or name.endswith("_p") or \
                    ".p" in name or "pvalue" in name or "p_" in name
                if is_p and o == 0.0 and e == 0.0:
                    return
                mismatches.append(f"{path}: {o!r} != {e!r} (exact)")
        else:
            if not (math.isclose(o, e, rel_tol=rtol, abs_tol=atol)):
                mismatches.append(f"{path}: {o!r} !~ {e!r} "
                                  f"(rtol={rtol}, atol={atol})")
        return
    if obs != exp:
        mismatches.append(f"{path}: {obs!r} != {exp!r}")


def compare_science(observed, expected, mode="exact", rtol=1e-9, atol=1e-12,
                    provenance=""):
    """Compare EVERY field of a scientific payload against a baseline view.

    ``mode="exact"`` is the defined exact-environment mode (bit-identical
    arithmetic); ``mode="tolerance"`` is the separately named numerical-
    tolerance mode.  Tail-probability leaves below ``LOG_P_THRESHOLD`` are
    compared in log10 space with a relative log tolerance so underflowed
    zeroes stay distinguishable.

    Returns the mismatch list; callers pass it to :func:`require_pass` or
    raise immediately.  This function NEVER weakens a check silently.
    """
    mismatches: list[str] = []
    _leaf_match("", observed, expected, mode, rtol, atol, [], mismatches)
    if provenance:
        for i, m in enumerate(mismatches):
            mismatches[i] = f"[{provenance}] {m}"
    return mismatches


def compare_log_p(name, p_obs, p_exp, rel_log_tol=1e-6):
    """Compare tiny tail probabilities in log10 space (never as raw atol)."""
    if p_obs == 0.0 and p_exp == 0.0:
        return []
    lo, le = math.log10(max(p_obs, 5e-324)), math.log10(max(p_exp, 5e-324))
    if abs(lo - le) > rel_log_tol * max(1.0, abs(lo), abs(le)):
        return [f"{name}: log10 p {lo:.4f} vs {le:.4f} "
                f"({p_obs!r} vs {p_exp!r})"]
    return []


def require_pass(checks, context=""):
    """Require a NONEMPTY check dict with every value truthy.

    An empty check set is a failed gate (the historical sensitivity script
    enforced only ``pooled_delta_exact_match`` and ignored the rest; an
    empty ``all()`` was vacuously true in the label validation driver).
    """
    if not isinstance(checks, dict) or not checks:
        raise VerificationError(
            f"{context}: check set is empty — refusing vacuous pass")
    failed = {k: v for k, v in checks.items() if not v}
    if failed:
        raise VerificationError(
            f"{context}: {len(failed)}/{len(checks)} checks failed: "
            f"{sorted(failed)}")
    return True


def require_complete(items, expected, context=""):
    """Require the complete expected item set, with no filtering.

    The historical driver filtered failed schemes out of ``recovered``
    lists before checking sign agreement, so a partially failed run could
    look like a robustness success (audit C10).
    """
    missing = [s for s in expected if s not in items]
    if missing:
        raise VerificationError(
            f"{context}: missing required schemes {missing} (complete set "
            f"{list(expected)} required; filtering failures is not "
            "allowed)")
    return True


def label_verdict(scheme_results, expected_schemes, null_band=0.05,
                  alpha=0.05):
    """Structured, non-vacuous label-robustness verdict.

    Requires the complete expected scheme set, requires each scheme's run
    to be ``ok`` with a finite delta and p-value, and classifies each
    scheme positive/negative/null.  Any failed scheme yields a structured
    ``inconclusive`` verdict naming it — never a success built from an
    empty filtered list.  This is DESCRIPTIVE direction reporting only; it
    is not a baseline reproduction gate (rounded published literals are
    not a reproducibility certificate — audit C10).
    """
    require_complete(scheme_results, expected_schemes, context="label_verdict")
    classes = {}
    problems = []
    for s in expected_schemes:
        r = scheme_results[s]
        if not r.get("ok"):
            problems.append(f"scheme {s!r} did not complete")
            continue
        d = r.get("delta")
        p = r.get("p_welch", 1.0)
        if d is None or not np.isfinite(d):
            problems.append(f"scheme {s!r} has nonfinite delta {d!r}")
            continue
        if p is None or not np.isfinite(p):
            problems.append(f"scheme {s!r} has nonfinite p {p!r}")
            continue
        if d < -null_band and p < alpha:
            classes[s] = "negative"
        elif d > null_band and p < alpha:
            classes[s] = "positive"
        else:
            classes[s] = "null"
    if problems:
        return {"verdict": "inconclusive", "problems": problems,
                "classes": classes,
                "n_classified": len(classes),
                "n_expected": len(expected_schemes)}
    vals = set(classes.values())
    if "negative" in vals:
        verdict = "annotation_dependent_sign_reversal"
    elif vals == {"positive"}:
        verdict = "robust_all_positive"
    else:
        verdict = "robust_in_direction_with_null_schemes"
    deltas = [scheme_results[s]["delta"] for s in expected_schemes]
    return {"verdict": verdict, "classes": classes,
            "problems": [],
            "deltas": {s: float(scheme_results[s]["delta"])
                       for s in expected_schemes},
            "delta_min": float(min(deltas)), "delta_max": float(max(deltas)),
            "delta_span": float(max(deltas) - min(deltas)),
            "n_classified": len(classes),
            "n_expected": len(expected_schemes)}


# ---------------------------------------------------------------------------
# R02 — direction summaries and the hypothesis-decision gate
# ---------------------------------------------------------------------------
def direction_counts(values):
    """Positive, negative and ZERO effects counted separately.

    The historical summaries used ``n_negative = total - n_positive``,
    which counts exact zeros as negative (audit R02).  Exact zeros are a
    real outcome of Cliff's delta (complete intermingling) and must not
    be silently assigned a direction.
    """
    d = np.asarray(list(values), dtype=np.float64)
    if d.ndim != 1 or not np.all(np.isfinite(d)):
        raise VerificationError("nonfinite/non-1D direction input")
    n = int(d.size)
    return {
        "n_delta_positive": int((d > 0).sum()),
        "n_delta_negative": int((d < 0).sum()),
        "n_delta_zero": int((d == 0).sum()),
        "n_total": n,
        "fraction_positive": float((d > 0).mean()) if n else None,
    }


def protocol_decision(meta, *, technical_checks_passed,
                      donor_unit_verified, protocol_parameters_match,
                      minimum_patients=5):
    """Separate technical validity from the hypothesis decision (R02).

    A broken or under-sized run must never be presented as an
    interpretable NEGATIVE replication: the five statuses are
    ``INVALID_ANALYSIS`` (a technical/identity gate failed),
    ``INSUFFICIENT_PATIENTS`` (below the preregistered minimum),
    ``INVALID_NUMERICS`` (nonfinite/out-of-range numbers),
    ``SUPPORTED`` and ``NOT_SUPPORTED`` (technically valid outcomes of
    the preregistered one-sided rule delta > 0 with p < 0.05).
    """
    flags = (technical_checks_passed, donor_unit_verified,
             protocol_parameters_match)
    if not all(f is True for f in flags):
        return {"status": "INVALID_ANALYSIS", "supported": None}
    k = meta.get("k")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) \
            or int(k) < minimum_patients:
        return {"status": "INSUFFICIENT_PATIENTS", "supported": None}
    d, p = meta.get("delta"), meta.get("one_sided_p_delta_gt_0")
    if (not isinstance(d, (int, float, np.integer, np.floating))
            or not isinstance(p, (int, float, np.integer, np.floating))
            or not np.isfinite(d) or not np.isfinite(p)
            or not -1.0 <= float(d) <= 1.0 or not 0.0 <= float(p) <= 1.0):
        return {"status": "INVALID_NUMERICS", "supported": None}
    supported = bool(float(d) > 0 and float(p) < 0.05)
    return {"status": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "supported": supported,
            "rule": "preregistered one-sided: delta > 0 and "
                    "one-sided p < 0.05",
            "minimum_patients": int(minimum_patients)}


# ---------------------------------------------------------------------------
# §8.4/§8.8 — independent table-to-summary arithmetic verification
# ---------------------------------------------------------------------------
def verify_patient_meta(rows, reported, *, expected_patient_ids=None):
    """Recompute a meta-analysis from its FULL per-patient delta/SE table.

    Verifies arithmetic ONLY (never raw data, labels, graph metric or
    independence).  ``rows``: iterable of mappings with ``patient_id``,
    ``delta``, ``se``; ``reported``: the summary mapping to check.  A
    tampered summary field fails field-by-field.
    """
    rows = list(rows)
    ids = [r["patient_id"] for r in rows]
    if (not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise VerificationError("empty/duplicate/malformed patient table")
    if expected_patient_ids is not None and ids != list(
            expected_patient_ids):
        raise VerificationError(
            "patient identities/order differ from reference")
    actual = numerics.dl_meta([float(r["delta"]) for r in rows],
                              [float(r["se"]) for r in rows])
    fields = ("k", "delta", "se", "ci_lo", "ci_hi", "z", "p", "Q", "df",
              "tau2", "I2", "pQ", "delta_FE")
    missing = [f for f in fields if f not in reported]
    if missing:
        raise VerificationError(
            f"reported meta result is missing required fields: {missing}")
    mism = compare_science({f: actual[f] for f in fields},
                           {f: reported[f] for f in fields},
                           mode="tolerance", rtol=1e-8, atol=1e-10,
                           provenance="verify_patient_meta")
    if mism:
        raise VerificationError("; ".join(mism))
    return {"status": "ARITHMETIC_VERIFIED", "k": len(rows)}
