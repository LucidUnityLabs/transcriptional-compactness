"""Verification-gate tests (C09, C10): every field enforced, no vacuous
passes, log-space tiny-p comparison, complete scheme sets."""

import pytest

from lib.verification import (VerificationError, compare_log_p, compare_science,
                          label_verdict, require_complete, require_pass)


# ---------------------------------------------------------------------------
# compare_science (C09)
# ---------------------------------------------------------------------------
BASELINE = {
    "pooled_delta": 0.16801092468924114,
    "se": 0.12188877196441941,
    "ci": [-0.07089106836102091, 0.4069129177395032],
    "p": 0.08404061877133445,
    "i2": 96.14265770302804,
    "k": 18,
    "direction": {"n_pos": 12, "n_neg": 6},
    "bootstrap": {"mean": 0.1729, "ci": [-0.0268, 0.3821]},
}


def test_compare_science_passes_on_identity():
    assert compare_science(BASELINE, BASELINE) == []


def test_compare_science_detects_every_changed_field():
    # the legacy gate enforced ONLY pooled delta; here a changed SE alone
    # must be reported
    obs = dict(BASELINE, se=0.2)
    mm = compare_science(obs, BASELINE)
    assert any("se:" in m for m in mm)
    for field, new in [("pooled_delta", 0.17), ("p", 0.09),
                       ("i2", 90.0), ("k", 17)]:
        mm = compare_science(dict(BASELINE, **{field: new}), BASELINE)
        assert any(field in m for m in mm), field
    mm = compare_science(dict(BASELINE, direction={"n_pos": 11,
                                                   "n_neg": 7}), BASELINE)
    assert any("direction" in m for m in mm)
    mm = compare_science(dict(BASELINE, bootstrap={"mean": 0.0,
                                                   "ci": [0, 1]}), BASELINE)
    assert any("bootstrap" in m for m in mm)
    mm = compare_science({k: v for k, v in BASELINE.items()
                          if k != "bootstrap"}, BASELINE)
    assert any("bootstrap: missing" in m for m in mm)


def test_compare_science_log_space_tiny_p():
    """atol=1e-8 must NOT make 1e-38 and 2e-38 equal."""
    exp = {"p": 1e-38}
    obs = {"p": 2e-38}
    assert compare_science(obs, exp, mode="tolerance", atol=1e-8) == []
    assert compare_log_p("p", 2e-38, 1e-38) != []


def test_require_pass_refuses_empty_and_failed():
    with pytest.raises(VerificationError, match="empty"):
        require_pass({})
    with pytest.raises(VerificationError, match="1/3"):
        require_pass({"a": True, "b": False, "c": True})
    assert require_pass({"a": True, "b": 1}) is True


def test_gate_history_documented():
    """The legacy sensitivity script computed se/ci/p/i2/k/direction/
    bootstrap exact-match checks but stopped execution only on the
    pooled-delta check — pinned here by importing the committed script
    constants (documentation of the fail-before behaviour)."""
    from conftest import load_legacy

    sens = load_legacy("sens")
    gate_fields = [k for k in dir(sens) if "EXPECTED" in k]
    assert gate_fields == ["EXPECTED_BASELINE"]
    expected_checks = {
        "pooled_delta_exact_match", "se_exact_match", "ci_exact_match",
        "one_sided_p_exact_match", "i2_exact_match", "k_exact_match",
        "direction_counts_exact_match", "bootstrap_exact_match"}
    src = (sens.__file__)
    text = open(src).read()
    for c in expected_checks:
        assert c in text
    # only ONE of them gates the exit:
    assert 'if not gate["pooled_delta_exact_match"]:' in text
    # and no other gate field appears in an if-statement
    for c in expected_checks - {"pooled_delta_exact_match"}:
        assert f'if not gate["{c}"]' not in text


# ---------------------------------------------------------------------------
# label_verdict (C10)
# ---------------------------------------------------------------------------
SCHEMES = ["original", "alt1_broad", "alt2_cluster"]


def _run(delta, p=1e-4, ok=True):
    return {"ok": ok, "delta": delta, "p_welch": p}


def test_label_verdict_all_positive():
    res = {s: _run(0.3 + 0.1 * i) for i, s in enumerate(SCHEMES)}
    v = label_verdict(res, SCHEMES)
    assert v["verdict"] == "robust_all_positive"
    assert v["n_classified"] == 3


def test_label_verdict_failed_scheme_is_inconclusive():
    """Legacy filtered failed schemes from recovered_deltas and could
    treat the survivors as agreement; corrected refuses."""
    res = {s: _run(0.3) for s in SCHEMES}
    res["alt2_cluster"] = {"ok": False}
    v = label_verdict(res, SCHEMES)
    assert v["verdict"] == "inconclusive"
    assert "alt2_cluster" in v["problems"][0]


def test_label_verdict_missing_scheme_rejected():
    res = {"original": _run(0.3)}
    with pytest.raises(VerificationError, match="missing required"):
        label_verdict(res, SCHEMES)


def test_label_verdict_nonfinite_delta_rejected():
    res = {s: _run(0.3) for s in SCHEMES}
    res["alt1_broad"] = {"ok": True, "delta": float("nan"),
                         "p_welch": 0.01}
    v = label_verdict(res, SCHEMES)
    assert v["verdict"] == "inconclusive"
    assert any("nonfinite" in p for p in v["problems"])


def test_label_verdict_null_vs_negative():
    res = {s: _run(0.3) for s in SCHEMES}
    res["alt1_broad"] = _run(0.01, p=0.5)
    assert label_verdict(res, SCHEMES)["verdict"] == \
        "robust_in_direction_with_null_schemes"
    res["alt1_broad"] = _run(-0.4)
    assert label_verdict(res, SCHEMES)["verdict"] == \
        "annotation_dependent_sign_reversal"


def test_legacy_labelval_vacuous_all_documented(legacy_labelval):
    """FAIL-BEFORE documentation: the committed driver's decomposition
    'all positive' check is vacuously true when every scheme fails."""
    dec = {"original": {"ok": False}, "alt1_broad": {"ok": False},
           "alt2_cluster": {"ok": False}}
    dec_all_pos = all(dec[s].get("delta", 0) > 0
                      for s in legacy_labelval.SCHEMES
                      if dec.get(s, {}).get("ok"))
    assert dec_all_pos is True  # vacuous — the documented defect


def test_legacy_labelval_partial_run_keyerror_documented(legacy_labelval):
    """FAIL-BEFORE documentation: a partially successful run (some scheme
    not ok) makes the verdict loop index a missing 'delta' key."""
    schemes = {"original": {"ok": True, "delta": 0.3, "p_welch": 1e-4},
               "alt1_broad": {"ok": False},
               "alt2_cluster": {"ok": True, "delta": 0.4, "p_welch": 1e-4}}
    with pytest.raises(KeyError):
        for s in legacy_labelval.SCHEMES:
            d = schemes[s]["delta"]  # noqa: B023 - mirrors committed code
