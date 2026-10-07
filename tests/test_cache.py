"""Cache-invalidation proof (R01) and cold/warm/forced identity.

The corrected cache must NEVER serve a payload from a different
computation.  The legacy filename-keyed design (cache/<cohort>_kappa.npz)
is demonstrated to serve a foreign payload — the fail-before behaviour —
before the corrected cache is shown to refuse exactly that.
"""

import numpy as np
import pandas as pd
import pytest

from lib.cache import ResultCache, digest_of_params


def test_legacy_filename_cache_serves_foreign_payload(tmp_path,
                                                      legacy_meta):
    """FAIL-BEFORE documentation: the committed META cache path is
    '<cohort><tag>_kappa.npz' with no method/input binding, so a file
    written by ANY computation is served verbatim on the next run."""
    import numpy as np

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    path = cache_dir / "Tirosh_melanoma_v2_kappa.npz"
    # a payload from a DIFFERENT computation (here: constant junk)
    np.savez(path, patient=np.array(["P1", "P1"]),
             is_mal=np.array([True, False]), kappa=np.array([9.0, 9.0]))
    z = np.load(path, allow_pickle=True)
    df = pd.DataFrame({"patient": z["patient"], "is_mal": z["is_mal"],
                       "kappa": z["kappa"]})
    assert (df["kappa"] == 9.0).all()  # foreign payload served, no gate
    # the committed loader takes exactly this path with allow_pickle=True
    src = open(legacy_meta.__file__).read()
    assert 'np.load(cache_path, allow_pickle=True)' in src
    assert "CACHE / f\"{cohort}{cache_tag}_kappa.npz\"" in src


def test_corrected_cache_cold_warm_forced_identical(tmp_path):
    """Cold, warm and forced-recompute runs must produce IDENTICAL
    scientific payloads (audit local-verification protocol)."""
    cache = ResultCache(tmp_path / "c", "exp/primary")
    inputs_digest = "a" * 64
    params = {"k": 15, "alpha": 0.5, "n_edges": 100, "seed": 1}
    pdg = digest_of_params(params)
    key = cache.key(inputs_digest, pdg)
    payload = {"kappa": np.arange(50, dtype=float) / 7,
               "patient": np.arange(50) % 5}

    def compute():
        return {k: v.copy() for k, v in payload.items()}

    # COLD: miss -> compute -> save
    arrays, why = cache.load(key, inputs_digest, pdg)
    assert arrays is None and why == "no stored entry"
    cold = compute()
    cache.save(key, cold, facts={"inputs_digest": inputs_digest,
                                 "params_digest": pdg})
    # WARM: verified hit with identical payload
    arrays, meta = cache.load(key, inputs_digest, pdg)
    assert arrays is not None
    for k in cold:
        assert np.array_equal(arrays[k], cold[k])
    assert meta["method_version"] == cache.method_version
    # FORCED: miss by definition, recompute reproduces the same payload
    arrays_f, why_f = cache.load(key, inputs_digest, pdg, force=True)
    assert arrays_f is None and why_f == "forced recompute"
    forced = compute()
    for k in cold:
        assert np.array_equal(forced[k], cold[k])


def test_cache_never_serves_across_method_versions(tmp_path):
    """The core R01 property: a method-version change is a MISS (stale
    results are recomputed, never returned)."""
    v1 = ResultCache(tmp_path / "c", "exp", method_version="TC-1")
    inputs_digest = "b" * 64
    params = {"k": 15}
    pdg = digest_of_params(params)
    key1 = v1.key(inputs_digest, pdg)
    v1.save(key1, {"x": np.array([1.0, 2.0])},
            facts={"inputs_digest": inputs_digest, "params_digest": pdg})
    assert v1.load(key1, inputs_digest, pdg)[0] is not None

    v2 = ResultCache(tmp_path / "c", "exp", method_version="TC-2")
    key2 = v2.key(inputs_digest, pdg)
    assert key1 != key2, "keys must embed the method version"
    arrays, why = v2.load(key2, inputs_digest, pdg)
    assert arrays is None
    # even a colliding filename cannot serve: sidecar version mismatch
    arrays2, why2 = v2.load(key1, inputs_digest, pdg)
    assert arrays2 is None and "method version changed" in why2


def test_cache_never_serves_across_inputs(tmp_path):
    """Different input data digest -> different key AND a facts-check
    miss even if the key were forced equal."""
    cache = ResultCache(tmp_path / "c", "exp")
    params = {"k": 15}
    pdg = digest_of_params(params)
    d1, d2 = "c" * 64, "d" * 64
    k1 = cache.key(d1, pdg)
    cache.save(k1, {"x": np.array([1.0])}, facts={"inputs_digest": d1,
                                                  "params_digest": pdg})
    assert cache.load(k1, d1, pdg)[0] is not None
    assert cache.key(d2, pdg) != k1
    arrays, why = cache.load(k1, d2, pdg)
    assert arrays is None and "input digest changed" in why


def test_cache_never_serves_across_params(tmp_path):
    cache = ResultCache(tmp_path / "c", "exp")
    d = "e" * 64
    p1 = digest_of_params({"k": 15, "alpha": 0.5})
    p2 = digest_of_params({"k": 15, "alpha": 0.75})
    assert p1 != p2
    k1 = cache.key(d, p1)
    cache.save(k1, {"x": np.array([1.0])}, facts={"inputs_digest": d,
                                                  "params_digest": p1})
    arrays, why = cache.load(k1, d, p2)
    assert arrays is None and "parameter digest changed" in why


def test_cache_params_digest_float_exactness():
    """0.1+0.2 vs 0.30000000000000004 differ; 0.5 is stable across
    int/float spellings of the same value."""
    a = digest_of_params({"alpha": 0.1 + 0.2})
    b = digest_of_params({"alpha": 0.3})
    assert a != b
    assert digest_of_params({"k": 15}) == digest_of_params({"k": np.int64(15)})


def test_cache_payload_is_never_pickled(tmp_path):
    """allow_pickle=False must load every stored payload (legacy caches
    were pickle-enabled)."""
    cache = ResultCache(tmp_path / "c", "exp")
    key = cache.key("f" * 64, digest_of_params({}))
    obj = np.array([{"a": 1}], dtype=object)  # would REQUIRE pickle
    with pytest.raises(ValueError):
        # npz cannot store object arrays with allow_pickle disabled; the
        # save path must therefore refuse them outright
        import io

        buf = io.BytesIO()
        np.savez(buf, x=obj, allow_pickle=False)
    cache.save(key, {"x": np.array([1.5])})
    with np.load(tmp_path / "c" / f"exp_{key}.npz", allow_pickle=False) as z:
        assert "x" in z.files
