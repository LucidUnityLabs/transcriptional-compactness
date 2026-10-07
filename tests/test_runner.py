"""Runner artifact-serialization tests (R03): strict JSON, atomic writes.

Output artifacts must never contain bare ``NaN``/``Infinity`` tokens, a
crashed rerun must never destroy a previously approved artifact at the
same path, and strict reads reject the non-standard constants, duplicate
keys and ``1e999``-style overflow that ``json.loads`` would otherwise
accept (the audit's R03 contract; ``1e999`` deserializes to ``inf``
WITHOUT firing ``parse_constant``).
"""

import json

import pytest

from lib.runner import ArtifactError, atomic_json, load_json_strict


def test_atomic_json_preserves_int_and_bool_types(tmp_path):
    """Integer counts and booleans survive the round trip as JSON int /
    bool (never float drift), and ``None`` stays explicit ``null``."""
    path = tmp_path / "result.json"
    atomic_json(path, {"k": 4, "ok": True, "optional": None,
                       "delta": 0.5})
    raw = json.loads(path.read_text())
    assert raw["k"] == 4 and isinstance(raw["k"], int)
    assert raw["ok"] is True
    assert raw["optional"] is None
    back = load_json_strict(path)
    assert back == raw


def test_atomic_json_rejects_nonfinite_payloads(tmp_path):
    """A nonfinite measured value is an ERROR (represent missingness as
    null plus a status), never a bare NaN token in an artifact."""
    path = tmp_path / "result.json"
    with pytest.raises(ArtifactError, match="serializable"):
        atomic_json(path, {"delta": float("nan")})
    with pytest.raises(ArtifactError):
        atomic_json(path, {"ci_hi": float("inf")})
    assert not path.exists()  # nothing half-written left behind


def test_atomic_json_failure_never_destroys_previous_artifact(tmp_path):
    """A failed rerun must leave the previously approved artifact's bytes
    intact at the same path (transactional publication)."""
    path = tmp_path / "result.json"
    atomic_json(path, {"approved": 1})
    old = path.read_bytes()
    with pytest.raises(ArtifactError):
        atomic_json(path, {"approved": float("nan")})
    assert path.read_bytes() == old
    # no temp files left in the destination directory
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_load_json_strict_rejects_nan_and_infinity(tmp_path):
    path = tmp_path / "a.json"
    for token in ("NaN", "Infinity", "-Infinity"):
        path.write_text('{"x": %s}' % token)
        with pytest.raises(ArtifactError, match="non-standard"):
            load_json_strict(path)


def test_load_json_strict_rejects_duplicate_keys(tmp_path):
    """A duplicated key silently keeps only the LAST value in plain
    ``json.load`` — in an approved artifact that is corruption."""
    path = tmp_path / "a.json"
    path.write_text('{"k": 35, "k": 18}')
    with pytest.raises(ArtifactError, match="duplicate"):
        load_json_strict(path)


def test_load_json_strict_rejects_numeric_overflow(tmp_path):
    """``1e999`` parses to ``inf`` without firing ``parse_constant``;
    finiteness must be re-checked on the parsed tree (nested too)."""
    path = tmp_path / "a.json"
    path.write_text('{"x": 1e999}')
    with pytest.raises(ArtifactError, match="non-finite"):
        load_json_strict(path)
    path.write_text('{"outer": {"deep": [1.0, -1e999]}}')
    with pytest.raises(ArtifactError, match="non-finite"):
        load_json_strict(path)


def test_load_json_strict_rejects_malformed_json(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"x": 1,,}')
    with pytest.raises(ArtifactError, match="invalid artifact JSON"):
        load_json_strict(path)
