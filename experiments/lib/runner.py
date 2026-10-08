"""Corrected-driver scaffolding: input gates, blocked-run artifacts,
self-reproduction gates.

A corrected driver NEVER fabricates results when its GEO/figshare inputs
are absent: it writes a ``results_corrected.json`` artifact with
status "blocked" naming every missing input (with accession/URL), and
exits nonzero.  When inputs exist it runs the corrected pipeline, binds
every artifact to the library METHOD_VERSION, and on re-runs verifies the
FULL payload against the previously written artifact through
``verification.compare_science`` (the audit's complete-field gate: a
changed SE, CI, p-value, heterogeneity, count or bootstrap field is a
failure, not a warning).

Artifact writes are STRICT and ATOMIC (audit R03): ``json.dump`` would
otherwise emit bare ``NaN``/``Infinity``, and a crashed run could leave a
half-written or stale-but-plausible artifact at the same path.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path


class BlockedInputs(RuntimeError):
    pass


class ArtifactError(RuntimeError):
    """An artifact could not be strictly serialized or read."""


def _reject_constant(value):
    raise ArtifactError(f"non-standard JSON constant {value!r} in artifact")


def _unique_object(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ArtifactError(f"duplicate JSON key {k!r} in artifact")
        obj[k] = v
    return obj


def _reject_nonfinite(value):
    """Reject nonfinite floats anywhere in a parsed artifact (R03).

    ``1e999`` deserializes to ``inf`` WITHOUT triggering
    ``parse_constant`` (which only fires for the literal ``NaN`` /
    ``Infinity`` tokens), so finiteness must be re-checked on the parsed
    tree.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(
            f"non-finite float {value!r} in artifact (e.g. 1e999 "
            "overflow); represent missing data as null with a status")
    if isinstance(value, dict):
        return {k: _reject_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_reject_nonfinite(v) for v in value]
    return value


def load_json_strict(path):
    """Read an artifact JSON strictly (R03).

    Rejects ``NaN``/``Infinity`` constants, duplicate keys and numeric
    overflow such as ``1e999`` that would otherwise deserialize as
    infinity.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        parsed = json.loads(text, parse_constant=_reject_constant,
                            object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid artifact JSON: {exc}") \
            from exc
    try:
        return _reject_nonfinite(parsed)
    except ArtifactError as exc:
        raise ArtifactError(f"{path}: {exc}") from exc


def atomic_json(path, payload):
    """Strict, atomic artifact write (R03).

    * the payload must be JSON-strict: nonfinite floats are an ERROR
      (represent missingness with ``null`` plus an explanatory status),
      never a bare ``NaN`` token;
    * write to a temporary file in the destination directory, then
      ``replace`` — a previously approved artifact is never destroyed by
      a failed write;
    * a ``replace`` whose acknowledgment is lost counts as completed only
      when the destination provably holds the staged bytes, so a landed
      failure receipt never masks the exception it reports (an uncertain
      publication must stay uncertain); otherwise the failure stands and
      the previous artifact is intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.dumps(payload, indent=2, sort_keys=True,
                          allow_nan=False)
    except ValueError as exc:
        raise ArtifactError(
            f"{path}: payload is not strictly serializable ({exc}); "
            "nonfinite measured values are errors, not JSON constants — "
            "represent missing data as null with a status") from exc
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
        try:
            Path(tmp).replace(path)
        except OSError:
            landed = False
            try:
                landed = (not Path(tmp).exists()
                          and path.read_text(encoding="utf-8") == data)
            except OSError:
                landed = False
            if not landed:
                raise
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()
    return path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _display_path(path):
    """Repo-relative display path when possible (committed blocked
    artifacts must not embed machine-specific absolute paths)."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def check_inputs(required):
    """``required``: list of (role, path, provenance_description).

    Returns the list of missing entries (role, path, provenance).
    """
    missing = []
    for role, path, provenance in required:
        if not Path(path).exists():
            missing.append({"role": role, "path": _display_path(path),
                            "provenance": provenance})
    return missing


def write_results(out_path, payload):
    return atomic_json(out_path, payload)


def blocked_payload(driver_name, missing, note=""):
    return {
        "driver": driver_name,
        "status": "blocked",
        "blocked_reason": "required inputs absent; rerun wiring only — "
                          "results are never fabricated",
        "missing_inputs": missing,
        "note": note,
        "wallclock_seconds": 0.0,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime()),
    }


def run_or_block(driver_name, required, out_path, run_fn, note="", *,
                 self_reproduce=True, argv=None):
    """Standard corrected-driver entry point.

    1. missing inputs -> blocked artifact + exit code 2;
    2. inputs present -> run; payload carries status "ok";
    3. if a previous artifact exists and ``self_reproduce``, compare the
       FULL new payload against it with ``compare_science`` (exact mode)
       and refuse to overwrite on mismatch unless ``--update`` was passed
       (the mismatch is reported field-by-field, never silently
       accepted); with ``--update`` the new artifact is written and the
       change is recorded in ``supersedes``.
    """
    from lib.verification import VerificationError, compare_science

    out_path = Path(out_path)
    missing = check_inputs(required)
    if missing:
        write_results(out_path, blocked_payload(driver_name, missing, note))
        print(f"[{driver_name}] BLOCKED — missing {len(missing)} required "
              f"input(s):")
        for m in missing:
            print(f"  - {m['role']}: {m['path']}\n    {m['provenance']}")
        print(f"[{driver_name}] wrote blocked artifact {out_path}")
        sys.exit(2)

    t0 = time.time()
    payload = run_fn()
    payload.setdefault("driver", driver_name)
    payload.setdefault("status", "ok")
    payload["wallclock_seconds"] = time.time() - t0
    payload["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())

    update = "--update" in (argv or sys.argv[1:])
    volatile = {"wallclock_seconds", "created_utc", "supersedes"}
    if self_reproduce and out_path.exists():
        previous = load_json_strict(out_path)
        if previous.get("status") != "blocked":
            # runtime/reporting metadata is kept OUT of the scientific
            # comparison (audit protocol: cache-hit/runtime information
            # stays separate from the scientific payload)
            sci_new = {k: v for k, v in payload.items()
                       if k not in volatile}
            sci_old = {k: v for k, v in previous.items()
                       if k not in volatile}
            mism = compare_science(sci_new, sci_old, mode="exact",
                                   provenance="self-reproduction")
            if mism:
                if not update:
                    raise VerificationError(
                        f"{driver_name}: rerun differs from the committed "
                        f"artifact in {len(mism)} field(s); pass --update "
                        "to supersede after review (first differences): "
                        + "; ".join(mism[:10]))
                payload["supersedes"] = {
                    "previous_created_utc": previous.get("created_utc"),
                    "n_changed_fields": len(mism),
                    "first_differences": mism[:20],
                }
    write_results(out_path, payload)
    print(f"[{driver_name}] wrote {out_path} (status={payload['status']})")
    return payload
