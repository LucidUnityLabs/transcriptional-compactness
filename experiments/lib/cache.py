"""Content-addressed, method-bound result cache (TC-1).

Fixes audit R01 relative to the historical ``cache/<cohort>_kappa.npz``
design:

* Legacy cache keys were FILENAMES.  A file written by a different method
  version, different input data or different parameters was served
  verbatim on the next run (``allow_pickle=True`` on top), so a cache
  could silently return results from a different computation.
* Here the cache key is a SHA-256 over (namespace, METHOD_VERSION, the
  digest of every input that determines the payload, and the parameter
  dictionary).  The stored sidecar records exactly that identity; a load
  whose current identity does not match the stored identity is a MISS and
  the computation reruns — a cache can never return results from a
  different computation.
* No pickle: payloads are plain NPZ arrays plus a JSON sidecar
  (``allow_pickle=False``), plus retained provenance facts (input digests,
  parameters, method version, wallclock) for auditability.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import METHOD_VERSION

IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_fileobj(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def digest_of_inputs(paths):
    """Stable digest of an ordered set of input files (existence checked)."""
    h = hashlib.sha256()
    h.update(b"inputs-v1")
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file())
            if not files:
                raise FileNotFoundError(f"cache input directory empty: {p}")
            h.update(p.name.encode())
            for file in files:
                h.update(str(file.relative_to(p)).encode())
                h.update(_sha256_file(file).encode())
            continue
        if not p.is_file():
            raise FileNotFoundError(f"cache input missing: {p}")
        h.update(str(p.name).encode())
        h.update(_sha256_file(p).encode())
    return h.hexdigest()


def digest_of_params(params):
    """Stable, sorted, float-exact digest of a parameter mapping."""
    def norm(v):
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, (np.floating, float)):
            return float.hex(float(v))
        if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, (np.ndarray,)):
            return [norm(x) for x in v.tolist()]
        return v

    blob = json.dumps(norm(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ResultCache:
    """Method- and input-bound cache; mismatches are misses, never serves.

    ``namespace`` identifies the analysis (e.g. ``"HELDOUT_GSE161529/"
    "primary"``).  ``method_version`` defaults to the library-wide
    :data:`METHOD_VERSION`; pass an explicit version when a driver pins
    its own.  ``force=True`` recomputes (overwrites) unconditionally.
    """

    def __init__(self, root, namespace, method_version=METHOD_VERSION):
        self.root = Path(root)
        self.namespace = str(namespace)
        self.method_version = str(method_version)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, inputs_digest, params_digest):
        h = hashlib.sha256()
        h.update(b"tc-cache-v1|")
        h.update(self.namespace.encode())
        h.update(b"|")
        h.update(self.method_version.encode())
        h.update(b"|")
        h.update(str(inputs_digest).encode())
        h.update(b"|")
        h.update(str(params_digest).encode())
        return h.hexdigest()

    def save(self, key, arrays, facts=None):
        base = self.root / f"{self.namespace.replace('/', '__')}_{key}"
        npz_path = base.with_suffix(".npz")
        meta_path = base.with_suffix(".meta.json")
        tmp_npz = npz_path.with_suffix(".npz.tmp")
        with open(tmp_npz, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
        tmp_npz.replace(npz_path)
        # payload digest: a tampered/corrupted NPZ can never be served as
        # a valid cached computation (audit R01: check the payload digest)
        with open(npz_path, "rb") as f:
            payload_sha256 = _sha256_fileobj(f)
        meta = {
            "cache_schema": "tc-cache-v1",
            "namespace": self.namespace,
            "method_version": self.method_version,
            "key": key,
            "payload_sha256": payload_sha256,
            "facts": facts or {},
            "created": time.time(),
        }
        tmp_meta = meta_path.with_suffix(".meta.json.tmp")
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        tmp_meta.replace(meta_path)
        return npz_path

    def load(self, key, inputs_digest, params_digest, force=False):
        """Return (arrays, meta) on a verified hit; (None, reason) on miss.

        A hit requires the sidecar to exist AND to record the exact
        current method version, input digest and parameter digest.  Any
        divergence is a miss (stale/foreign caches are never served);
        ``force=True`` is always a miss by definition.
        """
        if force:
            return None, "forced recompute"
        base = self.root / f"{self.namespace.replace('/', '__')}_{key}"
        npz_path = base.with_suffix(".npz")
        meta_path = base.with_suffix(".meta.json")
        if not npz_path.is_file() or not meta_path.is_file():
            return None, "no stored entry"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"unreadable sidecar: {exc}"
        if meta.get("cache_schema") != "tc-cache-v1":
            return None, "foreign cache schema"
        if meta.get("method_version") != self.method_version:
            return None, (f"method version changed: stored "
                          f"{meta.get('method_version')!r} != current "
                          f"{self.method_version!r} — stale cache is not "
                          "served (legacy filename-keyed caches had no "
                          "such protection)")
        if meta.get("key") != key:
            return None, "key mismatch"
        # inputs/params digests are embedded in the key by construction;
        # they are re-verified from the sidecar facts when provided.
        facts = meta.get("facts") or {}
        if "inputs_digest" in facts and facts["inputs_digest"] != \
                inputs_digest:
            return None, "input digest changed"
        if "params_digest" in facts and facts["params_digest"] != \
                params_digest:
            return None, "parameter digest changed"
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        except Exception as exc:  # noqa: BLE001 - any load failure is a miss
            return None, f"unreadable payload: {exc}"
        with open(npz_path, "rb") as f:
            payload_sha256 = _sha256_fileobj(f)
        if meta.get("payload_sha256") != payload_sha256:
            return None, ("payload digest mismatch (tampered or corrupted "
                          "cache file is never served)")
        return arrays, meta
