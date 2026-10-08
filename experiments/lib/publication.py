"""Immutable attempt sets published by one atomic pointer, never three file renames.

Readers resolve and validate the pointer once. Legacy flat files remain an
explicit fallback until the first accepted attempt; they are never replaced.
"""
from pathlib import Path
import csv
import os
import shutil
import tempfile
import time
import uuid
from .runner import ArtifactError, atomic_json, load_json_strict, check_inputs, blocked_payload
from .axes_provenance import sha

SCHEMA = 'tc-accepted-set-v1'


class PublicationUncertain(ArtifactError):
    """Pointer rename may have happened; inspect the pointer before retrying."""


def pointer_path(result):
    return Path(result).with_suffix('.accepted.json')


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ArtifactError('artifact path escapes attempt')
    return path


def read_set(result, *, require_pointer=False):
    """Read one pointer snapshot, verify manifest and every member, no silent fallback.

    Returns (role -> path, manifest). Flat fallback describes historical bytes,
    not a claim that they were atomically published by this API.
    """
    result = Path(result)
    pointer = pointer_path(result)
    if not pointer.exists():
        if require_pointer: raise ArtifactError('accepted pointer absent')
        return {'result': result}, {'schema': 'legacy-flat', 'status': 'historical'}
    p = load_json_strict(pointer)
    if p.get('schema') != SCHEMA: raise ArtifactError('foreign accepted pointer')
    manifest_path = _inside(result.parent, p['manifest'])
    if sha(manifest_path) != p['manifest_sha256']: raise ArtifactError('accepted manifest digest mismatch')
    m = load_json_strict(manifest_path)
    if m.get('schema') != SCHEMA or m.get('status') != 'complete' or m.get('attempt_id') != p['attempt_id']:
        raise ArtifactError('incomplete/foreign accepted manifest')
    required = {'result', 'provenance'} | set(m.get('required_roles', []))
    if not required <= set(m['artifacts']): raise ArtifactError('incomplete accepted artifact set')
    paths = {}
    for role, entry in m['artifacts'].items():
        path = _inside(manifest_path.parent, entry['path'])
        if not path.is_file() or sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ArtifactError(f'accepted artifact missing/changed: {role}')
        paths[role] = path
    provenance = load_json_strict(paths['provenance'])
    if provenance.get('attempt_id') != m['attempt_id']: raise ArtifactError('foreign attempt provenance')
    return paths, m


def load_result(result):
    paths, _ = read_set(result)
    return load_json_strict(paths['result'])


def resolve_table_result(result, table):
    """Compatibility resolver for flat-path CLI callers, one consistent snapshot."""
    paths, m = read_set(result)
    if m['schema'] == 'legacy-flat': return Path(table), paths['result']
    matches = [p for role,p in paths.items() if role.startswith('table:') and p.name == Path(table).name]
    if len(matches) != 1: raise ArtifactError('requested table not in accepted set')
    return matches[0], paths['result']


class AttemptCache:
    """Wrap ResultCache and retain exactly the key consumed by this attempt."""
    def __init__(self, root, namespace):
        from .cache import ResultCache
        self.cache = ResultCache(root, namespace)
        self.used_key = None

    def key(self, *args): return self.cache.key(*args)

    def load(self, key, *args, **kwargs):
        result = self.cache.load(key, *args, **kwargs)
        if result[0] is not None: self.used_key = key
        return result

    def save(self, key, *args, **kwargs):
        result = self.cache.save(key, *args, **kwargs)
        self.used_key = key
        return result

    def receipt(self, path):
        if self.used_key is None: raise ArtifactError('attempt has no used cache key')
        base = f"{self.cache.namespace.replace('/', '__')}_{self.used_key}"
        atomic_json(path, {'namespace':self.cache.namespace, 'key':self.used_key,
                          'sidecar':base+'.meta.json','payload':base+'.npz'})


def validate_breast_set(attempt, payload, arms, variants):
    """Check CSV against complete JSON donor rows and cache payload/sidecar pairs.

    No graph calculation or NPZ numerical inspection. The calculation owns
    numerical validity; this gate checks coherent serialization/provenance.
    """
    attempt = Path(attempt)
    if payload.get('status') != 'ok': raise ArtifactError('cannot accept non-ok calculation')
    if set(payload['comparators']) != set(arms): raise ArtifactError('arm set differs')
    roles = {'result': attempt/'result.json', 'provenance': attempt/'provenance.json'}
    for arm in arms:
        if set(payload['comparators'][arm]) != set(variants): raise ArtifactError('variant set differs')
        for variant in variants:
            path = attempt / f'per_patient_{arm}_{variant}_corrected.csv'
            with path.open(newline='') as handle: rows = list(csv.DictReader(handle))
            reported = payload['comparators'][arm][variant]['per_patient']
            if len(rows) != len(reported) or len({r['patient_id'] for r in rows}) != len(rows):
                raise ArtifactError('CSV donor identity/count mismatch')
            for row in rows:
                expected = reported.get(row['patient_id'])
                if expected is None: raise ArtifactError('CSV donor not in JSON')
                for field in ('delta', 'se'):
                    if float(row[field]) != expected[field]: raise ArtifactError(f'CSV/JSON {field} mismatch')
                for field in ('n_malignant','n_comparator'):
                    if int(row[field]) != expected[field]: raise ArtifactError(f'CSV/JSON {field} mismatch')
            roles[f'table:{arm}:{variant}'] = path
    sidecars = []
    for arm in arms:
        for variant in variants:
            receipt_path = attempt/f'cache_{arm}_{variant}.json'
            receipt = load_json_strict(receipt_path)
            expected_namespace = f'HELDOUT_GSE161529/{arm}/{variant}'
            if receipt['namespace'] != expected_namespace: raise ArtifactError('foreign used cache receipt')
            base = f"{expected_namespace.replace('/', '__')}_{receipt['key']}"
            if receipt['sidecar'] != base+'.meta.json' or receipt['payload'] != base+'.npz':
                raise ArtifactError('used cache receipt differs')
            sidecars.append(attempt/'cache'/receipt['sidecar'])
            roles[f'cache-receipt:{arm}:{variant}'] = receipt_path
    namespaces = set()
    for sidecar in sidecars:
        meta = load_json_strict(sidecar)
        npz = sidecar.with_name(sidecar.name.removesuffix('.meta.json') + '.npz')
        facts = meta.get('facts', {})
        if meta.get('cache_schema') != 'tc-cache-v1' or meta.get('method_version') != payload['method_version'] or not facts.get('inputs_digest') or not facts.get('params_digest'):
            raise ArtifactError('cache provenance incomplete')
        from .cache import ResultCache
        cache = ResultCache(attempt/'cache', meta['namespace'], meta['method_version'])
        if cache.key(facts['inputs_digest'],facts['params_digest']) != meta['key'] or sha(npz) != meta.get('payload_sha256'):
            raise ArtifactError('cache payload/key mismatch')
        if sidecar.stem.removesuffix('.meta') != f"{meta['namespace'].replace('/', '__')}_{meta['key']}":
            raise ArtifactError('cache filename identity differs')
        arrays, reason = cache.load(meta['key'], facts['inputs_digest'], facts['params_digest'])
        if arrays is None: raise ArtifactError(f'cache validation failed: {reason}')
        import json
        import numpy as np
        expected_fields = {'patient','is_mal','kappa','degree','n_incident_measured','coverage_fraction','status','facts'}
        if set(arrays) != expected_fields: raise ArtifactError('cache array schema differs')
        lengths = {len(arrays[name]) for name in expected_fields - {'facts'}}
        if len(lengths) != 1 or arrays['facts'].shape != (1,): raise ArtifactError('cache array axes differ')
        measured = arrays['status'] == 'measured'
        if not np.isfinite(arrays['kappa'][measured]).all() or not np.isfinite(arrays['coverage_fraction']).all():
            raise ArtifactError('cache measured values nonfinite')
        _, arm, variant = meta['namespace'].split('/')
        expected_facts = payload['comparators'][arm][variant]['pipeline_facts']
        stored_facts = json.loads(str(arrays['facts'][0]))
        if stored_facts != expected_facts: raise ArtifactError('cache/result pipeline facts differ')
        namespaces.add(meta['namespace'])
        roles['cache-meta:'+meta['namespace']] = sidecar
        roles['cache-payload:'+meta['namespace']] = npz
    if namespaces != {f'HELDOUT_GSE161529/{a}/{v}' for a in arms for v in variants}:
        raise ArtifactError('cache namespace set differs')
    return roles


def publish_set(result, attempt, roles, *, replace_pointer=None):
    """Validate every staged byte, then publish a SINGLE accepted pointer.

    Any exception at/after replace is explicitly uncertain; never restore an
    older pointer blindly. Failed attempt files are diagnostic-only. A caller
    resolves the pointer to determine acceptance before starting a fresh retry.
    """
    result, attempt = Path(result), Path(attempt)
    attempt_id = attempt.name
    entries = {}
    for role, path in roles.items():
        path = Path(path)
        relative = str(path.relative_to(attempt))
        entries[role] = {'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size}
    manifest = {'schema': SCHEMA, 'status': 'complete', 'attempt_id': attempt_id,
                'required_roles': sorted(roles), 'artifacts': entries}
    manifest_path = attempt/'manifest.json'
    atomic_json(manifest_path, manifest)
    # Re-read each byte and strict JSON before the sole commit operation.
    for role, entry in entries.items():
        path = _inside(attempt, entry['path'])
        if sha(path) != entry['sha256']: raise ArtifactError('staged artifact changed')
        if path.suffix == '.json': load_json_strict(path)
    pointer = pointer_path(result)
    pointer_payload = dict(schema=SCHEMA, attempt_id=attempt_id,
                           manifest=str(manifest_path.relative_to(result.parent)),
                           manifest_sha256=sha(manifest_path))
    # fsync complete staged files/directories before publishing their reference.
    for path in [*roles.values(), manifest_path]:
        with open(path,'rb') as handle: os.fsync(handle.fileno())
    for directory in {attempt, attempt/'cache'}:
        if directory.exists():
            fd = os.open(directory, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    fd, temporary = tempfile.mkstemp(prefix='.'+pointer.name, dir=result.parent)
    tmp = Path(temporary)
    try:
        with os.fdopen(fd,'w') as handle:
            import json
            handle.write(json.dumps(pointer_payload,sort_keys=True,allow_nan=False))
            handle.flush(); os.fsync(handle.fileno())
        try:
            (replace_pointer or os.replace)(tmp, pointer)
            fd = os.open(result.parent,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except Exception as exc:
            raise PublicationUncertain(f'accepted pointer publication uncertain; inspect {pointer}: {exc}') from exc
    finally:
        if tmp.exists(): tmp.unlink()
    return pointer


def run_breast_attempt(driver, required, result, calculate, arms, variants, *, argv=(), seed_cache=None):
    """Calculation/cache/gate failures cannot mutate the previous accepted set."""
    from .verification import compare_science, VerificationError
    result = Path(result)
    parent = result.parent / 'attempts' / result.stem
    parent.mkdir(parents=True,exist_ok=True)
    attempt = parent / (time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex)
    attempt.mkdir()
    try:
        missing = check_inputs(required)
        if missing:
            atomic_json(attempt/'failure.json',blocked_payload(driver,missing))
            raise SystemExit(2)
        previous_paths, previous_manifest = read_set(result)
        cache_root = attempt/'cache'; cache_root.mkdir()
        # Copy only the validated last accepted cache set; do not share mutable files.
        if previous_manifest['schema'] == SCHEMA:
            for role, path in previous_paths.items():
                if role.startswith(('cache-meta:', 'cache-payload:')):
                    shutil.copyfile(path,cache_root/path.name)
        elif seed_cache is not None:
            # Legacy warm cache is verified by ResultCache.load when actually used.
            for path in Path(seed_cache).glob('*'):
                if path.suffix == '.npz' or path.name.endswith('.meta.json'):
                    shutil.copyfile(path,cache_root/path.name)
        start = time.time()
        payload = calculate(attempt)
        payload.setdefault('driver',driver); payload.setdefault('status','ok')
        payload['wallclock_seconds'] = time.time()-start
        payload['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        if previous_paths['result'].exists():
            old = load_json_strict(previous_paths['result'])
            volatile = {'created_utc','wallclock_seconds','supersedes'}
            if old.get('status') != 'blocked':
                diffs = compare_science({k:v for k,v in payload.items() if k not in volatile},
                                        {k:v for k,v in old.items() if k not in volatile},mode='exact',provenance='self-reproduction')
                if diffs:
                    if '--update' not in argv: raise VerificationError('complete-field reproduction gate failed: '+ '; '.join(diffs[:10]))
                    payload['supersedes'] = {'first_differences':diffs[:20], 'n_changed_fields':len(diffs),
                                            'previous_created_utc':old.get('created_utc')}
        atomic_json(attempt/'result.json',payload)
        atomic_json(attempt/'provenance.json',{'attempt_id':attempt.name,
                    'loaded_driver_sha256':payload['loaded_driver_sha256'],
                    'method_version':payload['method_version'],
                    'patient_unit_evidence':payload['patient_unit_evidence'],
                    'whole_source_coverage':payload['whole_source_coverage'],
                    'execution_selection':payload['execution_selection']})
        roles = validate_breast_set(attempt,payload,arms,variants)
        publish_set(result,attempt,roles)
        print(f'accepted complete attempt {attempt.name}; pointer {pointer_path(result)}')
        return payload
    except Exception as exc:
        atomic_json(attempt/'failure.json',{'status':'publication_uncertain' if isinstance(exc,PublicationUncertain) else 'failed',
                    'attempt_id':attempt.name,'reason':str(exc)})
        raise
