"""Versioned receipts for complete emitted breast axes; legacy receipts stay immutable."""
from pathlib import Path
import hashlib
from .runner import load_json_strict, atomic_json

SCHEMA = 'tc-breast-axes-v2'
OBJECTS = ('ERTotal', 'HER2', 'TNBC', 'ERTotalSub', 'HER2Sub', 'TNBCSub')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def emitted_names(name):
    return [f'{name}_cells.tsv', f'{name}_cluster_crosswalk.tsv'] + (
        [f'{name}_marker_scores.tsv'] if name.endswith('Sub') else [])


def output_hashes(axes, name):
    return {filename: sha(Path(axes) / filename) for filename in emitted_names(name)}


def verify_axes(axes, name, receipt, *, rds_sha256=None, rscript_sha256=None):
    if receipt.get('schema') != SCHEMA or receipt.get('named_axes_verified') is not True:
        raise ValueError(f'{name}: complete versioned axes receipt required')
    if set(receipt.get('emitted_sha256', {})) != set(emitted_names(name)):
        raise ValueError(f'{name}: incomplete emitted axes hash coverage')
    for field, expected in [('rds_sha256', rds_sha256), ('rscript_sha256', rscript_sha256)]:
        if expected is not None and receipt.get(field) != expected:
            raise ValueError(f'{name}: extraction input changed: {field}')
    if output_hashes(axes, name) != receipt['emitted_sha256']:
        raise ValueError(f'{name}: emitted axes content changed')
    return receipt


def bind_reviewed_axes(geo, script, reviewed_repo):
    """Bind unchanged reviewed TSV bytes, without claiming a new R/RDS execution.

    reviewed_repo is the archived repository tree from the pre-edit receipt.
    Old native-R receipts and label manifest must also be byte-identical.
    """
    geo, reviewed_repo = Path(geo), Path(reviewed_repo)
    axes = geo / 'axes_corrected'
    old_geo = reviewed_repo / 'data/GSE161529'
    bindings = {}
    for name in OBJECTS:
        old_source = axes / f'{name}_source.json'
        if sha(old_source) != sha(old_geo / 'axes_corrected' / old_source.name):
            raise ValueError(f'{name}: reviewed native-R receipt changed')
        source = load_json_strict(old_source)
        if source.get('named_axes_verified') is not True or source['rscript_sha256'] != sha(script):
            raise ValueError(f'{name}: reviewed named-axis/input receipt invalid')
        observed = output_hashes(axes, name)
        if observed != output_hashes(old_geo / 'axes_corrected', name):
            raise ValueError(f'{name}: reviewed emitted axes changed')
        receipt = dict(source, schema=SCHEMA, emitted_sha256=observed,
                       legacy_source_receipt_sha256=sha(old_source),
                       binding_basis='byte-identical reviewed emitted axes; prior native-R input receipts, no RDS reparse or R rerun')
        path = axes / f'{name}_source_v2.json'
        if path.exists():
            if load_json_strict(path) != receipt:
                raise ValueError(f'{name}: existing v2 receipt differs; refuse overwrite')
        else:
            atomic_json(path, receipt)
        bindings[name] = dict(receipt_sha256=sha(path), **receipt)
    labels = geo / 'labels'
    legacy = labels / 'LABELS_MANIFEST_CORRECTED.json'
    if sha(legacy) != sha(old_geo / 'labels' / legacy.name):
        raise ValueError('reviewed label manifest changed')
    manifest = load_json_strict(legacy)
    import gzip
    for filename, expected in manifest['uncompressed_sha256'].items():
        if sha(labels / filename) != sha(old_geo / 'labels' / filename):
            raise ValueError('reviewed label bytes changed')
        h = hashlib.sha256()
        with gzip.open(labels / filename, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''): h.update(chunk)
        if h.hexdigest() != expected: raise ValueError('reviewed label content mismatch')
    manifest.update(provenance_schema=SCHEMA, complete_axes=bindings,
                    legacy_label_manifest_sha256=sha(legacy))
    target = labels / 'LABELS_MANIFEST_CORRECTED_V2.json'
    if target.exists():
        if load_json_strict(target) != manifest: raise ValueError('existing v2 manifest differs')
    else:
        atomic_json(target, manifest)
    return target


def verify_label_axes(geo, manifest, script):
    if manifest.get('provenance_schema') != SCHEMA or set(manifest.get('complete_axes', {})) != set(OBJECTS):
        raise ValueError('complete six-object axes provenance required')
    for name, binding in manifest['complete_axes'].items():
        path = Path(geo) / 'axes_corrected' / f'{name}_source_v2.json'
        if sha(path) != binding['receipt_sha256']: raise ValueError('axes receipt changed')
        receipt = load_json_strict(path)
        if receipt != {k:v for k,v in binding.items() if k != 'receipt_sha256'}:
            raise ValueError('label manifest axes binding differs')
        verify_axes(path.parent, name, receipt, rscript_sha256=sha(script))
