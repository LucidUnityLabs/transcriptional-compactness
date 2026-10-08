"""Verify whole source cell universes before any breast sampling/calculation.

Preserves full label files. Explicit whole-specimen exclusions require exact
source hashes and never retain coincidental barcode overlaps. --write saves
a reproducible coverage manifest; default verifies every field against it.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/GSE161529'
DEST = ROOT / 'config/GSE161529_raw_label_coverage.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    labels = DATA / 'labels'
    samples = DATA / 'samples'
    source = json.loads((samples / 'MANIFEST.json').read_text())['mapping']
    exclusions_path = ROOT / 'config/GSE161529_source_exclusions.json'
    exclusions = json.loads(exclusions_path.read_text())['samples']
    raw = {}
    raw_hashes = {}
    for sample, mapping in source.items():
        path = samples / (mapping['stem'] + '-barcodes.tsv.gz')
        with gzip.open(path, 'rt') as handle:
            cells = [line.strip() for line in handle]
        if len(cells) != len(set(cells)):
            raise ValueError(f'duplicate source barcode: {sample}')
        raw[sample] = set(cells)
        raw_hashes[sample] = sha(path)
    rows, label_hashes = [], {}
    specimen_sets = []
    for filename in ('cells_primary.tsv.gz', 'cells_secondary.tsv.gz'):
        path = labels / filename
        label_hashes[filename] = sha(path)
        table = pd.read_csv(path, sep='\t', dtype=str)
        specimen_sets.append(set(table['sample']))
        for sample, frame in table.groupby('sample', sort=True):
            prefix = sample + '_'
            if not frame.barcode.str.startswith(prefix).all():
                raise ValueError(f'invalid author specimen prefix: {sample}')
            cells = set(frame.barcode.str[len(prefix):])
            absent = cells - raw[sample]
            if sample in exclusions:
                evidence = exclusions[sample]
                if (evidence['raw_barcode_sha256'] != raw_hashes[sample]
                        or evidence['label_file_sha256'][filename] != label_hashes[filename]
                        or set(frame.patient) != {evidence['donor']}):
                    raise ValueError(f'exclusion content/identity changed: {sample}')
                status = 'whole specimen excluded: source cell-universe mismatch'
            elif absent:
                raise ValueError(f'{sample}: {len(absent)} full label cells absent from raw source')
            else:
                status = 'complete exact source barcode coverage'
            rows.append({'label_file': filename, 'sample': sample,
                         'gsm': source[sample]['gsm'], 'donors': sorted(set(frame.patient)),
                         'n_labels': len(cells), 'n_raw_cells': len(raw[sample]),
                         'n_exact_matches': len(cells & raw[sample]), 'n_absent': len(absent),
                         'raw_barcode_sha256': raw_hashes[sample], 'status': status})
    if specimen_sets[0] != specimen_sets[1] or specimen_sets[0] != set(source):
        raise ValueError('full both-arm specimen coverage differs from acquisition mapping')
    return {'status': 'all eligible specimens have complete exact barcode coverage',
            'original_endpoint_coverage': 'incomplete; explicit whole-source exclusions',
            'n_original_source_specimens': len(source),
            'n_eligible_source_specimens': len(set(source) - set(exclusions)),
            'excluded_specimens': sorted(exclusions), 'label_file_sha256': label_hashes,
            'exclusions_sha256': sha(exclusions_path),
            'label_extraction_manifest_sha256': sha(labels / 'LABELS_MANIFEST_CORRECTED.json'),
            'loaded_verifier_sha256': sha(Path(__file__)), 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    actual = build()
    if args.write:
        DEST.write_text(json.dumps(actual, indent=2, sort_keys=True) + '\n')
    elif json.loads(DEST.read_text()) != actual:
        raise SystemExit('FULL_BREAST_SOURCE_COVERAGE_MISMATCH')
    print(f"verified {actual['n_eligible_source_specimens']}/{actual['n_original_source_specimens']} specimens; exclusions {actual['excluded_specimens']}; full label files preserved")
