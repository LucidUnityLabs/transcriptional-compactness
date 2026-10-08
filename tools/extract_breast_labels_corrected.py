"""Corrected, source-backed Seurat factor/axis label extraction (D03/D04).

Unlike the historical extractor, author cluster codes are factor level
positions as used by the deposited R scripts. Every RNA axis and every
sample/cluster is checked before deterministic label tables are written.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from lib.acquisition import matrix_sources, parse_soft_supplementary
from lib.runner import atomic_json, load_json_strict
from lib.axes_provenance import (SCHEMA, verify_axes, output_hashes, bind_reviewed_axes, emitted_names as output_hashes_names)

GEO = ROOT / 'data/GSE161529'
OBJECTS = ['ERTotal', 'HER2', 'TNBC', 'ERTotalSub', 'HER2Sub', 'TNBCSub']
IMMUNE = {'BCell', 'TCell', 'TCell2', 'NK', 'DC', 'Macro'}
PINNED_T = {'ERTotalSub': {1, 8}, 'HER2Sub': {2, 7}, 'TNBCSub': {1, 5}}
PINNED_EXTRA = {'ERTotalSub': {7}, 'HER2Sub': {9}}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


def deterministic_tsv(frame, path):
    content = frame.sort_values(['sample', 'barcode', 'label']).to_csv(sep='\t', index=False).encode()
    with open(path, 'wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as f: f.write(content)
    return hashlib.sha256(content).hexdigest()


def tumor_blocks():
    text = (GEO/'author_sources/Tables/InferCNV-Annotation.txt').read_text()
    blocks = {}
    object_name = None
    for line in text.splitlines():
        if line.startswith('##### '):
            object_name = {'ER-Total': 'ERTotal', 'HER2': 'HER2', 'TNBC': 'TNBC'}.get(line.split()[1])
        match = re.search(r'\(cluster (\d+)\): (.+?);', line)
        if match and object_name:
            stored = int(match.group(1))
            samples = [s.strip() for s in match.group(2).split(',')]
            for sample in samples:
                if (sample, stored) in blocks.setdefault(object_name, {}): raise ValueError('duplicate tumor annotation')
                blocks[object_name][sample, stored] = True
    if set(blocks) != {'ERTotal', 'HER2', 'TNBC'}: raise ValueError('author annotation incomplete')
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=OBJECTS)
    parser.add_argument('--bind-reviewed-axes', type=Path, help='pre-edit archived repository tree; bind unchanged reviewed outputs without R execution')
    args = parser.parse_args()
    if args.bind_reviewed_axes:
        print(bind_reviewed_axes(GEO, ROOT/'tools/extract_breast_axes.R', args.bind_reviewed_axes))
        return
    axes = GEO/'axes_corrected'; axes.mkdir(exist_ok=True, parents=True)
    for name in ([args.object] if args.object else OBJECTS):
        source = GEO/'figshare_tmp'/f'SeuratObject_{name}.rds'
        receipt = axes/f'{name}_source_v2.json'
        source_hash = sha(source)
        script_hash = sha(ROOT/'tools/extract_breast_axes.R')
        if receipt.exists():
            verify_axes(axes, name, load_json_strict(receipt),
                        rds_sha256=source_hash, rscript_sha256=script_hash)
            continue
        if any((axes / filename).exists() for filename in output_hashes_names(name)):
            raise ValueError('existing axes require reviewed v2 binding; refuse unverified reuse/overwrite')
        subprocess.run(['Rscript', str(ROOT/'tools/extract_breast_axes.R'), str(source), str(axes)], check=True)
        atomic_json(receipt, {'schema': SCHEMA, 'rds_sha256': source_hash, 'rscript_sha256': script_hash,
                              'emitted_sha256': output_hashes(axes, name),
                              'cluster_rule': 'R as.integer(factor) = level position; never cast printed labels',
                              'named_axes_verified': True})
    if args.object: return
    blocks = tumor_blocks()
    frames, immune_frames, checks = [], [], []
    soft = GEO/'GSE161529_family.soft.gz'
    supp = parse_soft_supplementary(soft)
    sample_donors = {}
    text = gzip.decompress(soft.read_bytes()).decode()
    for block in text.split('^SAMPLE = ')[1:]:
        gsm = block.splitlines()[0].strip()
        match = re.search(r'!Sample_characteristics_ch1 = patient: (.+)', block)
        if match: sample_donors[gsm] = match.group(1).strip()
    all_cells = {n: pd.read_csv(axes/f'{n}_cells.tsv', sep='\t', dtype={'sample': str, 'cluster_stored': str}) for n in OBJECTS}
    aliases = json.loads((ROOT/'config/GSE161529_sample_aliases.json').read_text())
    for sample, alias in aliases.items():
        if alias['author_sample'] not in (GEO/'author_sources/RCode/ER.R').read_text():
            raise ValueError('alias absent from author sample list')
        block = next(b for b in text.split('^SAMPLE = ')[1:] if b.splitlines()[0].strip() == alias['gsm'])
        if f"patient: {alias['geo_patient']}\n" not in block or f"cell population: {alias['geo_population']}\n" not in block:
            raise ValueError('alias GEO specimen/donor evidence mismatch')
    mapping = matrix_sources(set(s for d in all_cells.values() for s in d['sample']), supp, explicit_aliases=aliases)
    for name, cells in all_cells.items():
        cells['patient'] = cells['sample'].map(lambda s: sample_donors[mapping[s]['gsm']])
        if cells['barcode'].duplicated().any(): raise ValueError(f'{name}: duplicate barcodes')
        if name in blocks:
            # Crosswalk explicitly relates every printed factor label to
            # the author's R integer code and annotation printed cluster.
            xwalk = cells[['sample', 'cluster_stored', 'cluster']].drop_duplicates()
            if not all(int(r.cluster_stored)+1 == r.cluster for r in xwalk.itertuples()):
                raise ValueError('deposited factor mapping differs from companion code: review complete crosswalk')
            epi_codes = set(k+1 for s,k in blocks[name])
            cells['label'] = [
                'malignant' if blocks[name].get((r.sample, int(r.cluster_stored)), False)
                else 'normal_epithelial' if r.cluster in epi_codes else 'other'
                for r in cells.itertuples()]
            observed = set(zip(cells['sample'], cells['cluster_stored'].astype(int)))
            absent = set(blocks[name]) - observed
            if absent: raise ValueError(f'{name}: tumor annotation references absent sample/cluster: {absent}')
            frames.append(cells)
        else:
            scores = pd.read_csv(axes/f'{name}_marker_scores.tsv', sep='\t')
            if not np.isfinite(scores['score']).all(): raise ValueError('nonfinite marker score')
            assignments = {}
            for cl, d in scores.groupby('cluster'):
                vals = dict(zip(d['panel'], d['score']))
                top = max(vals, key=vals.get)
                nonimmune = max(v for k,v in vals.items() if k not in IMMUNE)
                panel_call = top in IMMUNE and vals[top] - nonimmune > .1
                assignments[cl] = panel_call or cl in PINNED_T[name] or cl in PINNED_EXTRA.get(name, set())
            if set(cells['cluster']) != set(assignments): raise ValueError('incomplete immune cluster score set')
            cells['label'] = cells['cluster'].map(lambda c: 'immune' if assignments[c] else 'other')
            immune_frames.append(cells)
        checks.append({'object': name, 'n_cells': len(cells), 'n_samples': cells['sample'].nunique(),
                       'n_sample_cluster_rows': len(cells[['sample','cluster']].drop_duplicates())})
    full, sub = pd.concat(frames), pd.concat(immune_frames)
    # The authors construct Sub by excluding epithelial Total cells.
    # This is a complete barcode crosswalk gate, not a few cluster sets.
    barcode_crosswalks = []
    for total, subset in [('ERTotal','ERTotalSub'), ('HER2','HER2Sub'), ('TNBC','TNBCSub')]:
        epi = full[(full['object']==total) & full['label'].isin(['malignant','normal_epithelial'])]
        micro = sub[sub['object']==subset]
        overlap = set(epi['barcode']) & set(micro['barcode'])
        if overlap: raise ValueError(f'{total}/{subset}: {len(overlap)} epithelial/microenvironment barcode conflicts')
        if set(micro['barcode']) - set(full[full['object']==total]['barcode']): raise ValueError('Sub cells absent from Total')
        barcode_crosswalks.append({'total':total,'subset':subset,'n_total_cells':int((full.object==total).sum()),'n_subset_cells':len(micro),'n_epithelial_subset_conflicts':0,'n_subset_cells_absent_from_total':0})
    primary = pd.concat([full[full.label=='malignant'], sub[sub.label=='immune']])
    secondary = full[full.label.isin(['malignant','normal_epithelial'])]
    lab = GEO/'labels_v2'; lab.mkdir(exist_ok=True)
    if any(lab.iterdir()): raise ValueError('labels_v2 already exists; refuse overwrite')
    bindings = {}
    for name in OBJECTS:
        path = axes/f'{name}_source_v2.json'
        bindings[name] = dict(receipt_sha256=sha(path), **verify_axes(axes, name, load_json_strict(path)))
    cols = ['barcode','sample','patient','object','cluster','label']
    hashes = {name: deterministic_tsv(df[cols], lab/name) for name,df in
              [('cells_primary.tsv.gz',primary),('cells_secondary.tsv.gz',secondary)]}
    atomic_json(lab/'LABELS_MANIFEST_CORRECTED_V2.json', {'provenance_schema':SCHEMA, 'complete_axes':bindings, 'status':'ok','method_version':'TC-1', 'objects':checks,
        'uncompressed_sha256': hashes, 'n_primary_cells':len(primary),'n_secondary_cells':len(secondary),
        'barcode_crosswalks':barcode_crosswalks,'sample_aliases':aliases,
        'donor_source':'GSE161529 GEO family SOFT patient characteristic, exact GSM/sample source mapping',
        'immune_rule':'author marker means + 0.1 margin + documented pinned overrides; pipeline-derived biological labels',
        'luminal_endpoint':'unavailable: absence of inferCNV malignant call is not affirmative normal luminal status'})
    print('validated labels',len(primary),len(secondary),flush=True)


if __name__ == '__main__': main()
