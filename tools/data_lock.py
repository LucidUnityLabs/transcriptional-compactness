"""Build/verify a source-attributed SHA256 lock from acquisition receipts.

Only successfully downloaded bytes enter the lock. Derived artifacts are
separately identified; their content is never attributed to a source URL.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from lib.runner import atomic_json
DATA = ROOT/'data'
AUTHOR_COMMIT = '65cc0fe3fa1a72af8cfff0d3ba32c7545091cc4f'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def build():
    records={}
    for receipt in sorted(DATA.glob('acquisition_receipts*.json')):
        for e in json.loads(receipt.read_text()).get('files', []):
            if not e.get('path') or not e.get('url'): continue
            path=DATA/e['path']
            if not path.is_file(): continue
            digest=sha(path)
            if e.get('sha256') != digest: raise ValueError(f'receipt mismatch: {path}')
            rec={k:e[k] for k in ('path','url','source','sha256','publisher_md5') if k in e}
            rec['bytes']=path.stat().st_size
            records[e['path']]=rec
    for acc in ('GSE176031','GSE161529'):
        path=DATA/f'{acc}_family.soft.gz'
        if path.exists():
            records[path.name]={'path':path.name,'url':f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/soft/{acc}_family.soft.gz','source':f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}','sha256':sha(path),'bytes':path.stat().st_size}
    author=DATA/'GSE161529/author_sources'
    for p in sorted(author.rglob('*')):
        if p.is_file() and p.suffix in ('.R','.txt'):
            relative=str(p.relative_to(author));path=str(p.relative_to(DATA))
            records[path]={'path':path,'url':f'https://raw.githubusercontent.com/yunshun/HumanBreast10X/{AUTHOR_COMMIT}/{relative}','source':f'https://github.com/yunshun/HumanBreast10X/tree/{AUTHOR_COMMIT}','sha256':sha(p),'bytes':p.stat().st_size}
    features=DATA/'GSE161529/samples/GSE161529_features.tsv.gz'
    if features.exists():
        path=str(features.relative_to(DATA));records[path]={'path':path,'url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161529/suppl/GSE161529_features.tsv.gz','source':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161529','sha256':sha(features),'bytes':features.stat().st_size}
    matrix_manifest=DATA/'GSE161529/samples/MANIFEST_CORRECTED.json'
    if matrix_manifest.exists():
        manifest=json.loads(matrix_manifest.read_text())
        for e in manifest.get('files',manifest.get('downloads',[])):
            if not e.get('url'): continue
            # Acquisition receipts can use destination or path keys.
            name=e['url'].rsplit('/',1)[-1];p=matrix_manifest.parent/name
            if p.exists():
                rel=str(p.relative_to(DATA));records[rel]={'path':rel,'url':e['url'],'source':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161529','sha256':sha(p),'bytes':p.stat().st_size}
    contact_slices=[]
    for receipt in sorted((DATA/'hic_contact_slices').glob('*.json')):
        entry=json.loads(receipt.read_text())
        p=receipt.with_suffix('.npz')
        if not p.is_file() or sha(p)!=entry['sha256']:
            raise ValueError(f'consumed Hi-C slice mismatch: {p}')
        rel=str(p.relative_to(DATA))
        contact_slices.append(entry|{'path':rel,'source':'Rao et al. 2014 author-deposited remote Hi-C asset; observed/KR requested slice', 'bytes':p.stat().st_size})
    derived=[]
    for folder in ('GSE161529/axes_corrected','GSE161529/labels'):
        for p in sorted((DATA/folder).glob('*')):
            if p.is_file(): derived.append({'path':str(p.relative_to(DATA)),'sha256':sha(p),'bytes':p.stat().st_size,'kind':'derived; reproduce with tools/extract_breast_labels_corrected.py'})
    return {'format_version':1,'consumed_remote_slices':contact_slices,'status':'acquired_public_sources','claim':'Observed source bytes and attributed provenance; scientific endpoint validation is separate. Historical byte identity was not recorded and is not claimed.','files':sorted(records.values(),key=lambda e:e['path']),'derived_files':derived,'breast_author_commit':AUTHOR_COMMIT,'source_backed_donor_map':{'path':'config/GSE131907_specimen_donor.tsv','sha256':sha(ROOT/'config/GSE131907_specimen_donor.tsv'),'source':'GSE131907_Lung_Cancer_Feature_Summary.xlsx; unique sample/patient pairs rows 4–61'},'selected_visium_version':'10X CytAssist_FFPE_Human_Breast_Cancer Space Ranger 2.0.0; undocumented historical bytes not assumed identical'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');a=p.parse_args();dest=ROOT/'config/data.lock.json'
    if a.verify:
        d=json.loads(dest.read_text())
        for e in d['files']+d.get('derived_files',[])+d.get('consumed_remote_slices',[]):
            path=DATA/e['path']
            if not path.exists() or sha(path)!=e['sha256']:raise SystemExit(f'CONTENT_MISMATCH: {e["path"]}')
        print(f'verified {len(d["files"])} public sources and {len(d.get("derived_files",[]))} derived artifacts')
    else:
        d=build();atomic_json(dest,d);print(f'wrote {dest}: {len(d["files"])} sources')
if __name__=='__main__':main()
