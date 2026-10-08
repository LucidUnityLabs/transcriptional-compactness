"""Small independent corruption/failure fixtures; no biological calculation."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from lib import publication as pub
from lib.axes_provenance import SCHEMA, emitted_names, output_hashes, verify_axes
from lib.runner import atomic_json, ArtifactError, load_json_strict
from lib.cache import ResultCache

ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location('tc_successor_fixture',ROOT/'experiments/HELDOUT_GSE161529/run_corrected.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mutation',['missing','tampered'])
@pytest.mark.parametrize('member',emitted_names('HER2Sub'))
def test_axes_refuse_each_incomplete_or_changed_emission(tmp_path, member, mutation):
    for name in emitted_names('HER2Sub'): (tmp_path/name).write_text('reviewed fixture '+name)
    receipt = {'schema':SCHEMA,'named_axes_verified':True,'rds_sha256':'a'*64,'rscript_sha256':'b'*64,
               'emitted_sha256':output_hashes(tmp_path,'HER2Sub')}
    verify_axes(tmp_path,'HER2Sub',receipt,rds_sha256='a'*64,rscript_sha256='b'*64)
    if mutation == 'missing': (tmp_path/member).unlink()
    else: (tmp_path/member).write_text('changed')
    with pytest.raises((ValueError,FileNotFoundError)): verify_axes(tmp_path,'HER2Sub',receipt)


@pytest.mark.parametrize('ncols',[1,3])
@pytest.mark.parametrize('detection',[False,True])
def test_complete_triple_gate_precedes_selected_column_work(tmp_path,monkeypatch,ncols,detection):
    import pandas as pd
    d = driver()
    stem = 'fixture'
    with gzip.open(tmp_path/f'{stem}-barcodes.tsv.gz','wt') as handle: handle.write('selected\nunselected\n')
    # Only column one is requested/detected, so both headers evade old selected gates.
    # Invalid trailing content proves rejection occurs at the dimension gate.
    with gzip.open(tmp_path/f'{stem}-matrix.mtx.gz','wt') as handle:
        handle.write(f'%%MatrixMarket matrix coordinate integer general\n1 {ncols} 1\ninvalid trailing entry\n')
    monkeypatch.setattr(d,'SAMPLES_DIR',tmp_path);monkeypatch.setattr(d,'load_features',lambda:['gene'])
    labels = pd.DataFrame({'sample':['s'],'barcode':['s_selected'],'patient':['0001'],'label':['malignant']})
    with pytest.raises(d.DataValidationError,match='complete barcode count'):
        d.load_counts_strict(labels,{'s':{'stem':stem}},detection_labels=labels if detection else None)


def calculate_fixture(attempt, delta=0.2):
    row = {'patient_id':'0001','n_malignant':10,'n_comparator':11,'delta':delta,'se':0.1}
    with (attempt/'per_patient_primary_excluded_corrected.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    namespace='HELDOUT_GSE161529/primary/excluded'
    cache=ResultCache(attempt/'cache',namespace)
    key=cache.key('input','params')
    facts={'metric':'fixture-only'}
    cache.save(key,{'patient':np.array(['0001']),'is_mal':np.array([True]),'kappa':np.array([.2]),
                   'degree':np.array([1]),'n_incident_measured':np.array([1]),'coverage_fraction':np.array([1.]),
                   'status':np.array(['measured']),'facts':np.array([json.dumps(facts)])},
               {'inputs_digest':'input','params_digest':'params'})
    base=f'{namespace.replace("/","__")}_{key}'
    atomic_json(attempt/'cache_primary_excluded.json',{'namespace':namespace,'key':key,'sidecar':base+'.meta.json','payload':base+'.npz'})
    return {'method_version':'TC-1','loaded_driver_sha256':'fixture','patient_unit_evidence':{'verified':True},
            'whole_source_coverage':{'scope':'synthetic'},'execution_selection':{'arms':['primary'],'variants':['excluded']},
            'comparators':{'primary':{'excluded':{'per_patient':{'0001':{k:v for k,v in row.items() if k != 'patient_id'}},'pipeline_facts':facts}}}}


def run_fixture(result,calculate=calculate_fixture,argv=()):
    return pub.run_breast_attempt('fixture',[],result,calculate,('primary',),('excluded',),argv=argv)


def snapshot(result):
    paths,m=pub.read_set(result,require_pointer=True)
    return pub.pointer_path(result).read_bytes(), {role:path.read_bytes() for role,path in paths.items()}


@pytest.mark.parametrize('stage',['calculate','cache','pool','result','provenance','validate','manifest','rename'])
def test_each_prepublication_failure_preserves_accepted_set_and_retry(tmp_path,monkeypatch,stage):
    result=tmp_path/'results_primary_excluded_corrected.json'
    # Legacy flat bytes also remain immutable after adoption.
    result.write_text('{"status":"blocked"}')
    flat=result.read_bytes()
    run_fixture(result); before=snapshot(result)
    with monkeypatch.context() as patch:
        if stage in {'calculate','cache','pool'}:
            def failed(attempt):
                if stage != 'calculate': calculate_fixture(attempt)
                raise RuntimeError(stage+' failure')
            call=lambda:run_fixture(result,failed)
        elif stage == 'validate':
            patch.setattr(pub,'validate_breast_set',lambda *a: (_ for _ in ()).throw(RuntimeError('validate failure')))
            call=lambda:run_fixture(result)
        elif stage == 'rename':
            patch.setattr(pub.os,'replace',lambda *a: (_ for _ in ()).throw(OSError('rename failure')))
            call=lambda:run_fixture(result)
        else:
            original=pub.atomic_json
            def failed_write(path,payload):
                if Path(path).name == stage+'.json': raise OSError(stage+' failure')
                return original(path,payload)
            patch.setattr(pub,'atomic_json',failed_write)
            call=lambda:run_fixture(result)
        with pytest.raises((RuntimeError,OSError,ArtifactError)):call()
    assert snapshot(result)==before and result.read_bytes()==flat
    run_fixture(result)
    assert snapshot(result)[1]==before[1] or pub.load_result(result)['comparators']==pub.load_result(result)['comparators']
    assert result.read_bytes()==flat


def test_gate_failure_and_blocked_inputs_preserve_last_set(tmp_path):
    result=tmp_path/'results.json';run_fixture(result);before=snapshot(result)
    from lib.verification import VerificationError
    with pytest.raises(VerificationError):run_fixture(result,lambda p:calculate_fixture(p,.3))
    assert snapshot(result)==before
    with pytest.raises(SystemExit) as exc:
        pub.run_breast_attempt('fixture',[('missing',tmp_path/'absent','fixture')],result,calculate_fixture,('primary',),('excluded',))
    assert exc.value.code==2 and snapshot(result)==before
    run_fixture(result,lambda p:calculate_fixture(p,.3),argv=('--update',))
    assert pub.load_result(result)['comparators']['primary']['excluded']['per_patient']['0001']['delta']==.3
    for role,path in pub.read_set(result)[0].items(): assert path.is_file()


@pytest.mark.parametrize('role',['result','provenance','table:primary:excluded','cache-meta:HELDOUT_GSE161529/primary/excluded','cache-payload:HELDOUT_GSE161529/primary/excluded'])
@pytest.mark.parametrize('mutation',['missing','tamper'])
def test_read_set_rejects_corruption_without_stale_fallback(tmp_path,role,mutation):
    # Stale flat bytes belong beside an accepted pointer: a flat result that
    # exists before the first attempt is a prior science payload and must
    # pass the self-reproduction gate, so stage the legacy bytes after adoption.
    result=tmp_path/'results.json';run_fixture(result);result.write_text('{"status":"stale"}')
    paths,_=pub.read_set(result);path=paths[role]
    if mutation=='missing':path.unlink()
    else:path.write_bytes(b'changed')
    with pytest.raises(ArtifactError):pub.read_set(result)


def test_postrename_uncertainty_is_truthful_and_old_attempt_retained(tmp_path,monkeypatch):
    result=tmp_path/'results.json';run_fixture(result);old_paths,_=pub.read_set(result)
    old={role:path.read_bytes() for role,path in old_paths.items()}
    original=pub.os.replace
    # Ack loss is simulated only where the protocol calls the outcome
    # uncertain: the accepted-pointer rename and the failure receipt of the
    # same attempt.  A process-wide fault would first strike the calculation
    # cache rename, which is an ordinary failure, not publication uncertainty.
    uncertain_names={pub.pointer_path(result).name,'failure.json'}
    def uncertain(src,dst):
        original(src,dst)
        if Path(dst).name in uncertain_names:raise OSError('ack lost after rename')
    with monkeypatch.context() as patch:
        patch.setattr(pub.os,'replace',uncertain)
        with pytest.raises(pub.PublicationUncertain):run_fixture(result,lambda p:calculate_fixture(p,.3),argv=('--update',))
    assert {role:path.read_bytes() for role,path in old_paths.items()}==old
    assert pub.load_result(result)['comparators']['primary']['excluded']['per_patient']['0001']['delta']==.3
    failures=list(tmp_path.rglob('failure.json'))
    assert load_json_strict(failures[-1])['status']=='publication_uncertain'
    run_fixture(result,lambda p:calculate_fixture(p,.3))


def test_table_reader_compatibility_uses_one_pointer_snapshot(tmp_path):
    result=tmp_path/'results.json';table=tmp_path/'per_patient_primary_excluded_corrected.csv'
    table.write_text('stale flat csv');run_fixture(result)
    resolved_table,resolved_result=pub.resolve_table_result(result,table)
    assert resolved_table.parent==resolved_result.parent and resolved_table != table
    assert table.read_text()=='stale flat csv'
    with pytest.raises(ArtifactError):pub.resolve_table_result(result,tmp_path/'foreign.csv')
