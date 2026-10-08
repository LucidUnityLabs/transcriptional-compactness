import numpy as np
import pandas as pd

from lib import cohort


def test_global_cap_preserves_both_group_minima():
    # The old protected ranges covered the first group's surplus rather
    # than the beginning of the second group; at the exact floor it lost
    # comparator cells despite having reserved the minimum budget.
    patient = np.repeat(["a", "b", "c"], 120)
    mal = np.tile(np.r_[np.ones(60, bool), np.zeros(60, bool)], 3)
    for seed in range(20):
        sel, audit = cohort.stratified_sample_minima_first(
            patient, mal, ~mal, min_per_group=10, total_cap=60, seed=seed)
        assert len(sel) == len(set(sel)) == 60
        for pid in np.unique(patient):
            for group in (mal, ~mal):
                assert np.sum((patient[sel] == pid) & group[sel]) == 10


def test_empty_measured_contrasts_are_explicit():
    cells = pd.DataFrame({"patient": ["a"], "is_mal": [True],
                          "status": ["not_sampled"], "kappa": [np.nan]})
    tab, estimable, exclusions = cohort.per_patient_contrasts(cells)
    assert tab.empty and "delta" in tab
    assert cohort.pool_cohort(tab, estimable)["status"] == "INSUFFICIENT_PATIENTS"


def test_directory_cache_digest_binds_relative_paths_and_content(tmp_path):
    from lib.cache import digest_of_inputs
    d=tmp_path/'source'; d.mkdir()
    (d/'a').write_text('same')
    first=digest_of_inputs([d])
    (d/'a').rename(d/'b')
    assert digest_of_inputs([d]) != first
    second=digest_of_inputs([d])
    (d/'b').write_text('changed')
    assert digest_of_inputs([d]) != second


def test_streamed_detection_universe_retains_named_selected_order(tmp_path):
    from lib.bio_io import read_umi_tsv_selected
    f=tmp_path/'counts.tsv'
    f.write_text('gene\ta\tb\tc\ng1\t0\t4\t2\ng2\t1\t0\t0\n')
    X,genes,detected=read_umi_tsv_selected(f,['c','a'],detection_cell_ids=['a','b','c'])
    assert genes == ['g1','g2']
    np.testing.assert_array_equal(X,[[2,0],[0,1]])
    np.testing.assert_array_equal(detected,[2,1])


def test_deposited_lung_donor_crosswalk_has_exact_coverage():
    from pathlib import Path
    # This curated artifact is a complete public specimen map, not a
    # sample-name heuristic; leading zeros in donor IDs are preserved.
    p=Path(__file__).resolve().parents[1]/'config/GSE131907_specimen_donor.tsv'
    d=pd.read_csv(p,sep='\t',dtype=str)
    assert len(d)==d['sample'].nunique()==58
    assert d['donor'].nunique()==44
    assert d['donor'].str.fullmatch('P[0-9]{4}').all()


def test_streamed_matrix_detection_deduplicates_coordinates(tmp_path):
    from lib.bio_io import read_mtx_selected
    f=tmp_path/'counts.mtx'
    f.write_text('%%MatrixMarket matrix coordinate integer general\n2 3 5\n1 1 2\n1 1 3\n1 2 1\n2 3 0\n2 2 4\n')
    x=read_mtx_selected(f,[1],detection_cols=[1,2,3])
    np.testing.assert_array_equal(x.counts,[[5,0]])
    np.testing.assert_array_equal(x.gene_detection,[2,1])
    only_detection=read_mtx_selected(f,[],detection_cols=[1,2,3])
    assert only_detection.counts.shape==(0,2)
    np.testing.assert_array_equal(only_detection.gene_detection,[2,1])


def test_explicit_sample_alias_keeps_injective_source_gate():
    import pytest
    from lib.acquisition import matrix_sources,AcquisitionError
    source={'GSM1':{'stem':'GSM1_ER-MH0040','matrix_url':'https://example/matrix','barcodes_url':'https://example/barcodes'}}
    alias={'ER_0040_T':{'gsm':'GSM1','source':'author primary tumor list + GEO donor/Total population'}}
    assert matrix_sources(['ER_0040_T'],source,explicit_aliases=alias)['ER_0040_T']['gsm']=='GSM1'
    with pytest.raises(AcquisitionError,match='non-injective'):
        matrix_sources(['ER_0040','ER_0040_T'],source,explicit_aliases=alias)
    with pytest.raises(AcquisitionError,match='invalid source'):
        matrix_sources(['ER_0040_T'],source,explicit_aliases={'ER_0040_T':{'gsm':'GSM1'}})


def test_compact_full_graph_distances_match_independent_networkx_reference():
    import networkx as nx
    from lib.numerics import _dijkstra
    rng=np.random.default_rng(76)
    for n in (5,17,41):
        G=nx.cycle_graph(n)
        for a,b in G.edges(): G[a][b]['weight']=float(rng.uniform(.001,8))
        for _ in range(n):
            a,b=map(int,rng.choice(n,2,replace=False))
            G.add_edge(a,b,weight=float(rng.uniform(.001,8)))
        G=nx.relabel_nodes(G,{i:f'cell-{i:03d}' for i in G})
        G.add_node('isolated')
        cache={}
        for a in G:
            actual=_dijkstra(G,a,cache,cache_limit=3)
            reference=nx.single_source_dijkstra_path_length(G,a,weight='weight')
            for b in G:
                assert (actual.get(b) is None)==(b not in reference)
                if b in reference: np.testing.assert_allclose(actual.get(b),reference[b],rtol=1e-13,atol=1e-13)


def test_extension_cache_replays_edge_draw_and_rng_and_honors_budget(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import networkx as nx
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('extension_adapter',root/'tools/rerun_extensions.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # The adapter's real scientific cache is isolated; its source module
    # still loads the untouched E3 design and the verified POT primitive.
    from lib.cache import ResultCache
    original=ResultCache.__init__
    monkeypatch.setattr(ResultCache,'__init__',lambda self,unused,namespace:original(self,tmp_path/'cache',namespace))
    out=tmp_path/'outputs';out.mkdir()
    adapter,records=mod.prepare('E3_puram_hnscc',out)
    G=nx.cycle_graph(5)
    nx.set_edge_attributes(G,1.,'weight')
    r1=np.random.default_rng(19)
    a=adapter.ollivier_ricci_edges(G,alpha=.5,max_edges=3,rng=r1)
    assert len(a)==3
    next1=r1.random()
    r2=np.random.default_rng(19)
    b=adapter.ollivier_ricci_edges(G,alpha=.5,max_edges=3,rng=r2)
    assert b==a and r2.random()==next1
    assert len(records)==2
    assert all(np.isclose(k,.25) for k in a.values())


def test_classifier_missing_mitochondrial_measurement_is_unknown(tmp_path):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('qc_adapter',root/'tools/rerun_extensions.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    adapter,_=module.prepare('CLASSIFIER_AUC',tmp_path)
    libsize,detected,pct=adapter.compute_qc_features(np.array([[2.,0.],[4.,7.]]),['a','b'])
    np.testing.assert_array_equal(libsize,[6,7]);np.testing.assert_array_equal(detected,[2,1])
    assert np.isnan(pct).all()
    import pytest
    with pytest.raises(ValueError,match='invalid assay'):
        adapter.compute_qc_features(np.array([[-1.,0.]]),['a'],is_log1p=True)


def test_label_verdict_missing_probability_is_inconclusive():
    from lib.verification import label_verdict
    result=label_verdict({'original':{'ok':True,'delta':.5}},['original'])
    assert result['verdict']=='inconclusive'
    assert 'nonfinite p' in result['problems'][0]


def test_cache_code_identity_is_bound_at_import_not_later_file_edits(tmp_path,monkeypatch):
    from pathlib import Path
    from lib.cache import ResultCache
    rng=np.random.default_rng(121)
    X=rng.normal(size=(30,80))
    patient=np.repeat(['a','b'],40)
    mal=np.tile(np.r_[np.ones(20,bool),np.zeros(20,bool)],2)
    cache=ResultCache(tmp_path,'loaded-code')
    first,facts=cohort.compute_cohort_kappa(X,patient,mal,~mal,k=3,n_edges=20,seed=13,min_per_group=3,cache=cache,cache_inputs_digest='same-source')
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'newer on-disk source' if p.suffix=='.py' else read(p))
    second,repeat=cohort.compute_cohort_kappa(X,patient,mal,~mal,k=3,n_edges=20,seed=13,min_per_group=3,cache=cache,cache_inputs_digest='same-source')
    pd.testing.assert_frame_equal(first,second)
    assert 'cache_hit_reason' in repeat


def test_changed_normalized_input_cannot_reuse_same_raw_source_cache(tmp_path):
    from lib.cache import ResultCache
    rng=np.random.default_rng(99)
    X=rng.normal(size=(20,60));patient=np.repeat(['a','b'],30)
    mal=np.tile(np.r_[np.ones(15,bool),np.zeros(15,bool)],2)
    cache=ResultCache(tmp_path,'actual-input')
    _,first=cohort.compute_cohort_kappa(X,patient,mal,~mal,k=3,n_edges=20,seed=1,min_per_group=3,cache=cache,cache_inputs_digest='same-raw-source')
    changed=X.copy();changed[0,0]+=3
    _,second=cohort.compute_cohort_kappa(changed,patient,mal,~mal,k=3,n_edges=20,seed=1,min_per_group=3,cache=cache,cache_inputs_digest='same-raw-source')
    assert 'cache_hit_reason' not in second
    assert len(list(tmp_path.glob('*.meta.json')))==2


def test_author_yost_response_table_keeps_clinical_footnote_and_lesion_identity():
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('yost_source',root/'tools/yost_response_crosswalk.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if not mod.SOURCE.exists():
        import pytest
        pytest.skip('optional public author clinical workbook unavailable')
    responses=mod.extract()
    assert responses['su002']=='Responder'
    assert responses['su010']=='Non-responder' and 'su010-S' not in responses
    assert len(responses)==11 and sum(v=='Responder' for v in responses.values())==6


def test_unlazy_and_fully_lazy_transport_boundaries_and_fixed_edge_identity():
    from lib.numerics import ricci_edges
    import networkx as nx
    import pytest
    G=nx.complete_graph(3);nx.set_edge_attributes(G,2.,'weight')
    for alpha,expected in [(0.,.5),(.5,.75),(1.,0.)]:
        measured=ricci_edges(G,alpha=alpha,selected_edges=[(2,0)],backend='pot')
        assert list(measured)==[(2,0)]
        assert measured[(2,0)].kappa==pytest.approx(expected,abs=1e-13)
    for bad in [[(0,1),(1,0)],[(0,4)]]:
        with pytest.raises(ValueError,match='unique actual'):
            ricci_edges(G,selected_edges=bad)
    with pytest.raises(ValueError,match='mutually exclusive'):
        ricci_edges(G,n_edges=1,selected_edges=[(0,1)])


def test_embedded_random_gene_control_uses_verified_transport(tmp_path):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('random_control',root/'tools/rerun_extensions.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    adapter,records=mod.prepare('T6_random_hvg_control',tmp_path)
    adapter.K_NN=3;adapter.N_EDGES=15
    X=np.random.default_rng(7).normal(size=(20,5))
    result=adapter.run_pipeline(X,np.array(['Malignant']*10+['T cells']*10),19)
    assert result['n_edges_sampled']==15 and len(records)==1
    assert len(records[0]['measured_edge_ids'])==15
    assert np.isfinite(result['cliff_delta'])


def test_spatial_adapter_preserves_delaunay_design_and_physical_neighbor_budget(tmp_path):
    import pytest
    pytest.importorskip('scanpy')
    pytest.importorskip('h5py')
    import importlib.util
    from pathlib import Path
    import types, pandas as pd
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('spatial_adapter',root/'tools/rerun_extensions.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    visium,_=mod.prepare('T4_visium_spatial',tmp_path)
    pts=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    fake=types.SimpleNamespace(obs=pd.DataFrame({'pxl_row':pts[:,0],'pxl_col':pts[:,1]}),n_obs=4)
    G=visium.build_phys_graph(fake)
    assert len(G)==4 and G.number_of_edges()==5
    hd,_=mod.prepare('N2_visium_hd',tmp_path)
    import inspect
    assert inspect.signature(hd.build_phys_graph).parameters['k'].default==6


def test_acquisition_lock_only_command_accepts_no_positional_accessions(tmp_path,monkeypatch):
    import importlib.util, json, sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('acquisition_cli',root/'tools/acquire_public.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    (tmp_path/'data').mkdir();lock=tmp_path/'lock.json';lock.write_text('{"files": []}')
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['acquire_public','--lock',str(lock)])
    mod.main()
    assert json.loads((tmp_path/'data/acquisition_receipts_locked.json').read_text())['status']=='ok'


def test_source_identity_exclusion_removes_whole_specimen_and_binds_actual_bytes(tmp_path,monkeypatch):
    import importlib.util,hashlib,json,gzip,pandas as pd
    from pathlib import Path
    import pytest
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('source_exclusion_driver',root/'experiments/HELDOUT_GSE161529/run_corrected.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    labels=tmp_path/'labels';labels.mkdir();samples=tmp_path/'samples';samples.mkdir()
    source=labels/'cells_primary.tsv.gz'
    pd.DataFrame({'barcode':['bad_A-1','bad_B-1','good_C-1','good_D-1'], 'sample':['bad','bad','good','good'], 'patient':['0001','0001','0025','0025'], 'label':['malignant','immune','malignant','immune']}).to_csv(source,sep='\t',index=False,compression='gzip')
    raw=samples/'bad-barcodes.tsv.gz'
    with gzip.open(raw,'wt') as f:f.write('A-1\nUNRELATED-1\n')
    exclusions=tmp_path/'exclusions.json';exclusions.write_text(json.dumps({'samples':{'bad':{'donor':'0001','raw_barcode_file':raw.name,'raw_barcode_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'label_file_sha256':{source.name:hashlib.sha256(source.read_bytes()).hexdigest()},'reason':'synthetic same-cell source mismatch','required_to_restore':'authenticated same-cell source'}}}))
    monkeypatch.setattr(driver,'LAB_DIR',labels);monkeypatch.setattr(driver,'SAMPLES_DIR',samples);monkeypatch.setattr(driver,'SOURCE_EXCLUSIONS',exclusions)
    before=source.read_bytes();kept,evidence=driver.resolve_labels('primary','excluded')
    assert list(kept.patient)==['0025','0025'] and source.read_bytes()==before
    assert evidence['source_exclusions'][0]['n_labels_removed']==2
    with gzip.open(raw,'wt') as f:f.write('replacement\n')
    with pytest.raises(driver.DataValidationError,match='barcode content changed'):
        driver.resolve_labels('primary','excluded')


def test_authentic_breast_full_source_preflight_without_numerical_work(monkeypatch):
    import importlib.util
    from pathlib import Path
    import pytest
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('breast_full_preflight',root/'experiments/HELDOUT_GSE161529/run_corrected.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    if not all(path.exists() for _,path,_ in driver.REQUIRED_INPUTS):
        pytest.skip('authentic breast source artifacts are optional local research data')
    def forbid_numerical(*args,**kwargs):
        raise AssertionError('provenance-only preflight must not calculate an endpoint')
    monkeypatch.setattr(driver,'run_variant',forbid_numerical)
    result=driver.run(arms=(),variants=())
    assert result['patient_unit_verified'] is True
    assert result['comparators']=={}
    coverage=result['whole_source_coverage']
    assert coverage['n_original_source_specimens']==27
    assert coverage['n_eligible_source_specimens']==26
    assert coverage['excluded_specimens']==['ER_0001']
    assert len(coverage['rows'])==54
    assert sum(row['n_absent']==0 for row in coverage['rows'])==52
