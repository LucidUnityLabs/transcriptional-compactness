"""Run preserved exploratory designs with TC-1 graph/transport primitives.

Historical files are never edited or overwritten. Results live under each
experiment's corrected/ directory. These retain the exploratory contrast,
assay scale and cell-level tests; they are NOT confirmatory donor inference.
The four decision-critical analyses use their dedicated corrected drivers.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import copy
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from lib import METHOD_VERSION, numerics
from lib.runner import atomic_json, load_json_strict

IMPORTED_ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ReferenceChoices(ast.NodeTransformer):
    """Make precision/SVD choices explicit, without changing assay scales."""
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "float32":
            node.attr = "float64"
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "PCA":
            node.keywords = [k for k in node.keywords if k.arg != "svd_solver"]
            node.keywords.append(ast.keyword(arg="svd_solver", value=ast.Constant("full")))
        return node


def prepare(experiment, out):
    source = ROOT / "experiments" / experiment / "run.py"
    text = source.read_text()
    if experiment == 'F4_hic_genome_wide':
        text = text.replace('robust cancer-vs-normal curvature signature', 'descriptive K562-versus-GM12878 curvature difference')
        text = text.replace('The Flavahan/Hnisz TAD-disorganization story is correct;', 'This cell-line contrast does not test a TAD-disorganization mechanism;')
    if experiment == 'N4_subclone_stratification':
        text = text.replace('CNV-defined subclones', 'expression-derived chromosome-arm score clusters')
    if experiment == 'E5_hparam_sweep':
        start = text.index('        # Precompute adjacency lists')
        end = text.index('        graph_secs =', start)
        text = text[:start] + text[end:]
        start = text.index('            kappa = {}')
        end = text.index('            per_cell =', start)
        text = text[:start] + '            kappa = ollivier_ricci_edges(G, alpha=alpha, selected_edges=sample_edges)\n\n' + text[end:]
    if experiment == 'T6_random_hvg_control':
        start = text.index('    neigh =', text.index('def run_pipeline'))
        end = text.index('    per_cell =', start)
        text = text[:start] + '    kappa = ollivier_ricci_edges(G, alpha=ALPHA, selected_edges=sample_edges)\n\n' + text[end:]
    if experiment == "N5b_prostate":
        text = text.replace('if patient is None:\n', 'if patient is None or "org" in f.lower():\n')
    if experiment == "E6_pseudotime_paul":
        text = text.replace("adata = sc.datasets.paul15()", "adata = sc.datasets.paul15()\n    adata.X = adata.X.astype(np.float64)")
        text = text.replace("sc.tl.pca(adata, n_comps=50, random_state=SEED)", "sc.tl.pca(adata, n_comps=50, random_state=SEED, svd_solver='full', dtype='float64')")
    if experiment == "CLASSIFIER_AUC":
        text = text.replace('if not m:\n            continue', 'if not m or "org" in f.lower():\n            continue')
        text = text.replace("X_counts = X_counts[:, keep]\n", "X_counts = X_counts[:, keep]\n    full_library = X_counts.sum(axis=0)\n")
        text = text.replace("libsize = X_counts.sum(axis=0)\n    libsize[libsize == 0] = 1\n    Xn", "libsize = full_library\n    Xn")
        text = text.replace("X_lin = np.clip(X_lin, 0, None)", "\n        if not np.isfinite(X_lin).all() or (X_lin < 0).any(): raise ValueError('invalid assay expression')")
        text = text.replace("pct_mito = np.zeros(X.shape[1], dtype=np.float64)", "pct_mito = np.full(X.shape[1], np.nan, dtype=np.float64)")
        text = text.replace('res_pm    = aggregate_runs(X_pm,    y, seeds)', "res_pm = aggregate_runs(X_pm, y, seeds) if np.isfinite(X_pm).all() else {k:float('nan') for k in ('auc_roc_mean','auc_roc_std','auc_pr_mean','auc_pr_std','accuracy_mean','f1_mean')}")
        text = text.replace('            cd["pct_mito"],\n','').replace('            held["pct_mito"],\n','').replace('            np.concatenate([c["pct_mito"] for c in train_idx]),\n','')
        text = text.replace('print(f"  !! LOADER FAILED: {e}")\n            continue',
                            'raise RuntimeError(f"required cohort {name} failed: {e}") from e')
    # Redirect import-time log opens as well as main-time output files.
    text = text.replace('HERE / "run.log"', repr(str(out / "run.log")))
    if experiment == "T2_wu_clonality":
        text = text.replace('HERE / f"{full_name}.', '(ROOT / "data" / "T2_wu_clonality") / f"{full_name}.')
    tree = ReferenceChoices().visit(ast.parse(text))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
                 and isinstance(n.test, ast.Compare)
                 and isinstance(n.test.left, ast.Name)
                 and n.test.left.id == "__name__")]
    ast.fix_missing_locations(tree)
    compiled_source_sha = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    mod = types.ModuleType("tc_extension_" + experiment)
    mod.__file__ = str(source)
    exec(compile(tree, str(source), "exec"), mod.__dict__)
    mod.corrected_source_sha256 = compiled_source_sha
    mod.HERE = out
    mod.OUT = out
    # Source constants are resolved before HERE/OUT changes, so data stay
    # at the original input locations, never beneath corrected/.
    for key, value in list(mod.__dict__.items()):
        if isinstance(value, Path) and value.suffix == ".gz" and value.parent == source.parent:
            mod.__dict__[key] = ROOT / "data" / experiment / value.name
    if experiment == "F2_icb_stratified":
        from yost_response_crosswalk import extract
        mod.YOST_RESPONSE = extract()
        mod.COUNTS = ROOT / "data/E7_clonality/GSE123813_bcc_scRNA_counts.txt.gz"
        mod.META = ROOT / "data/E7_clonality/GSE123813_bcc_all_metadata.txt.gz"
        mod.N1_PER_PATIENT = ROOT / "experiments/N1_sadefeldman_icb/corrected/n1_sadefeldman_per_patient.csv"
    if hasattr(mod, "fetch_chr_matrix"):
        original_fetch = mod.fetch_chr_matrix
        def locked_contact_slice(url, chrom, res):
            import urllib.request
            stem = hashlib.sha256(json.dumps({"url":url,"chrom":chrom,"resolution":res,"normalization":"KR","matrix":"observed"},sort_keys=True).encode()).hexdigest()
            folder = ROOT / "data/hic_contact_slices"
            folder.mkdir(parents=True,exist_ok=True)
            array_file = folder / f"{stem}.npz"
            receipt_file = folder / f"{stem}.json"
            if array_file.exists() and receipt_file.exists():
                receipt = json.loads(receipt_file.read_text())
                if hashlib.sha256(array_file.read_bytes()).hexdigest() != receipt["sha256"]:
                    raise ValueError("Hi-C contact-slice content mismatch")
                with np.load(array_file,allow_pickle=False) as z:
                    return z["matrix"],int(z["n_bins"]),int(z["chromosome_bp"])
            matrix,n_bins,nbp = original_fetch(url,chrom,res)
            if not np.isfinite(matrix).all() or np.any(matrix < 0): raise ValueError("invalid consumed contact matrix")
            with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=60) as response:
                identity = {k:response.headers.get(k) for k in ("ETag","Last-Modified","Content-Length")}
            temporary = array_file.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f,matrix=matrix,n_bins=np.asarray(n_bins),chromosome_bp=np.asarray(nbp))
            temporary.replace(array_file)
            atomic_json(receipt_file,{"url":url,"chromosome":chrom,"resolution":res,"normalization":"KR","matrix":"observed","source_asset_headers":identity,"hicstraw_version":version("hic-straw"),"sha256":hashlib.sha256(array_file.read_bytes()).hexdigest(),"bytes":array_file.stat().st_size,"kind":"consumed remote normalized contact-matrix slice; not a checksum of the entire .hic asset"})
            return matrix,n_bins,nbp
        mod.fetch_chr_matrix = locked_contact_slice
    records = []
    from lib.cache import ResultCache, digest_of_params

    def graph(X, k=15):
        G = numerics.knn_graph(X, k=k)
        if experiment == "T1_umi_confound":
            from sklearn.neighbors import NearestNeighbors
            distances = NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(X)[0]
            return G, distances
        return G

    def ricci(G, alpha=.5, n_edges=None, rng=None, **kwargs):
        selected_edges = kwargs.pop('selected_edges', None)
        if "max_edges" in kwargs:
            if n_edges is not None: raise ValueError("conflicting edge budgets")
            n_edges = kwargs.pop("max_edges")
        if kwargs: raise ValueError(f"unrecognized scientific transport options: {sorted(kwargs)}")
        before = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
        graph_payload = {"nodes": list(G.nodes()), "edges": [(int(u), int(v), float(d)) for u,v,d in G.edges(data="weight")], "rng": before}
        digest = hashlib.sha256(json.dumps(graph_payload, sort_keys=True).encode()).hexdigest()
        import ot.lp.emd_wrap
        pot_binary = hashlib.sha256(Path(ot.lp.emd_wrap.__file__).read_bytes()).hexdigest()
        binding = {"selected_edges": selected_edges, "pot_binary_sha256": pot_binary, "alpha": alpha, "n_edges": n_edges, "numerics_sha256": numerics.IMPORTED_SOURCE_SHA256, "numpy": version("numpy"), "scipy": version("scipy"), "POT": version("POT")}
        cache = ResultCache(ROOT/"cache/extensions", experiment)
        pdg = digest_of_params(binding)
        key = cache.key(digest, pdg)
        hit, reason = cache.load(key, digest, pdg)
        if hit is not None and set(hit) != {"edge_ids", "kappa", "rng_after"}:
            raise ValueError("unexpected extension cache schema")
        if hit is not None:
            measured = {tuple(map(int,e)): types.SimpleNamespace(kappa=float(k)) for e,k in zip(hit["edge_ids"],hit["kappa"])}
            if rng is not None: rng.bit_generator.state = json.loads(str(hit["rng_after"][0]))
        else:
            measured = numerics.ricci_edges(G, alpha=alpha, n_edges=n_edges, rng=rng, selected_edges=selected_edges)
            after = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
            cache.save(key, {"edge_ids": np.asarray(list(measured), dtype=np.int64), "kappa": np.asarray([c.kappa for c in measured.values()]), "rng_after": np.asarray([json.dumps(after, sort_keys=True)])}, facts={"inputs_digest":digest,"params_digest":pdg})
        cells = numerics.cell_curvature(G, measured)
        records.append({"graph_nodes": len(G), "graph_edges": G.number_of_edges(),
                        "alpha": alpha, "measured_edge_ids": sorted([list(e) for e in measured]),
                        "cells": {str(n): vars(c) | {"kappa": c.kappa if c.status == "measured" else None}
                                  for n, c in cells.items()}})
        return {e: c.kappa for e, c in measured.items()}

    def cell_means(G, edges):
        cells = numerics.cell_curvature(G, {e: types.SimpleNamespace(kappa=c) for e, c in edges.items()})
        return {n: c.kappa for n, c in cells.items()}

    if hasattr(mod, "build_knn_graph"):
        mod.build_knn_graph = graph
    if hasattr(mod, "build_trans_graph"):
        mod.build_trans_graph = lambda adata, k=15: numerics.knn_graph(adata.obsm["X_pca"], k=k)
    if hasattr(mod, "build_phys_graph") and experiment != 'T4_visium_spatial':
        columns = ("center_x", "center_y") if experiment == "F3_merfish_multi" else ("array_row", "array_col")
        physical_k = 6 if experiment == 'N2_visium_hd' else 10
        def physical(adata, k=physical_k):
            G = numerics.knn_graph(np.column_stack([adata.obs[c].values for c in columns]), k=k)
            if experiment == "F3_merfish_multi":
                return G, float(np.median([d for _, _, d in G.edges(data="weight")]))
            return G
        mod.build_phys_graph = physical
    mod.ollivier_ricci_edges = ricci
    if experiment == "T3_bakry_emery":
        mod.ollivier_ricci_per_cell = lambda G, alpha=.5, n_edges=None, rng=None: cell_means(
            G, ricci(G, alpha=alpha, n_edges=n_edges, rng=rng))
    if experiment == 'N4_subclone_stratification':
        def per_cell_array(G, n_edges, alpha, rng):
            cells = cell_means(G, ricci(G, alpha=alpha, n_edges=n_edges, rng=rng))
            return np.asarray([cells[n] for n in range(len(G))], dtype=np.float64)
        mod.ollivier_ricci_per_cell = per_cell_array
    for name in ("per_cell_mean_curvature", "per_node_mean_curvature", "per_node_mean_curv"):
        if hasattr(mod, name):
            setattr(mod, name, cell_means)
    if hasattr(mod, "cliffs_delta"):
        mod.cliffs_delta = lambda a, b: numerics.cliff_placements(a, b).delta
    return mod, records


def run(experiment):
    out = ROOT / "experiments" / experiment / "corrected"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "run_manifest.json"
    atomic_json(artifact, {"status": "running", "method_version": METHOD_VERSION})
    records = []
    try:
        mod, records = prepare(experiment, out)
        mod.main()
        if not records:
            raise RuntimeError("no corrected transport computations were performed")
        for output in out.glob('*.json'):
            if output != artifact:
                load_json_strict(output)
        metric_status = {}
        if experiment == 'CLASSIFIER_AUC':
            import pandas as pd
            within = pd.read_csv(out / 'per_cohort_auc.csv')
            loco = pd.read_csv(out / 'leave_one_out_auc.csv')
            expected = {name for name, loader in mod.COHORT_LOADERS}
            for table in (within, loco):
                cohort_column = 'cohort' if 'cohort' in table else 'held_out_cohort'
                if set(table[cohort_column]) != expected or len(table) != len(expected):
                    raise ValueError('classifier cohort coverage is incomplete')
                for column in table.select_dtypes(include='number'):
                    missing = ~np.isfinite(table[column].to_numpy())
                    if missing.any():
                        if column != 'pct_mito_auc_roc_mean':
                            raise ValueError(f'nonfinite measured classifier metric: {column}')
                        for name in table.loc[missing, cohort_column]:
                            metric_status[f'{name}/{column}'] = 'unavailable: mitochondrial genes absent from source assay; no zero measurement imputed'
        atomic_json(out / "geometry.json", records)
        files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(out.rglob("*")) if p.is_file() and p != artifact and p.suffix != ".log"}
        payload = {"status": "exploratory_complete", "method_version": METHOD_VERSION,
                   "experiment": experiment, "files": files,
                   "n_corrected_geometries": len(records),
                   "inference": "historical exploratory design; dependent cell-level tests are descriptive, not donor validation",
                   "implementation": "float64/full-SVD, identity-safe kNN, full-graph verified POT transport, measured-incidence means",
                   "historical_source_sha256": hashlib.sha256((ROOT / "experiments" / experiment / "run.py").read_bytes()).hexdigest(),
                   "loaded_numerics_sha256": numerics.IMPORTED_SOURCE_SHA256,
                   "loaded_adapter_sha256": IMPORTED_ADAPTER_SHA256,
                   "corrected_source_sha256": mod.corrected_source_sha256}
        if experiment == "CLASSIFIER_AUC":
            payload["benchmark"] = "balanced cohort-transductive cell-split benchmark; split SD is not a donor confidence interval; assay-expression sums are not universally UMI depth"
            payload['metric_status'] = metric_status
            payload['multifeature_predictors'] = ['curvature', 'assay expression sum', 'detected genes']
        atomic_json(artifact, payload)
        return 0
    except Exception as e:
        atomic_json(artifact, {"status": "failed", "method_version": METHOD_VERSION,
                              "experiment": experiment, "exception_type": type(e).__name__,
                              "reason": str(e), "n_corrected_geometries": len(records)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    args = parser.parse_args()
    sys.exit(run(args.experiment))
