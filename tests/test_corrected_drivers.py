"""End-to-end corrected-DRIVER tests on synthetic GEO-shaped fixtures.

These prove the corrected rerun wiring is real (strict readers, identity
gates, corrected numerics, DL pooling, method-bound cache, blocked and
self-reproduction gates) — not merely blocked stubs — without any
GEO/figshare input.
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
sys.path.insert(0, str(EXPERIMENTS))


def load_driver(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lung_fixture(tmp_path_factory):
    """Synthetic GSE131907-shaped inputs: UMI TSV + annotation."""
    rng = np.random.default_rng(20260507)
    d = tmp_path_factory.mktemp("kim")
    n_genes = 80
    barcodes = []
    meta = []
    cell = 0
    for s in range(6):
        for grp, n in (("mal", 25), ("imm", 25)):
            for _ in range(n):
                bc = f"cell{cell:05d}"
                base = rng.normal(0.4, 0.3, n_genes) * (grp == "mal")
                meta.append((bc, f"S{s}", grp))
                barcodes.append((bc, rng.poisson(np.abs(base) + .2)
                                 .astype(int)))
                cell += 1
    umi = d / "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
    with gzip.open(umi, "wt") as f:
        f.write("gene\t" + "\t".join(b for b, _ in barcodes) + "\n")
        for gi in range(n_genes):
            f.write(f"GENE{gi}\t" +
                    "\t".join(str(c[1][gi]) for c in barcodes) + "\n")
    ann_rows = []
    for bc, sample, grp in meta:
        ann_rows.append({
            "barcode": bc, "Sample": sample,
            "Cell_type.refined": "Epithelial cells" if grp == "mal"
            else "T/NK cells",
            "Cell_subtype": "tS1" if grp == "mal" else "-"})
    ann = pd.DataFrame(ann_rows).set_index("barcode")
    ann_path = d / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"
    ann.to_csv(ann_path, sep="\t", compression={"method": "gzip"})
    return umi, ann_path


@pytest.fixture(scope="module")
def lung_driver():
    return load_driver("experiments/HELDOUT_GSE131907/run_corrected.py",
                       "lung_corrected")


def test_lung_driver_blocked_artifact(lung_driver, lung_fixture, tmp_path,
                                      monkeypatch, capsys):
    monkeypatch.setattr(lung_driver, "KIM_MTX", tmp_path / "missing.gz")
    monkeypatch.setattr(lung_driver, "KIM_ANN", lung_fixture[1])
    monkeypatch.setattr(lung_driver, "REQUIRED_INPUTS", [("umi_matrix", tmp_path / "missing.gz", "synthetic GSE131907 fixture")])
    out = tmp_path / "results_corrected.json"
    with pytest.raises(SystemExit) as ei:
        lung_driver.run_or_block("t", lung_driver.REQUIRED_INPUTS, out,
                                 lambda: {}, argv=[])
    assert ei.value.code == 2
    payload = json.load(open(out))
    assert payload["status"] == "blocked"
    assert payload["missing_inputs"][0]["role"] == "umi_matrix"
    assert "GSE131907" in payload["missing_inputs"][0]["provenance"]


def test_lung_driver_end_to_end_and_self_reproduction(lung_driver,
                                                      lung_fixture,
                                                      tmp_path,
                                                      monkeypatch):
    umi, ann = lung_fixture
    monkeypatch.setattr(lung_driver, "KIM_MTX", umi)
    monkeypatch.setattr(lung_driver, "KIM_ANN", ann)
    monkeypatch.setattr(lung_driver, "DONOR_MAP_PATH", tmp_path / "absent_map.tsv")
    monkeypatch.setattr(lung_driver, "HERE", tmp_path)
    cache = tmp_path / "cache"
    payload1 = lung_driver.run(cache_root=cache)
    assert payload1["method_version"] == "TC-1"
    assert payload1["metric"] == "ollivier-ricci-full-graph-v1"
    assert payload1["patient_unit_verified"] is False  # C08: Sample proxy
    assert "specimen" in payload1["unit_of_inference"]
    assert payload1["n_patients_passing_filter"] >= 5
    assert np.isfinite(payload1["dl_meta"]["delta"])

    # warm cache rerun: identical FULL payload (self-reproduction gate)
    payload2 = lung_driver.run(cache_root=cache)
    from lib.verification import compare_science

    assert compare_science(payload2, payload1, mode="exact") == []

    # run_or_block full mode: artifact written; second invocation with
    # the same inputs must pass the complete-field gate
    out = tmp_path / "results_corrected.json"
    reqs = [("umi", umi, "synthetic"), ("ann", ann, "synthetic")]
    argv = []
    p1 = lung_driver.run_or_block("lung", reqs, out,
                                  lambda: lung_driver.run(cache_root=cache),
                                  argv=argv)
    assert p1["status"] == "ok"
    p2 = lung_driver.run_or_block("lung", reqs, out,
                                  lambda: lung_driver.run(cache_root=cache),
                                  argv=argv)
    assert p2["status"] == "ok"

    # tampered rerun must FAIL the complete-field gate
    def tampered():
        p = lung_driver.run(cache_root=cache)
        p["dl_meta"]["se"] = p["dl_meta"]["se"] * 1.5
        return p

    from lib.verification import VerificationError

    with pytest.raises(VerificationError, match="rerun differs"):
        lung_driver.run_or_block("lung", reqs, out, tampered, argv=argv)
    # ...and --update supersedes it with a recorded change count
    p3 = lung_driver.run_or_block("lung", reqs, out, tampered,
                                  argv=["--update"])
    assert p3["supersedes"]["n_changed_fields"] >= 1
    assert p3["supersedes"]["previous_created_utc"]


# ---------------------------------------------------------------------------
# breast driver: strict per-sample MTX + identity gates on a 2-sample
# synthetic GSE161529-shaped layout
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def breast_driver(tmp_path_factory):
    driver = load_driver("experiments/HELDOUT_GSE161529/run_corrected.py",
                         "breast_corrected")
    driver.SOURCE_EXCLUSIONS = tmp_path_factory.mktemp('source-exclusions') / 'absent.json'
    return driver


@pytest.fixture(scope="module")
def breast_layout(tmp_path_factory):
    d = tmp_path_factory.mktemp("gse161529")
    labels_dir = d / "labels"
    samples_dir = d / "samples"
    labels_dir.mkdir()
    samples_dir.mkdir()
    rng = np.random.default_rng(5)

    def write_mtx(path, cols_nzz):
        """cols_nzz: {col: [(row, val), ...]}"""
        entries = [(r, c, v) for c, evs in cols_nzz.items()
                   for r, v in evs]
        n_rows = max(r for r, _, _ in entries) + 1
        with gzip.open(path, "wt") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write(f"{n_rows} {max(c for _, c, _ in entries) + 1} "
                    f"{len(entries)}\n")
            for r, c, v in sorted(entries):
                f.write(f"{r + 1} {c + 1} {v}\n")

    rows = []
    mapping = {}
    n_genes = 30
    for sample, stem, gsm in (("ER_0001_T", "GSM1_ER-MH0001-T", "GSM1"),
                              ("TN_0002", "GSM2_TN-MH0002", "GSM2")):
        cols_nzz = {}
        barcodes = []
        for grp in ("malignant", "immune"):
            for i in range(15):
                local = f"AAAC{i:03d}-{1 if grp == 'malignant' else 2}"
                full = f"{sample}_{local}"
                col = len(barcodes)
                base = rng.normal(0.5, 0.3, n_genes) * \
                    (grp == "malignant")
                ent = [(g, int(max(0, round(b))))
                       for g, b in enumerate(base) if b > 0.1]
                if not ent:
                    ent = [(0, 1)]
                cols_nzz[col] = ent
                barcodes.append(local)
                rows.append({"barcode": full, "sample": sample,
                             "patient": sample.split("_")[1],
                             "label": grp})
        with gzip.open(samples_dir / f"{stem}-barcodes.tsv.gz",
                       "wt") as f:
            f.write("\n".join(barcodes) + "\n")
        write_mtx(samples_dir / f"{stem}-matrix.mtx.gz", cols_nzz)
        mapping[sample] = {"gsm": gsm, "stem": stem,
                           "matrix_url": f"https://example.org/{stem}",
                           "barcodes_url": f"https://example.org/{stem}"}
    pd.DataFrame(rows).to_csv(labels_dir / "cells_primary.tsv.gz",
                              sep="\t", index=False,
                              compression={"method": "gzip"})
    # secondary labels: epithelial_not_called_malignant comparator
    sec = pd.DataFrame([dict(r, label="malignant" if r["label"] ==
                             "malignant" else "normal_epithelial")
                        for r in rows if r["sample"] == "TN_0002"])
    sec.to_csv(labels_dir / "cells_secondary.tsv.gz", sep="\t",
               index=False, compression={"method": "gzip"})
    with gzip.open(samples_dir / "GSE161529_features.tsv.gz", "wt") as f:
        for g in range(n_genes):
            f.write(f"ENSG{g:04d}\tGENE{g}\tGene Expression\n")
    with open(samples_dir / "MANIFEST.json", "w") as f:
        json.dump({"mapping": mapping,
                   "manifest": {"status": "ok", "files": [],
                                "errors": []}}, f)
    return d, labels_dir, samples_dir, mapping


def test_breast_driver_variant_resolution_and_strict_load(breast_driver,
                                                          breast_layout,
                                                          monkeypatch):
    d, labels_dir, samples_dir, mapping = breast_layout
    monkeypatch.setattr(breast_driver, "LAB_DIR", labels_dir)
    monkeypatch.setattr(breast_driver, "SAMPLES_DIR", samples_dir)
    monkeypatch.setattr(breast_driver, "FEATURES",
                        samples_dir / "GSE161529_features.tsv.gz")
    lab, identity = breast_driver.resolve_labels("primary", "excluded")
    assert identity["n_dual_labeled_barcodes"] == 0
    assert set(lab["label"]) == {"malignant", "immune"}
    counts, genes, aligned, info, per_sample = \
        breast_driver.load_counts_strict(lab, mapping)
    assert counts.shape[0] == 60
    assert counts.shape[1] == len(genes) == 30
    assert set(per_sample) == set(mapping)
    # strict load counts nnz actually read == declared
    for rec in per_sample.values():
        assert rec["nnz_read"] == rec["nnz_declared"]


def test_breast_driver_rejects_unmapped_label_sample(breast_driver,
                                                     breast_layout,
                                                     monkeypatch):
    d, labels_dir, samples_dir, mapping = breast_layout
    monkeypatch.setattr(breast_driver, "LAB_DIR", labels_dir)
    monkeypatch.setattr(breast_driver, "SAMPLES_DIR", samples_dir)
    lab, _ = breast_driver.resolve_labels("primary", "excluded")
    lab.loc[len(lab)] = {"barcode": "XX_0009_AAA-1", "sample": "XX_0009",
                         "patient": "0009", "label": "immune"}
    from lib.bio_io import DataValidationError

    with pytest.raises(DataValidationError, match="absent from the "
                                                  "acquisition mapping"):
        breast_driver.load_counts_strict(lab, mapping)
