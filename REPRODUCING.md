# bio_paths — Reproducibility / Run Order

Single-page guide for reproducing the patient-level Ollivier-Ricci curvature
analysis (scRNA-seq, malignant vs non-malignant). Companion documents:
`RESEARCH_NOTE.md` (methods + results), `PREREGISTRATION.md` (locked
held-out protocol), `DATA_PROVENANCE.md` (data + label provenance).

## 1. Environment

Python 3 (developed and run on CPython 3.14.7). Exact pinned versions in
`requirements.txt`. Third-party packages actually imported across the
pipeline (verified by grepping imports in `exp/META_patient_level/`,
`exp/HELDOUT_GSE131907/`, `exp/HELDOUT_GSE161529/`, `exp/FIGGEN_fig4.py`,
`exp/FIGGEN_fig5.py`):

| Package | Import as | Used for |
|---|---|---|
| numpy | `numpy` | arrays, RNG |
| scipy | `scipy.sparse`, `scipy.stats` | sparse matrices, normal/chi2/t tests |
| pandas | `pandas` | matrices + metadata loading |
| rdata | `rdata` | parsing figshare Seurat `.rds` objects |
| POT | `ot` | `ot.emd2` optimal transport for Ollivier-Ricci |
| matplotlib | `matplotlib.pyplot` | forest plots, figures |
| scikit-learn | `sklearn.decomposition`, `sklearn.neighbors` | PCA, kNN |
| networkx | `networkx` | kNN graph container |

Everything else is the standard library (`csv`, `gzip`, `json`, `os`,
`pathlib`, `re`, `subprocess`, `sys`, `time`, `warnings`, `gc`).

Setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. Data layout

All datasets live under `data/` — one directory (or file pair) per cohort,
named by GEO accession: `data/GSE161529/` (Pal breast atlas),
`data/GSE131907_kim_nsclc/`, `data/GSE72056_melanoma.txt`,
`data/GSE84465_GBM.csv`, `data/GSE103322_HNSCC.txt`,
`data/GSE81861_CRC_tumor_FPKM.csv`, `data/GSE111672_PDAC-*`,
`data/GSE176031_chen/`, `data/GSE147082/`, plus non-GEO dirs (depmap, paul15,
visium, merfish, ...). Data is **read-only**: no script writes into `data/`
except `exp/HELDOUT_GSE161529/download.py` and `extract_labels.py`, which
populate `data/GSE161529/{samples,labels,figshare_tmp}/` on first run.

The full dataset table (tumour type, accession, cell counts, format) is
**Methods 2.1 of `RESEARCH_NOTE.md`** — see there; it is not duplicated
here. Per-cohort provenance for the two held-out sets: `DATA_PROVENANCE.md`.

## 3. Execution order

Scripts are run from the repo root (`bio_paths/`) as `python3 exp/<name>/run.py`
unless noted. Each experiment directory is self-contained and writes its own
outputs in place.

1. **Discovery experiments** — `exp/` E-series (`E1_within_patient` ...
   `E7_clonality`), T-series (`T1`...`T6` controls), N-series (`N1`...`N5c`
   extension), F-series (`F1`...`F4` confirmatory). These established the
   pipeline conventions (OR primitives live in `exp/E1_within_patient/run.py`,
   copied verbatim downstream) and feed the manuscript Sections 2.4–2.19.
2. **Patient-level meta-analysis** — `exp/META_patient_level/run.py`.
   Pools the six cohorts with recoverable patient IDs.
3. **Held-out GSE131907 (secondary)** — `exp/HELDOUT_GSE131907/run.py`.
   Reads `data/GSE131907_kim_nsclc/` directly (no download step).
4. **Held-out GSE161529 (primary)** — in order:
   1. `exp/HELDOUT_GSE161529/download.py` — pulls the GEO sample manifest +
      matrices and the six figshare Seurat objects into `data/GSE161529/`.
   2. `exp/HELDOUT_GSE161529/extract_labels.py` — builds
      `data/GSE161529/labels/cells_primary.tsv.gz` and `cells_secondary.tsv.gz`.
   3. `exp/HELDOUT_GSE161529/run.py` — pre-registered two-arm validation.
5. **Sensitivity (post-hoc, not pre-registered)** —
   `exp/HELDOUT_GSE161529/run_sensitivity.py` — dual-label handling variants.
6. **Figures** — `exp/FIGGEN_fig4.py` (classifier AUC panel; reads
   `exp/CLASSIFIER_AUC/`), `exp/FIGGEN_fig5.py` (label-validation +
   batch-correction panel; reads `exp/LABEL_VALIDATION/`,
   `exp/BATCH_CORRECTION/`).

## 4. Seed

Every stochastic stage uses the fixed seed **20260507** (`SEED = 20260507`
in `META_patient_level/run.py`, both held-out `run.py`s, and
`run_sensitivity.py`), matching the pre-registered parameters. Reruns are
deterministic given the same package versions and input data.

## 5. Where results live

| Analysis | Outputs |
|---|---|
| Discovery experiments | inside each `exp/<experiment>/` (e.g. `exp/E1_within_patient/e1_summary.png`, `e1_per_patient.csv`) |
| Patient-level meta | `exp/META_patient_level/`: `results.json`, `per_patient_delta.csv`, `forest_plot_data.csv`, `forest_plot.png` |
| GSE131907 held-out | `exp/HELDOUT_GSE131907/`: `results.json`, `per_patient_delta.csv` |
| GSE161529 held-out | `exp/HELDOUT_GSE161529/`: `results.json`, `per_patient_delta_primary.csv`, `per_patient_delta_secondary.csv` |
| GSE161529 sensitivity | `exp/HELDOUT_GSE161529/results_sensitivity.json` |
| Manuscript figures | `exp/FIGGEN_fig4.png`, `exp/FIGGEN_fig5.png` |

Every run also appends a `run.log` in its experiment directory. Manually
executed runs are additionally recorded in
`_docs/computation_log.md` (workspace root).
