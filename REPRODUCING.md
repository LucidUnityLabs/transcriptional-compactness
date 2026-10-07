# bio_paths — Reproducibility / Run Order

Single-page guide for reproducing the patient-level Ollivier-Ricci curvature
analysis (scRNA-seq, malignant vs non-malignant). Companion documents:
`PREREGISTRATION.md` (locked held-out protocol), `DATA_PROVENANCE.md`
(data + label provenance), `README.md` (repository map).

## 1. Environment

Python 3 (developed and run on CPython 3.14.7). `requirements.txt` pins
the eight CORE distributions by direct inspection of the interpreter used
for all runs (`python3 -c "import <pkg>; print(<pkg>.__version__)`"):

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

**These eight pins are not the full import surface.** The repository-wide
import inventory (`python tools/import_inventory.py`, audit R06) also finds
direct third-party imports of `statsmodels` (E1/E1b/N1), `harmonypy`
(BATCH_CORRECTION), `scanpy` (E6/F3/N2/T4), `anndata`, `h5py` and
`hicstraw` (F3/N2/T4, F4/N3) — none of which are pinned in
`requirements.txt`. The earlier "everything else is the standard library"
claim was therefore false. The remaining imports (`csv`, `gzip`, `json`,
`os`, `pathlib`, `re`, `subprocess`, `sys`, `time`, `warnings`, `gc`) are
standard library.

Environment profiles (audit R06):

- `requirements/core.in` — the eight core pins plus build/test tooling,
  as an initial **candidate** input.
- `requirements/controls.in` — the full experiment profile (core + the
  six additional direct imports above), also a **candidate**.

These `.in` files are not locks. To produce a reviewed, hash-checked
environment, resolve ONCE on the target interpreter/platform:

```bash
python tools/lock_environment.py \
  --requirements requirements/core.in \
  --wheelhouse wheelhouse/core-cp314-linux-x86_64 \
  --lock requirements/core-cp314-linux-x86_64.lock
```

Review the generated lock + manifest, install/test offline in a fresh
target venv (`--require-hashes --no-index --find-links <wheelhouse>`),
then commit them. Release verification must consume reviewed locks and
never resolve a new environment. The eight-version `requirements.txt`
pins (no transitive pins, no hashes) are retained for the historical
record and quick local setup:

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
except `experiments/HELDOUT_GSE161529/download.py` and `extract_labels.py`,
which populate `data/GSE161529/{samples,labels,figshare_tmp}/` on first run.

The full dataset table (tumour type, accession, cell counts, format) is in
the manuscript (Methods 2.1, `manuscript/MANUSCRIPT.md`). Per-cohort
provenance for the two held-out sets: `DATA_PROVENANCE.md`.

## 3. Execution order

Scripts are run from the repository root as
`python3 experiments/<name>/run.py` unless noted. Each experiment
directory is self-contained and writes its own outputs in place.

1. **Discovery experiments** — `experiments/` E-series (`E1_within_patient`
   ... `E7_clonality`), T-series (`T1`...`T6` controls), N-series
   (`N1`...`N5c` extension), F-series (`F1`...`F4` confirmatory). These
   established the pipeline conventions (OR primitives live in
   `experiments/E1_within_patient/run.py`, copied verbatim downstream) and
   feed the manuscript Sections 2.4–2.19.
2. **Patient-level meta-analysis** — `experiments/META_patient_level/run.py`.
   Pools the six cohorts with recoverable patient IDs.
3. **Held-out GSE131907 (secondary)** —
   `experiments/HELDOUT_GSE131907/run.py`. Reads `data/GSE131907_kim_nsclc/`
   directly (no download step).
4. **Held-out GSE161529 (primary)** — in order:
   1. `experiments/HELDOUT_GSE161529/download.py` — pulls the GEO sample
      manifest + matrices and the six figshare Seurat objects into
      `data/GSE161529/`.
   2. `experiments/HELDOUT_GSE161529/extract_labels.py` — builds
      `data/GSE161529/labels/cells_primary.tsv.gz` and
      `cells_secondary.tsv.gz`.
   3. `experiments/HELDOUT_GSE161529/run.py` — pre-registered two-arm
      validation.
5. **Sensitivity (post-hoc, not pre-registered)** —
   `experiments/HELDOUT_GSE161529/run_sensitivity.py` — dual-label
   handling variants.
6. **Figures** — `experiments/FIGGEN_fig4.py` (classifier AUC panel; reads
   `experiments/CLASSIFIER_AUC/`), `experiments/FIGGEN_fig5.py`
   (label-validation + batch-correction panel; reads
   `experiments/LABEL_VALIDATION/`, `experiments/BATCH_CORRECTION/`).

## 3b. Corrected method (TC-1) and audit reruns

The 2026-10-06 correctness audit introduced a corrected numerical method
as a SEPARATELY IDENTIFIED path. The committed per-experiment `run.py`
scripts and their outputs are the preserved historical analysis; the
corrected method lives in the shared library `experiments/lib/`
(`numerics`, `bio_io`, `preprocess`, `verification`, `cache`,
`acquisition`, `cohort`, `runner`) and is executed by:

| Corrected driver | Artifact |
|---|---|
| `experiments/META_patient_level/run_corrected.py` | `results_corrected.json` |
| `experiments/HELDOUT_GSE131907/run_corrected.py` | `results_corrected.json` |
| `experiments/HELDOUT_GSE161529/run_corrected.py` | `results_corrected.json` |
| `experiments/HELDOUT_GSE161529/download_corrected.py` | `results_acquisition_corrected.json` |
| `experiments/LABEL_VALIDATION/run_corrected.py` | `results_corrected.json` |

Corrections (method version `TC-1`): full-graph support-to-support
transport metric; identity-safe kNN construction (coincident coordinates
rejected as non-estimable); verified optimal transport that fails
closed; coverage-aware per-cell summaries; exact Cliff placement
variances with explicit non-estimable status (no variance floor);
validating DerSimonian-Laird/bootstrap helpers; strict UMI and Matrix
Market readers; label/axis identity gates; binary-connectivity Louvain;
truly frozen-geometry label contrasts; content-addressed caches that can
never serve a different computation. Corrected pooled numbers are
EXPECTED to differ from the historical ones (the metric changed);
manuscript numbers must be regenerated from corrected artifacts only.

Corrected drivers exit with code 2 and write a `blocked` artifact naming
every missing input when the GEO/figshare data are absent — results are
never fabricated. `experiments/rerun_status.py` enumerates every
affected analysis and its data requirements into
`experiments/rerun_status.json`.

Tests: `python3 -m pytest tests/` (unit, reference, corruption, cache
and synthetic-pipeline layers; the POT backend test skips where POT is
not installed and is MANDATORY in CI, which installs `requirements.txt`
and sets `REQUIRE_POT=1` so a missing POT install fails the run rather
than skipping; `python -O -m pytest tests/` runs the same suite with
production asserts disabled).

### 3c. Deviations and corrections ledger (audit R08)

The dated `PREREGISTRATION.md` is preserved verbatim as protocol history.
This ledger records, per the audit, which analysis choices were specified
BEFORE inspecting each held-out endpoint and which are post hoc or
mathematical corrections:

| Choice | Status |
|---|---|
| Comparator definition (cell-type-matched non-malignant) | pre-specified (PREREGISTRATION) |
| Dual-label conflict resolution variants | **post hoc** (sensitivity analysis, not pre-registered) |
| Patient proxy mapping (specimen → donor, e.g. GSE131907 Sample used as patient) | **post hoc / proxy** — "patient-level" is not established by grouping a column (audit C08); specimen→donor map owner-gated |
| Normalization | log1p-CPM(1e6), gene filter ≥5 cells — pre-specified; **denominator order corrected** in TC-1 (full raw library before filtering, R04); historical runs filtered first |
| Sample caps | 60 cells/patient/group, total 3,500 — pre-specified; TC-1 `stratified_sample_minima_first` preserves group minima under the cap (R05) |
| Actual feature dimensions (HVG count, PCA components actually used) | recorded per run in corrected artifacts; historical docs stated targets, not actuals |
| Edge sampling | 4,000-edge sampled OR curvature — pre-specified; measured-edge IDs retained under TC-1 (C05 coverage) |
| Graph metric | **corrected post hoc** (mathematical bug): legacy restricted-support metric → TC-1 full-graph transport distance; the historical committed numbers embed the legacy metric and are preserved byte-identical, NOT regenerated |
| Seeds | see §4 — 20260507 pre-specified; three committed 20260508 exceptions documented above |
| Statistical reporting | DL preregistered interval kept; modified Hartung–Knapp reported as a sensitivity only (M04); bounded-interval estimand work owner-gated |

Mathematical bug fixes are not hidden as if they generated the historical
committed numbers: the historical artifacts stay frozen under their own
method, and corrected numbers must come only from TC-1 reruns with the
named data present (§3b, `experiments/rerun_status.json`).

## 4. Seeds (as committed — audit R04 note)

The predominant stochastic seed is **20260507** (`SEED = 20260507` in
`META_patient_level/run.py`, both held-out `run.py`s, and
`run_sensitivity.py`), matching the pre-registered parameters. Three
committed exceptions exist and are documented here rather than silently
harmonized (changing them would break exact legacy reproduction):

| Location | Seed | Scope |
|---|---|---|
| `META_patient_level/run.py` | `SEED + 1` = **20260508** | patient-clustered bootstrap (B=1000) |
| `LABEL_VALIDATION/run.py` | **20260508** | label-validation decomposition (comment: "matches N5b / N5c") |
| `N2_visium_hd/run.py` | **20260508** | Visium HD extension |

The earlier claim of a single universal seed was therefore not exact.
Reruns are deterministic given the same package versions and input data;
the corrected TC-1 drivers additionally use named per-stage seed streams
(`experiments/lib/cohort.py: rng_stream`).

## 5. Where results live

| Analysis | Outputs |
|---|---|
| Discovery experiments | inside each `experiments/<experiment>/` (e.g. `experiments/E1_within_patient/e1_summary.png`, `e1_per_patient.csv`) |
| Patient-level meta | `experiments/META_patient_level/`: `results.json`, `per_patient_delta.csv`, `forest_plot_data.csv`, `forest_plot.png` (+ TC-1 `results_corrected.json`) |
| GSE131907 held-out | `experiments/HELDOUT_GSE131907/`: `results.json`, `per_patient_delta.csv` (+ `results_corrected.json`) |
| GSE161529 held-out | `experiments/HELDOUT_GSE161529/`: `results.json`, `per_patient_delta_primary.csv`, `per_patient_delta_secondary.csv` (+ `results_corrected.json`, `results_acquisition_corrected.json`) |
| GSE161529 sensitivity | `experiments/HELDOUT_GSE161529/results_sensitivity.json` |
| Manuscript figures | `experiments/FIGGEN_fig4.png`, `experiments/FIGGEN_fig5.png` |

**Logging (corrected from earlier text):** the scripts print progress to
stdout/stderr and do NOT create `run.log` files, and there is no
`_docs/computation_log.md` in this repository. To retain a run log, capture
it in the shell with failure propagation:

```bash
set -euo pipefail
python3 experiments/META_patient_level/run.py 2>&1 | tee META_run.log
```

Without `set -euo pipefail` (or `pipefail` alone) a failed Python process
can hide behind a successful `tee`; do not describe console capture as an
automatic artifact.
