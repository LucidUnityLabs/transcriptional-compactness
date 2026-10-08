# Transcriptional Compactness

Patient-level meta-analysis of Ollivier-Ricci curvature on single-cell
RNA-seq kNN graphs: malignant cells versus cell-type-matched comparator
cells, across public tumour cohorts, with pre-registered held-out
validation.

Per-cell mean Ollivier-Ricci curvature (kappa) is computed on the kNN
graph of PCA-50 embeddings of public scRNA-seq cohorts, and tested as a
"transcriptional-compactness phenotype" of malignant cells. The study
combines a discovery patient-level random-effects meta-analysis, a
pre-registered two-cohort held-out validation (lung, breast), a
dual-label sensitivity analysis, and robustness/confound controls
(hyperparameter grid, UMI depth, proliferation, random gene panels,
alternative annotations, batch correction).

## Status

Preprint manuscript, not peer reviewed. No DOI yet.

The original transport routine used shortest paths restricted to endpoint
neighborhoods. Its numerical results and original manuscript are preserved
as a historical baseline, and must not be cited as validated TC-1 results.
Corrected rerun status is recorded in [experiments/rerun_status.json](experiments/rerun_status.json).
The source-backed lung specimen-to-donor map resolves 58 specimens to 44
donors; the historical sample-level inference does not establish donor-level
replication. Breast cells lacking a malignant call do not establish a normal
luminal comparator. Corrected claims will be reported only from validated
outputs, with those endpoint limitations explicit.

## Historical results — superseded by the TC-1 metric correction

- **Discovery meta-analysis:** pooled patient-level Cliff's delta
  = +0.51 [95% CI +0.44, +0.59], p = 7.5e-38 (35 patients, 6 cohorts).
- **Held-out lung (GSE131907, metastatic LUAD):** replicates — pooled
  delta = +0.247, one-sided p = 1.7e-5.
- **Held-out breast (GSE161529), immune comparator:** annotation-unstable
  — delta ranges -0.17 to +0.30 across dual-label handling variants.
- **Held-out breast (GSE161529), epithelial comparator:** reverses —
  pooled delta = -0.41 (1/11 patients positive).

These historical conclusions await corrected metric, identity, and donor
reruns. The original narrative remains in `manuscript/MANUSCRIPT.md` for
comparison; its retained figures and numbers are not corrected evidence.

## Repository layout

| Path | Contents |
|---|---|
| `manuscript/MANUSCRIPT.md` | Full preprint manuscript |
| `PREREGISTRATION.md` | Locked pre-registered held-out protocol |
| `experiments/` | Per-analysis code (`run.py`) and outputs (`results.json`, CSVs, summary texts, figure PNGs), including both held-out cohorts and the sensitivity analysis |
| `figures/` | Main-text figure PNGs (see `figures/FIGURES.md`) |
| `DATA_PROVENANCE.md` | Data sources and label-derivation chains for the held-out cohorts |
| `REPRODUCING.md` | Environment, data layout, execution order, seed |
| `requirements.txt` | Pinned package versions |

Note: internal documents (manuscript, REPRODUCING.md, DATA_PROVENANCE.md,
FIGURES.md) refer to the experiments directory as `exp/`; in this repo it
is `experiments/`.

## Reproducibility

- `python3 -m pip install -r requirements.txt` (CPython 3.14, pinned
  versions; full experiment profile and reviewed-lock workflow:
  `REPRODUCING.md` §1)
- Predominant seed **20260507**; three committed 20260508 exceptions and
  the per-stage seed table: `REPRODUCING.md` §4. Reruns are deterministic
  given the same package versions and input data
- Run order, data layout, and output locations: `REPRODUCING.md`

## Data availability

Raw data are **not included** in this repository. All cohorts are public
GEO deposits; accessions are listed in the manuscript (Methods 2.1) and
`DATA_PROVENANCE.md`. Held-out cohort download and label-extraction
scripts are included (`experiments/HELDOUT_GSE161529/download.py`,
`extract_labels.py`). Committed per-patient entries in `results.json`
files are aggregated statistics computed from these public cohorts, with
de-identified unit IDs.

## License

MIT — see `LICENSE`.
