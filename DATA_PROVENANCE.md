# Data Provenance — Held-Out Cohorts

One page. Sources and label-derivation chains only; numbers live in the
result files cited. Companion: `README.md` (run order), `RESEARCH_NOTE.md`
Methods 2.1 (full dataset table).

## GSE161529 (Pal et al. 2021 breast cancer atlas)

- **GEO accession:** GSE161529. Counts + per-sample metadata pulled from the
  GEO FTP supplementary files by `exp/HELDOUT_GSE161529/download.py`, which
  parses `data/GSE161529/GSE161529_family.soft.gz` for the GSM ->
  supplementary-file mapping (27-sample manifest) and writes per-sample
  matrices to `data/GSE161529/samples/`. Download log:
  `data/GSE161529/DOWNLOAD_LOG.txt`.
- **Cell cluster identities:** Chen et al. 2022 (Sci Data 9:96) companion
  deposit, figshare DOI **10.6084/m9.figshare.17058077** — six Seurat
  objects, downloaded and parsed by `exp/HELDOUT_GSE161529/extract_labels.py`:
  - `SeuratObject_ERTotal.rds`
  - `SeuratObject_HER2.rds`
  - `SeuratObject_TNBC.rds`
  - `SeuratObject_ERTotalSub.rds`
  - `SeuratObject_HER2Sub.rds`
  - `SeuratObject_TNBCSub.rds`

  Used fields: cell barcode, `group` (sample), `seurat_clusters`.
- **Companion repo:** github.com/yunshun/HumanBreast10X (same Chen et al.
  2022 deposit) — source of the malignant calls and marker panels below.

### Label derivation chain (`extract_labels.py`)

1. **Malignant calls:** `Tables/InferCNV-Annotation.txt` from the companion
   repo — inferCNV tumor column-blocks per (cluster, sample) on the
   Total-object clusters.
2. **+1 cluster shift:** annotation cluster `k` == `seurat_clusters k+1`.
   Validated in `extract_labels.py` (header): annotation tumor sets
   {0,4,5,6} / {0,3,7} / {0,2} equal the repo code's epithelial split sets
   {1,5,6,7} / {1,4,8} / {1,3} shifted by +1, for ER / HER2 / TNBC
   respectively (3/3 groups).
3. **Immune identity:** authors' marker panel `Signatures/ImmuneMarkers2.txt`
   applied to the authors' Sub-object (microenvironment) clusters, anchored
   to author-pinned assignments — T cells = ERTotalSub {1,8}, TNBCSub {1,5},
   HER2Sub {2,7} (companion code `ER.R`/`TNBC.R`/`HER2.R` + EV4 legend);
   ERTotalSub cluster 7 = cycling TAM; HER2Sub cluster 9 myeloid/luminal.
4. **Outputs:** `data/GSE161529/labels/cells_primary.tsv.gz`
   (malignant vs immune), `cells_secondary.tsv.gz`
   (malignant vs normal_epithelial), plus
   `cluster_marker_scores.tsv` / `cluster_sizes*.tsv`.
5. **Scope decision (recorded in `extract_labels.py`):** TumLN and Male
   analyses excluded — original-author tumour/immune calls exist only as
   figure annotations, not machine-readable labels.

### Dual-label conflict

26,439 barcodes carry **both** labels: malignant under the Total-object
inferCNV clusters and immune under the Sub-object clusters. Both labels
originate in the deposited annotation (figshare + companion repo), not in
pipeline code — see the independent code-audit entry in
`_docs/computation_log.md` (2026-09-06). Handling variants (baseline
exclusion / dual-as-malignant / dual-as-immune) are quantified in
`exp/HELDOUT_GSE161529/results_sensitivity.json`.

## GSE131907 (Kim et al. 2020 metastatic LUAD atlas)

- **Accession:** GSE131907. Raw UMI matrix + annotation in
  `data/GSE131907_kim_nsclc/` (no download script; files placed directly).
- **Annotation file:** `GSE131907_Lung_Cancer_cell_annotation.txt.gz`
  (original-author labels; malignant = `Cell_type.refined == Epithelial cells`
  with malignant subtypes, comparator = immune `Cell_type.refined` classes).
- **Sample-as-patient limitation:** `Sample` is the annotation's only
  patient-of-origin proxy (58 samples / 44 patients per the paper), so the
  run treats each sample as one unit. Recorded in
  `exp/HELDOUT_GSE131907/results.json` (`labels.patient_unit`).
