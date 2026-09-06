# Pre-Registration: Held-Out Cohort Validation

**Date:** 2026-06-29
**Study:** Per-cell Ollivier-Ricci curvature as a transcriptional-compactness phenotype of solid tumours
**Pre-registrator:** Tyler Lewis, Independent

---

## Hypothesis

Malignant cells have higher per-cell mean Ollivier-Ricci curvature (κ)
than non-malignant cells on the kNN graph of scRNA-seq PCA-50 embeddings.

## Prediction (directional, one-sided)

Cliff's δ (malignant vs non-malignant) > 0 in each held-out cohort,
with patient-level meta-analysis p < 0.05 (one-sided).

## Pre-registered cohorts

### Primary: GSE161529 (Pal et al. 2021 breast cancer atlas)
- 421,761 cells, 52 patients
- Annotation: Chen et al. 2022 (Sci Data) — original-author malignant/non-malignant labels
- Comparator: immune cells (primary), normal luminal epithelial (secondary)
- Prediction: δ > 0 for both comparators

### Secondary: GSE131907 (Kim et al. 2020 metastatic LUAD atlas)
- 208,506 cells, 44 patients
- Annotation: original-author cell_annotation.txt
- Comparator: immune cells
- Prediction: δ > 0

## Analysis plan (locked before data inspection)

1. Load each cohort. Compute PCA-50 on log-normalized expression (HVG-2000, if gene
   counts allow; otherwise use all genes).
2. Build kNN graph (k=15, Euclidean on PCA-50).
3. Compute per-cell mean Ollivier-Ricci κ via optimal transport (POT ot.emd2, same
   parameters as E1: spread parameter α=0.5, edge sampling ~4000 edges per cohort).
4. For each patient with ≥10 malignant and ≥10 non-malignant cells:
   compute patient-level Cliff's δ.
5. Random-effects meta-analysis (DerSimonian-Laird) across patients.
6. Report: pooled δ, 95% CI, I², p-value (one-sided test of δ > 0).
7. Decision rule: SUPPORTED if pooled δ > 0 and p < 0.05 (one-sided);
   NOT SUPPORTED otherwise.

## Parameters locked

- PCA dimensions: 50
- HVG count: 2000 (or all genes if < 2000)
- kNN k: 15
- OR spread α: 0.5
- Edge sampling: 4000 edges or all edges if < 4000
- Minimum cells per patient per group: 10
- Minimum patients per cohort: 5
- Meta-analysis method: DerSimonian-Laird random-effects
- Significance: one-sided p < 0.05

## What would falsify

- Pooled δ ≤ 0 in the primary cohort (GSE161529)
- Pooled δ ≤ 0 in BOTH cohorts
- Patient-level δ > 0 in < 50% of patients in the primary cohort

## What would NOT falsify (but would weaken)

- δ > 0 but p > 0.05 in one cohort (underpowered)
- δ > 0 in primary but δ ≤ 0 in secondary (tumor-type-specific)
- I² > 75% (heterogeneous but consistent direction)

---

This pre-registration is filed before inspection of GSE161529 or GSE131907 data.
The prediction is based on results from 7 independent cohorts (5/7 replicating
at patient level: Tirosh melanoma, Puram HNSCC, Olalekan ovarian, Darmanis GBM,
Peng PDAC underpowered). The directional hypothesis (malignant κ > non-malignant κ)
is the same across all tested cohorts.
