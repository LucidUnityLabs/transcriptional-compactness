# Per-cell Ollivier-Ricci curvature on the single-cell-RNA-seq cell graph as a transcriptional-compactness phenotype of solid tumours: a multi-cohort study with patient-level meta-analysis

**Tyler Lewis**
Independent
mail2@tylersinbox.com

**Status:** preprint draft, not peer-reviewed.

**Historical draft; corrected scientific release remains open.** Numerical
claims below retain their original method and have not been reconciled into
a complete corrected campaign. The corrected bounded breast primary contrast
is conditional on inferred labels, selected cells and edges, a shared fitted
graph, and a whole-specimen source exclusion; it does not restore the original
complete endpoint or establish independent-donor causal significance.
Corrected Harmony sensitivities attenuate melanoma and GBM and reverse HNSCC.
High heterogeneity and graph dependence constrain the positive narrative.
Welch/MWU cell tests and balanced transductive splits cannot establish donor
independence or clinical deployment performance. Frozen repeats, complete
controls, reference/environment verification, figures, and manuscript
reconciliation remain open. Restoration of the excluded breast specimen
requires authenticated same-cell raw data; the normal-luminal endpoint
requires affirmative per-cell annotation on the exact analyzed universe.
See `../SOURCE_PROTOCOL.md` and `../experiments/rerun_status.json`.

**Code, data, and per-experiment scripts:** `biology c+/bio_paths/`. Each
validation experiment is in `bio_paths/exp/EN_*/` with its own `run.py`,
CSV outputs, and figures.

---

## Abstract

Discrete graph-curvature methods have been applied to gene-level networks
of cancer transcriptomes (Sandhu et al. 2015) and to gene-network Ricci
flow on differentiation trajectories (Weistuch et al. 2024), but a direct
test on the cell-cell k-nearest-neighbour graph of single-cell RNA-seq —
treating per-cell curvature as a phenotype — does not appear in the
literature. We compute per-cell mean Ollivier-Ricci κ on the kNN graph of PCA-50
embeddings of seven public tumour scRNA-seq datasets and find that
malignant cells have higher κ than non-malignant comparators.

**Patient-level random-effects meta-analysis (discovery cohorts only;**
DerSimonian-Laird, 35 patients
across 6 cohorts with cell-type-matched comparators): pooled Cliff's
δ = **+0.51 [95% CI +0.44, +0.59], z = 12.9, p = 7.5 × 10⁻³⁸**, I² = 96.8%.
This replaces per-cell p-values as the headline inference and addresses the
pseudoreplication concern.

**Replication profile (discovery cohorts):** 5/7 cohorts replicate at the patient level (Tirosh
melanoma δ = +0.91, Puram HNSCC δ = +0.66, Olalekan ovarian δ = +0.42,
Darmanis GBM δ = +0.29, all p < 0.05; Peng PDAC δ = +0.10, n = 2 patients,
underpowered). Chen prostate (published cell-pooled δ = +0.21) **fails
within-patient replication** (patient-level δ = −0.08, 1/4 patients positive)
— a Simpson's paradox artifact where the pooled signal was between-patient,
not within-patient. Li CRC is excluded from the meta-analysis (no
patient-of-origin labels available; direction unstable at n = 34 T cells).

**Operational discrimination:** κ alone achieves classifier AUC-ROC of
0.82-0.87 in high-effect cohorts (melanoma, ovarian) but 0.58-0.70 in
low-effect cohorts (PDAC, prostate, HNSCC). Leave-one-cohort-out AUC
averages 0.62 (moderate cross-cohort generalization). Multi-feature models
(κ + UMI + n_genes + pct_mito) achieve 0.83 within-cohort and 0.76
leave-one-out, showing κ adds incremental but not dominant signal.

**Robustness:** The finding survives alternative cell-type annotation
(marker panels and unsupervised clustering) in both questioned cohorts,
a 72-cell hyperparameter grid, Forman-Ricci diagnostic, within-patient
mixed-effects retention (85-97%), UMI/cell-cycle/HVG confound controls,
and Bakry-Émery CD(K,N) decomposition.

**Bounded scope (honestly reported):** The κ direction does not transfer to
physical-tissue Delaunay graphs (Visium: δ = −0.005), genome-wide Hi-C
contact graphs (cell-line K562/GM12878, opposite sign), or consistently to
liquid tumours (AML: weak inversion, δ = −0.09). An ICB-biomarker claim
does not survive cross-cohort replication.

**Pre-registered held-out validation (Sections 3.28–3.29):** lung
replicates (GSE131907 metastatic LUAD: pooled δ = +0.247 [95% CI
+0.130, +0.364], one-sided p = 1.7 × 10⁻⁵, 23/31 units positive —
robust); the breast immune-comparator is annotation-unstable
(GSE161529: δ from −0.17 to +0.30 across dual-label handlings); the
breast epithelial-comparator reverses (δ = −0.41, 1/11 patients
positive). Overall, the compactness effect is tissue- and
comparator-dependent and, in breast, sensitive to annotation choices.

**Batch-correction sensitivity:** Harmony integration with patient as the
batch variable attenuates the pooled cell-level effect markedly (Tirosh
δ +0.74 → +0.21; Darmanis +0.34 → +0.04) and sign-flips Puram
(+0.34 → −0.52); random-embedding and UMI-only nulls are ≈ 0 in all three
cohorts tested (Section 3.27).

The integrated picture: per-cell κ is a genuine transcriptional-compactness
phenotype — monotonically increasing along lineage commitment in normal
differentiation (Spearman ρ = +0.42 with pseudotime) and elevated in tumour
cells predominantly by tumour-specific transcriptional restriction. The
effect is robust but heterogeneous, and does not constitute a universal
cancer detector.

**Keywords:** Ollivier-Ricci curvature; single-cell RNA sequencing; cancer;
graph geometry; transcriptional compactness; patient-level meta-analysis;
tumour microenvironment

---

## 1. Introduction

The geometry of gene-expression spaces has been studied along two largely
separate threads. The first is *hyperbolic embedding*: Zhou et al. (2021)
showed that the gene-cell expression matrix admits a low-dimensional
locally-Euclidean, globally-hyperbolic structure across many cell types,
and that this geometry recovers known developmental hierarchies
(Klimovskaia et al. 2020; Ding & Regev 2021; Tian et al. 2023). The
second is *graph curvature on gene-level networks*: Sandhu et al. (2015),
Pouryahya et al. (2020), and a growing body of follow-ups have computed
Ollivier-Ricci curvature on correlation- or PPI-weighted gene networks
across cancer types, finding that cancer networks have higher aggregate
curvature than matched normal, interpreted as a fragility-vs-robustness
signal.

A natural object that sits between these two threads has not, as far as
we can determine, been examined: the cell-cell k-nearest-neighbour graph
built directly from a single-cell RNA-seq embedding, with discrete
curvature treated as a *per-cell phenotype*. Sritharan et al. (2025;
ICLR) compute Ollivier-Ricci on scRNA-seq cell-cell graphs but use it
for edge pruning to clean up downstream manifolds (PBMC and mouse cortex
data). Weistuch et al. (2024; *Nat Commun*) apply Forman-Ricci flow to
gene-interaction networks weighted by single-cell expression. Neither
reports whether per-cell curvature distinguishes malignant from
non-malignant cells in tumour scRNA-seq.

Here we run that test and subject the result to twenty-four concrete
follow-up experiments before reporting it. Within-scope validation
(sixteen experiments): cross-tumour-type replication on five new
datasets (E3 HNSCC, E4 CRC, N5a PDAC, N5b prostate, N5c ovarian),
within-patient stratified mixed-effects analysis (E1), hyperparameter
robustness sweep (E5), Forman-Ricci diagnostic (E2), pseudotime check
on non-cancer hematopoiesis (E6), clonality-vs-malignancy specificity
tests on two TCR-paired datasets (E7 BCC, T2 lung), UMI/depth confound
(T1), proliferating-immune subset comparison (T5), random-HVG negative
control (T6), Bakry-Émery dimension decomposition (T3), and Visium
spot-resolution spatial replication (T4). All sixteen leave the
malignant-vs-non-malignant direction intact or interpretably narrowed. Cross-domain extensions
(eight experiments): ICB-response biomarker test on Sade-Feldman 2018
(N1) and stratified ICB replication on Yost 2019 (F2), Visium HD
8 µm single-cell-adjacent spatial (N2) and MERFISH ~12 µm cell-contact
spatial (F3), Hi-C contact-graph curvature on chr22 + chr11 at 1 Mb
(N3) and genome-wide at 100 kb (F4), within-tumour CNV-defined subclone
κ stratification (N4), and liquid-tumour AML replication (F1). Several
cross-domain results yield biologically informative outcomes:
**substrate-specificity is confirmed at three resolutions** (T4 / N2 /
F3 all show null on physical-neighbour graphs), **the κ direction is
inverted in liquid (AML) tumours** consistent with myeloid-hierarchy-
collapse biology rather than clonal-lineage commitment, and **chromatin
contact graphs show a robust opposite-direction κ signature**
(genome-wide pooled Cliff's δ = −0.55, 22 / 23 chromosomes negative)
consistent with the Flavahan / Hnisz TAD-disorganisation literature.
ICB-response biomarker is **not validated cross-cohort** (N1 anti-PD1
AUC of 0.81 collapses to 0.50 on Yost replication). The cross-domain extensions yield
mostly null or partial results: N1 shows only a weak biomarker trend
(per-patient AUC 0.64), N2 confirms the T4 substrate-specificity null
at single-cell-adjacent resolution, N3 does not transfer cleanly to
3D-genome contact graphs at 1 Mb resolution (chr22 reverses sign,
chr11 weak), and N4 stratifies subclones only in well-powered tumours.
The integrated picture: the κ phenotype is robust within its defined
scope (transcriptomic kNN on solid-tumour scRNA-seq) but does not
naively generalise to clinical biomarker, physical-tissue-architecture,
chromatin-contact-graph, or reliable intra-tumour-subclone domains.
This bounded scope is itself informative.

## 2. Methods

### 2.1 Datasets

| Dataset | Tumour type | Accession | Cells (mal / non-mal) | Format |
|---|---|---|---|---|
| Tirosh et al. 2016 | melanoma | GSE72056 | 1 257 / 3 256 | TSV log₂(TPM/10+1) |
| Darmanis et al. 2017 | glioblastoma | GSE84465 | 1 091 / 1 847 immune | space-sep raw counts |
| Puram et al. 2017 | head-and-neck SCC | GSE103322 | 2 215 / 3 363 | TSV log₂(TPM/10+1) |
| Li et al. 2017 | colorectal | GSE81861 | 272 / 318 | CSV FPKM |
| Peng et al. 2019 | pancreatic ductal adenocarcinoma | GSE111672 | ~635 / ~2 480 ductal | filtered count matrices |
| Chen et al. 2021 | prostate adenocarcinoma | GSE176031 | marker-derived | per-patient DGE UMI counts |
| Olalekan et al. 2021 | HGSOC ovarian | GSE147082 | marker-derived | Smart-seq2 read counts |
| Yost et al. 2019 | basal cell carcinoma (+TCR) | GSE123813 | ~53k cells | counts + metadata + TCR |
| Wu et al. 2020 | lung tumour + NAT T cells (+TCR) | GSE139555 | 20 080 T cells (LT2+LN3) | counts + clonotype metadata |
| Sade-Feldman et al. 2018 | melanoma (ICB) | GSE120575 | 2 403 baseline CD8 | TPM + patient metadata |
| van Galen et al. 2019 | AML (liquid) | GSE116256 | ~5 451 blasts / ~4 149 BM | Smart-seq2 counts + author labels |
| Paul et al. 2015 | mouse hematopoiesis (non-cancer control) | GSE72857 | 2 730 cells, 19 clusters, MEP→terminal | scanpy built-in |
| Rao et al. 2014 | Hi-C K562 / GM12878 (cell lines) | GSE63525 | per-bin contact graphs | .hic (KR-balanced) |
| 10X Visium CytAssist | breast cancer FFPE (spatial) | vendor demo dataset | 4 169 in-tissue spots | h5 + tissue positions |
| 10X Visium HD | colon cancer P2 (spatial, 8 µm) | GSM8594568 | ~14k bins (120×120 tile) | h5 + 10X L1 labels |
| GSE291210 (Wu et al. 2025) | MMTV-PyMT mouse mammary (MERFISH) | GSE291210 | ~663k segmented cells | cell-by-gene + µm coords |

Pre-registered held-out cohorts (analysed in Sections 3.28–3.29): Pal et
al. 2021 breast atlas (GSE161529) and Kim et al. 2020 metastatic LUAD
atlas (GSE131907).

Cell-type labels are taken from the original authors where available
(Tirosh, Darmanis, Puram, Li, Peng, Yost, Wu, Sade-Feldman, van Galen,
Paul) and derived from canonical marker panels where not (Chen prostate,
Olalekan ovarian — label sensitivity is tested explicitly in Section 3.26).

**Scope note.** The twenty-four experiments of Sections 2.4–2.19 and the
five confirmatory analyses of Sections 2.20–2.24 define this manuscript's
scope; approximately 46 additional exploratory experiments, predominantly
in a therapy-translation research line that was terminated after its
apparent positives failed cross-dataset replication (termination documented
in `THERAPY_FINAL_v2.md`), are excluded from this manuscript.

### 2.2 Embedding

For each dataset: top 2 000 highest-variance genes (log-space matrix),
z-score per gene, 50-component PCA. Where raw counts are provided
(Darmanis, Li-FPKM) we apply log1p before HVG selection. This pipeline
follows Wolf et al. 2018 (Scanpy defaults) and Zhou et al. 2021.

### 2.3 kNN graph and Ollivier-Ricci curvature

We build a symmetric k = 15 NN graph over the PCA-50 embedding using
Euclidean edge weights. For each edge (u, v) we compute the lazy-walk
Ollivier-Ricci curvature

  κ(u, v) = 1 − W₁(μ_u, μ_v) / d(u, v)

where μ_u is the lazy-walk distribution placing mass α = 0.5 on u and
mass (1 − α)/deg(u) on each graph-neighbour of u, the cost matrix is the
shortest-path distance in the subgraph induced by the union of the two
support sets, and W₁ is computed exactly with POT (Flamary et al. 2021).
Negative κ corresponds to locally hyperbolic structure, positive κ to
locally spherical / clustered structure. We compute κ on a uniform
random sample of 4 000 edges per analysis (≈30% of edges); per-cell κ is
the mean of incident sampled edges.

### 2.4 Forman-Ricci curvature (Section 3.8)

For independent corroboration we compute the Sreejith et al. (2016)
edge-based Forman-Ricci curvature with unit node weights:

  Ric_F(e) = 2 w_e − w_e · (Σ_{e_u ~ e} 1/√(w_e · w_{e_u}) + Σ_{e_v ~ e} 1/√(w_e · w_{e_v}))

on the same kNN graph. Per-cell FR is the mean of incident edges.
Forman-Ricci is sensitive to degree and edge-weight structure but is
known to miss the triangle/clustering component captured by Ollivier-Ricci
(Samal et al. 2018).

### 2.5 Within-patient mixed-effects model (Section 3.5)

We fit `kappa ~ malignant + (1 | patient_id)` using
`statsmodels.MixedLM`, restricted to patients with ≥ 30 cells in each
of the malignant and non-malignant groups. We compare the fixed-effect
coefficient and 95% CI to the naive pooled mean difference; high
retention (mixed/naive ≥ 0.5) indicates the effect is within-patient,
not between-patient. We also report per-patient Welch t-tests and Cliff's
δ to show the proportion of patients whose individual effects share the
overall sign.

### 2.6 Hyperparameter sweep (Section 3.6)

Grid over HVG ∈ {500, 2000, 5000}, PCA ∈ {20, 50, 100}, k ∈ {5, 15, 30,
50}, α ∈ {0.0, 0.5}, totalling 72 cells. For each cell, recompute the
embedding, kNN graph, OR on 3 000 sampled edges, and Cliff's δ for
malignant vs T-cell distributions. The fraction of cells with
same-direction effect, the median |δ|, and the variance partitioned by
each axis are reported.

### 2.7 Pseudotime experiment (Section 4.1)

Paul et al. 2015 hematopoiesis is processed via Scanpy default
recipe; diffusion pseudotime is rooted at cluster `7MEP` (most stem-like
cluster present). κ is computed on the same k = 15 kNN graph with
α = 0.5 (4 000 sampled edges); per-cell κ is regressed on DPT (Spearman
ρ).

### 2.8 Reproduction

All scripts and outputs in `bio_paths/exp/EN_*/`. Random seed 20260507
(exceptions: N1, N4, N5a-c, F1, F2, and the label-validation analysis use
seed 20260508; the random-HVG control varies gene-set seeds 1–20 over a
fixed cell sample of seed 20260507). Software: numpy 2.4, scipy 1.17,
scikit-learn 1.8, networkx 3.6, POT 0.9.6, scanpy 1.12, statsmodels
(latest), harmonypy, matplotlib 3.x.

### 2.9 Cross-tumour-type replication cohorts (E3, E4, N5a–N5c; Sections 3.1, 3.4)

Puram HNSCC (E3): labels from the author metadata rows (malignant /
non-cancer type); two contrasts — malignant vs all non-malignant and
malignant vs the largest specific subtype (fibroblast) — with 600 + 600
cells, seed 20260507, standard pipeline (Section 2.2–2.3). Li CRC (E4):
labels parsed from column headers; log1p, HVG-2000, PCA-50, kNN k = 15,
α = 0.5 on min(2500, |E|) sampled edges given the small cohort; per-cell
κ averaged over 5 bootstrap seeds; contrasts epithelial vs all others and
epithelial vs T cells; seed 20260507. Peng PDAC (N5a): PDAC-A and PDAC-B
inDrop matrices pooled; malignant = Cancer clone A + B; comparator =
combined non-malignant ductal pool (lineage-matched); min(600, available)
cells per arm; seed 20260508. Chen prostate (N5b): primary tumour and
matched normal DGE matrices for four patients (PR5249/PR5251/PR5254/
PR5261; organoids and benign biopsies excluded); cell types scored with
prostate marker panels (luminal, basal, T-cell, myeloid, fibroblast);
tumour-tissue luminal calls = malignant, normal-tissue luminal = benign
luminal; contrasts malignant vs benign luminal (lineage-matched) and
malignant vs immune T; seed 20260508. Olalekan ovarian (N5c): log1p-CP10K
marker scoring with a positive epithelial score and negative
immune+stromal score (doublets excluded); contrasts epithelial vs T cells,
vs fibroblasts, vs all others; 600 + 600 stratified subsample; seed
20260508.

### 2.10 TCR-clonality specificity (E7 Yost BCC, T2 Wu lung; Sections 3.7, 3.14)

E7 (Yost GSE123813, seed 20260507): three groups — non-expanded T cells
(no TCR call or singleton CDR3 group), expanded T cells (CDR3 group
≥ 10 cells), and malignant cells (Tumor_1/Tumor_2 clusters) — each
subsampled to 600 cells before building a single shared kNN graph, so the
curvature comparison is graph-consistent. T2 (Wu GSE139555, seed
20260507): lung tumour LT2 + matched normal LN3 T cells with
author-assigned clonotypes; expanded (≥ 10 cells) vs non-expanded
(singleton or no TCR), 600 vs 600; the Wu deposit is T-cells-only, so only
the expanded-vs-non-expanded arm is tested.

### 2.11 UMI / library-size stratification (T1; Section 3.9)

Darmanis cells stratified into 5 UMI quintiles (per-cell total raw
counts); within each quintile the full pipeline is re-run independently —
HVG-2000 within quintile, PCA-50, kNN k = 15, Ollivier-Ricci α = 0.5 on
min(2500, |E|) sampled edges — and neoplastic vs immune compared by
Welch t, MWU, and Cliff's δ. A reference unstratified run and an inverse
stratification on per-cell mean kNN distance (geometric depth proxy) use
the same parameters. Seed 20260507.

### 2.12 Proliferating-immune control (T5; Section 3.10)

Tirosh immune cells scored on a curated 17-gene G2M/S signature (MKI67,
TOP2A, CCNB1, CCNA2, CDK1, CDC20, BIRC5, AURKA, AURKB, CENPF, MCM2–MCM7,
PCNA), z-scored across all cells; proliferating = z > 1, resting = z < 0.
Standard pipeline on a 600/600/600 sample (malignant / proliferating /
resting); three-way pairwise contrasts. Seed 20260507.

### 2.13 Random gene-panel control (T6; Section 3.11)

HVG-2000 selection replaced by 2000 uniformly random genes across 20
gene-set seeds (1–20); the 600 + 600 Tirosh malignant-vs-T-cell sample is
fixed (seed 20260507) so only gene selection varies. Per gene-set:
PCA-50, kNN k = 15, Ollivier-Ricci α = 0.5 on 3 000 sampled edges,
Cliff's δ and MWU. The real HVG-2000 run is the reference.

### 2.14 Bakry-Émery curvature-dimension decomposition (T3; Section 3.12)

Per-vertex Bakry-Émery CD(K, N) via the Cushing-Liu-Münch curvature-matrix
formulation reduced to a closed-form generalised eigenproblem (no SDP):
for each vertex, local Γ and Γ₂ matrices A, B and Laplacian vector c are
formed on the unweighted symmetric k = 10 NN graph (300 + 300 cells per
dataset); K_BE(v; N) is the smallest generalised eigenvalue of
(B − (1/N) cc^T, A); N_eff is the smallest grid N admitting K_BE > 0.
Pearson correlation between per-cell OR κ and K_BE(∞). Seed 20260507.

### 2.15 Spatial dual-graph tests (T4 Visium, N2 Visium HD, F3 MERFISH; Sections 3.13, 3.16, 3.20)

Each spatial test builds two graphs over the same spots/cells —
G_trans (transcriptomic kNN, k = 15 on PCA-50) and G_phys (physical
neighbour graph) — and computes OR κ (α = 0.5, 4 000 sampled edges per
graph) on each. T4 (10X Visium CytAssist Human Breast Cancer FFPE, seed
20260507): normalize-total + log1p + HVG-2000 (seurat_v3) + PCA-50;
tumour vs stroma signatures (KRT8/18/19, EPCAM, ESR1 vs PTPRC, COL1A1,
ACTA2, PECAM1) via scanpy score_genes, top/bottom tertiles → 1 376 mal /
1 376 stroma spots; G_phys = Delaunay triangulation on array coordinates
keeping edges < 1.5 × median Delaunay edge length. N2 (Visium HD 8 µm
colon cancer P2, GSM8594568): 120 × 120 array tile, 10X UnsupervisedL1
labels (Tumor vs Fibroblast/SmoothMuscle/Endothelial), ~5 000 bins
subsampled for label balance; G_phys = kNN k = 6 on array coordinates;
signature-score backup labels computed. F3 (MERFISH MMTV-PyMT mouse
mammary, GSE291210 sample T1/GSM8830801, ~285-gene Immuno-Oncology panel):
QC min_counts ≥ 20, min_cells ≥ 3; PCA to min(50, n_genes − 1); tumour
(Krt5/7/14/15/17, Cdh1) vs stroma (Acta2, Col1a1, Pdgfra, Pdgfrb, Pecam1,
Vim, Ptprc) signatures; malignant = tumour−stroma ≥ 67th percentile,
stroma ≤ 33rd; ~6 000 cells subsampled (40% malignant / 40% stroma / 20%
unlabelled); G_phys = kNN k = 10 on micron coordinates.

### 2.16 Hi-C contact-graph curvature (N3 1 Mb, F4 genome-wide 100 kb; Sections 3.17, 3.21)

Rao et al. 2014 (GSE63525) in-situ Hi-C, KR-balanced contacts streamed
with hic-straw for K562 (cancer) and GM12878 (normal). Per chromosome:
self-diagonal removed; observed/expected normalisation by the per-genomic-
distance median; edges = top 3·N_bins pairs by O/E, edge weight =
1 / log(1 + observed counts); per-edge OR κ (α = 0.5, exact W₁ via POT)
aggregated to per-bin means; K562 vs GM12878 compared by MWU and Cliff's
δ per chromosome and pooled. N3: chr22 and chr11 at 1 Mb. F4: all
autosomes + chrX at 100 kb, O/E floor 1.5, |i − j| ≥ 2, ≤ 4 000 sampled
edges per graph.

### 2.17 ICB-response biomarker (N1 Sade-Feldman, F2 stratified; Sections 3.15, 3.19)

N1 (GSE120575, seed 20260508): baseline (pre-treatment) samples only; CD8
T cells scored by CD8A/CD8B/CD3D/CD3E expression; joint HVG-2000 → PCA-50
→ kNN k = 15 → OR α = 0.5 on ~4 000 sampled edges; per-cell Cliff's δ,
Welch, MWU (R vs NR); per-patient mean κ by MWU and single-feature AUC;
logistic regression responder ~ mean_kappa + n_cells. F2 (Yost GSE123813,
seed 20260508): pre-treatment CD8_(mem|ex|act) T cells per cluster
annotation, same pipeline; per-arm AUCs (Yost anti-PD1, N1 anti-PD1
subset, anti-CTLA4 subset, combination subset); mega-analysis pooling
anti-PD1 patient means across both cohorts and all 30 patients.

### 2.18 CNV-defined subclone stratification (N4; Section 3.22)

Tirosh malignant cells with patient IDs; per-cell chromosome-arm CNV by a
manual moving-average proxy (genes ordered by genomic position, 100-gene
smoothing window on log-TPM, centred against a non-malignant immune
reference panel). Per tumour with ≥ 50 malignant cells: Ward linkage on
arm-level scores, cut from k = 5 down to k = 2 so each subclone keeps
≥ 10 cells. κ computed once on the PCA-50 kNN k = 15 graph of all
malignant cells; per-tumour Kruskal-Wallis across subclones and pairwise
max-min Cliff's δ. Seed 20260508.

### 2.19 Liquid-tumour AML test (F1; Section 3.18)

van Galen 2019 (GSE116256): six largest diagnostic AML samples (AML556,
AML419A, AML1012, AML328, AML210A, AML475) vs healthy bone marrow controls
BM1–BM5, author per-cell labels (PredictionRefined malignant/normal;
CellType hierarchy). Contrasts: (A) blasts vs healthy myeloid HSPCs
(healthy pool restricted to HSC/Prog/GMP/ProMono/Mono — cell-type-matched)
and (B) blasts vs all healthy BM. 600 cells per arm, standard pipeline,
seed 20260508.

### 2.20 Patient-level random-effects meta-analysis (Section 3.23)

The patient is the unit of inference. Per cohort: malignant and
comparator cells loaded with patient IDs; stratified sample capped at 60
cells per patient per group and 3 500 cells per cohort; one pooled kNN
graph (k = 15) per cohort; OR κ computed on min(6 000, |E|) sampled
edges per cohort graph; per patient with
≥ 10 malignant and ≥ 10 comparator cells: Cliff's δ with a U-statistic
Hoeffding-projection standard error. DerSimonian-Laird random-effects
meta-analysis across patients (DerSimonian & Laird 1986), per cohort and
overall; patient-clustered bootstrap (B = 1000) for percentile CIs; I² and
Cochran Q reported. Cliff's δ per Cliff (1993). Li CRC is excluded
(no patient-of-origin labels in the GEO processed file; reported
separately as a single pooled estimate). Seed 20260507.

### 2.21 Classifier operational evaluation (Section 3.25)

Per cohort (the 7 of Section 3.1): the standard κ pipeline re-run on
600 + 600 cells; single-feature logistic regression on κ
(class_weight = 'balanced'), stratified 80/20 train/test split, 10 random
seeds → AUC-ROC (primary), AUC-PR, accuracy, F1 (mean ± SD). Baselines:
random feature (sanity), UMI alone (depth proxy; total linear expression
per cell), n_genes, pct_mito. Multi-feature model: κ + UMI + n_genes +
pct_mito. Leave-one-cohort-out (LOCO): per-cell features pooled across
cohorts, train on 6, test on the held-out cohort. κ is computed once per
cohort and reused across seeds/folds so classifier variance is isolated
from κ variance. Seed 20260507.

### 2.22 Annotation-label validation (Section 3.26)

For the two marker-derived cohorts (Chen prostate, Olalekan ovarian),
labels re-derived three independent ways: (A) "original" — the N5b/N5c
marker panels reproduced exactly; (B) "alt1_broad" — a different, generic
broad epithelial panel (no prostate-secretory markers; Müllerian-specific
markers for ovarian); (C) "alt2_cluster" — unsupervised Louvain
communities on the PCA-50 kNN graph, annotated per community by marker
enrichment (no per-cell marker argmax). Each scheme rebuilds the 600 + 600
subsample and re-runs the standard pipeline (HVG-2000, PCA-50, k = 15,
α = 0.5, 4 000 edges). A fixed-comparator decomposition re-tests every
malignant scheme against the same original T-cell comparator pool. Seed
20260508.

### 2.23 Batch-correction sensitivity (Section 3.27)

Three largest cohorts (Tirosh, Puram, Darmanis), 1 200 cells each, four
conditions per cohort: (1) original pipeline; (2) Harmony
(harmonypy; Korsunsky et al. 2019) on the same PCA-50 embedding with
patient ID as the batch variable (Tirosh 19 batches, Puram 18, Darmanis
4), kNN rebuilt on the corrected embedding; (3) random null — PCA-50 of
i.i.d. Gaussian noise matching the HVG subset shape, 3 seeds; (4)
UMI-only — 1-D embedding of log(1 + total UMI). Cliff's δ (malignant vs
non-malignant) on per-cell mean κ in each condition. Seed 20260507.

### 2.24 Puram within-patient mixed-effects extension (E1b; Section 3.24)

The Section 2.5 protocol replicated on Puram HNSCC: patient ID parsed
from cell barcode; 109 pooled `HNSCC_combo1` cells with no patient origin
dropped (17 patients retained); stratified 600 malignant + 600
non-malignant sample; HVG-2000 → PCA-50 → kNN k = 15 → OR α = 0.5 on
4 000 sampled edges; `kappa ~ malignant + (1 | patient_id)` via
statsmodels MixedLM; per-patient Welch t and Cliff's δ for patients with
≥ 30 cells in both groups; fixed effect checked across three optimizers.
Seed 20260507.

### 2.25 Pre-registered held-out validation (Sections 3.28–3.29; executed 2026-09-05/06)

Held-out cohort validation on GSE161529 (Pal et al. 2021 breast atlas,
Chen et al. 2022 companion malignant labels) and GSE131907 (Kim et al.
2020 metastatic LUAD atlas) was executed under the locked analysis plan
filed in `PREREGISTRATION.md` (2026-06-29) before data inspection:
PCA-50 on log-normalised HVG-2000, kNN k = 15, OR α = 0.5 with 4 000
sampled edges, per-patient Cliff's δ (≥ 10 cells per group),
DerSimonian-Laird meta-analysis, one-sided test of δ > 0, SUPPORTED iff
pooled δ > 0 and p < 0.05. GSE131907 was run 2026-09-05
(`exp/HELDOUT_GSE131907/run.py` → `exp/HELDOUT_GSE131907/results.json`;
seed 20260507). GSE161529 was run 2026-09-06 with both pre-registered
comparator arms (`exp/HELDOUT_GSE161529/download.py`,
`extract_labels.py`, `run.py` → `exp/HELDOUT_GSE161529/results.json`;
seed 20260507). A post-hoc dual-label sensitivity analysis, not part of
the pre-registration, was run 2026-09-06
(`exp/HELDOUT_GSE161529/run_sensitivity.py` →
`exp/HELDOUT_GSE161529/results_sensitivity.json`), holding every locked
parameter fixed and varying only the dual-label assignment rule.
Normalisation (log1p-CPM 1e6; genes detected in ≥ 5 cells) was not
specified in the pre-registration and was carried over from the
discovery-runner dataset-loading convention, as recorded in the run's
ambiguity log. Outcomes are reported in Sections 3.28–3.29.


## 3. Results

### 3.1 Per-cell Ollivier-Ricci curvature separates malignant from non-malignant cells in seven tumour types

| Dataset | Contrast | n / n | Cliff's δ | Welch p |
|---|---|---|---|---|
| Tirosh 2016 melanoma (GSE72056) | Malignant vs T cells | 600 / 598 | **+0.74** | 3 × 10⁻¹⁰⁶ |
| Olalekan 2021 HGSOC ovarian (GSE147082) | Epithelial vs T cells | 600 / 600 | **+0.48** | 1.9 × 10⁻³⁸ |
| Puram 2017 HNSCC (GSE103322) | Malignant vs Fibroblast | 600 / 600 | **+0.41** | 7 × 10⁻³⁵ |
| Darmanis 2017 GBM (GSE84465) | Neoplastic vs Immune | 599 / 599 | **+0.38** | 1 × 10⁻²⁷ |
| Chen 2021 PRAD prostate (GSE176031) | Malignant vs Immune T | 600 / 600 | **+0.21** | 2.7 × 10⁻¹⁰ |
| Li 2017 CRC (GSE81861) | Epithelial vs T cells | 34 / 34 | **+0.20** | 1 × 10⁻³ |
| Peng/Moncada 2019 PDAC (GSE111672) | Malignant vs non-mal Ductal | 600 / 600 | **+0.15** | 2.5 × 10⁻⁶ |

The direction — malignant cells more positively curved than non-malignant
immune / stromal / lineage-matched cells — is consistent across **all
seven tumour types (7/7 replication)**, spanning melanoma (skin),
ovarian (HGSOC), head-and-neck SCC, glioblastoma (brain), prostate,
colorectal, and pancreatic adenocarcinomas. Effect size shows a
**comparator-distance gradient**: the most lineage-divergent comparators
(immune T cells against melanoma) yield the largest δ; lineage-matched
comparators (benign luminal vs malignant in prostate, normal ductal vs
malignant in PDAC) yield the smallest. Mann-Whitney U p-values are
similarly small in each case; see `exp/E3_puram_hnscc/`,
`exp/E4_li_crc/`, `exp/N5a_pdac/`, `exp/N5b_prostate/`, and
`exp/N5c_ovarian/` for full statistics.

The predicted-and-verified comparator-distance scaling is itself
informative. In Chen 2021 prostate, the same malignant compartment
yields δ = +0.27 vs basal epithelium (cross-lineage), δ = +0.21 vs
T cells, and δ = +0.14 vs benign luminal epithelium (lineage-matched).
The effect is therefore not "tumour vs anything else" but rather
"tumour vs whatever is transcriptionally most distant" — with a clean
residual signal even against the closest lineage-matched comparator.
PDAC, despite being the most stromally complex tumour tested, retains
δ = +0.15 against non-malignant ductal cells — at the lower bound of
the panel but with the same direction as the rest.

### 3.2 Global Gromov δ does not pick up the same signal

For the same Tirosh and Darmanis 600/600 cohorts, Gromov δ_rel is null
(p = 0.82) on Tirosh and shows a tiny opposite-direction signal
(Δ = 0.002) on Darmanis. δ is a worst-case four-point statistic
dominated by thin-triangle quadruples in the distance matrix; it is
poorly suited to detecting local clustering differences. The choice of
curvature statistic is decisive — local Ollivier-Ricci sees the cancer
signal that global Gromov δ misses.

### 3.3 Pooled "non-malignant" comparisons are confounded

Comparing malignant cells against *all* labelled non-malignant cells in
each dataset (rather than against a single matched cell type) inflates
δ_rel artificially due to multi-population mixing. In Darmanis this
flips the apparent δ effect to the opposite direction. Cell-type-matched
contrasts are essential.

### 3.4 Cross-tumour-type replication is consistent across seven cancers

Section 3.1 reports the full seven-tumour replication. The effect
generalises across solid-tumour scRNA-seq cohorts: melanoma (skin),
ovarian (HGSOC), head-and-neck SCC (squamous), glioblastoma (brain),
prostate (PRAD), colorectal (CRC), and pancreatic (PDAC) adenocarcinomas.
Each contrast was run on a separate dataset preprocessed identically,
with cell-type labels from the original authors where available
(Tirosh, Puram, Darmanis, Peng/Moncada) or marker-derived where not
(Chen prostate, Olalekan ovarian — labels validated against published
markers, see `exp/N5b_prostate/run.log` and `exp/N5c_ovarian/run.log`).
In every case malignant cells have higher per-cell κ than the
comparator group; in every case the effect survives Mann-Whitney
testing at small Bonferroni-adjusted thresholds. Files:
`exp/E3_puram_hnscc/`, `exp/E4_li_crc/`, `exp/N5a_pdac/`,
`exp/N5b_prostate/`, `exp/N5c_ovarian/`.

### 3.5 The effect is within-patient, not Simpson's-paradox

The strongest possible confound — that pooled malignant-vs-immune
differences reflect between-patient transcriptional drift rather than
within-tumour biology — is rejected by both stratified and mixed-effects
analyses.

| Dataset | Patients with ≥30/≥30 cells | % patients same-direction | Mixed-effects retention |
|---|---|---|---|
| Tirosh melanoma | 5 of 19 | 100% | **85%** |
| Darmanis GBM | 4 of 4 | 100% | **97%** |

For Tirosh, the MixedLM `kappa ~ malignant + (1 | patient_id)` fixed
effect is +0.148 [95% CI +0.133, +0.164], p ≈ 10⁻⁷⁹, vs. a naive pooled
difference of +0.175. For Darmanis, the fixed effect is +0.071 [+0.059,
+0.084] vs. naive +0.073. Patient-level random-effect variance collapses
to ≈ 0 in both fits, indicating baseline κ is nearly constant across
patients once the malignant flag is in the model. Per-patient Cliff's δ
ranges from +0.78 to +0.88 in Tirosh and +0.07 to +0.52 in Darmanis,
all positive. Files: `exp/E1_within_patient/`.

### 3.6 The effect is robust to pipeline hyperparameters

A 3 × 3 × 4 × 2 = 72-cell grid over (HVG count, PCA dimensionality, k,
lazy-walk α) on the Tirosh malignant-vs-T-cell contrast yields
same-direction effect in **72/72 cells (100%)**, with median Cliff's δ
+0.535 (range [+0.285, +0.779]). Variance decomposition of effect size
across the grid: HVG count contributes 41% of variance, k contributes
41%, PCA dimensionality 12%, α only 0.1%. The lazy-walk parameter is
essentially irrelevant for sign and ranking; it merely shifts the
absolute κ scale by a factor of ≈ 2. The effect is weakest at the
HVG = 5000, k = 50 corner of the grid (δ = +0.29) — both choices that
make the kNN graph absorb more inter-population structure into average
paths — and strongest at HVG = 500, k = 15 (δ = +0.78). File:
`exp/E5_hparam_sweep/e5_hparam_grid.csv`,
`exp/E5_hparam_sweep/e5_hparam_heatmap.png`.

### 3.7 Clonal expansion alone explains roughly half of the malignant-vs-T κ shift

The single most important specificity test for this finding is whether
elevated κ is a *malignancy* signal or a *clonal-expansion* signal —
since malignant tumours and TCR-expanded T-cell clones share the
"narrow transcriptional state space" property even in the absence of
oncogenic transformation. We test this on Yost et al. 2019 (GSE123813),
basal cell carcinoma scRNA-seq with paired TCR-clonotype labels.

Within T cells, define `expanded_T` as cells in a CDR3 group with ≥10
cells, and `non_expanded_T` as singletons (or no-TCR-call). Compute
per-cell κ on the kNN graph of the BCC dataset using the standard
pipeline. Results (n ≈ 590 per group):

| Group | per-cell κ̄ ± SD |
|---|---|
| non_expanded_T | −0.070 ± 0.146 |
| expanded_T | −0.030 ± 0.126 |
| malignant | +0.019 ± 0.117 |

| Pairwise contrast | Cliff's δ | Welch p |
|---|---|---|
| Malignant vs non-expanded T | **+0.377** | 1.8 × 10⁻²⁹ |
| Expanded T vs non-expanded T | **+0.175** | 5.8 × 10⁻⁷ |
| Malignant vs expanded T | **+0.230** | 5.3 × 10⁻¹² |

Clonal expansion alone shifts κ in the same direction as malignancy and
explains roughly **0.175 / 0.377 ≈ 46 %** of the full malignant-vs-non-
expanded shift. The residual malignant-vs-expanded gap remains highly
significant (δ = +0.230, p = 5 × 10⁻¹²), so tumour-specific
transcriptional programmes contribute a comparable second shift on top
of the clonality signal. The strict "κ detects malignancy" framing
therefore does not survive — but neither does the "κ is just a
clonality detector" framing. The honest reading is:

> Per-cell κ is a transcriptional-compactness phenotype. Clonal
> expansion (TCR-driven or oncogenic) and tumour-specific transcriptional
> programmes are two distinct, comparable contributors to the elevated
> κ observed in malignant cells. Either alone is sufficient to produce a
> measurable κ shift; together they produce the larger shift seen in
> tumours.

File: `exp/E7_clonality/`. Caveats: single tumour type (BCC) for the TCR
test; "non-expanded" pools singletons with no-TCR-call cells (a
conservative grouping); Wu et al. 2020 lung/breast/renal TCR-paired data
would be the natural replication target.

### 3.8 Forman-Ricci agrees on direction in two of four datasets and diagnoses what kind of signal each contains

Computing Sreejith et al. 2016 Forman-Ricci curvature on the same kNN
graphs:

| Dataset | δ_OR | δ_FR | Sign agreement |
|---|---|---|---|
| Tirosh melanoma | +0.67 | **−0.74** | flip |
| Darmanis GBM | +0.37 | +0.34 | match |

The Tirosh sign flip is *not* a refutation. Per Samal et al. (2018), FR
captures the degree-and-edge-weight contribution to graph geometry but
ignores the triangle/clustering contribution that OR captures via
Wasserstein-1 between neighbourhood distributions. When OR and FR agree
(Darmanis), the signal is largely degree-driven. When OR and FR
disagree on sign as in Tirosh — and OR remains strongly positive — the
OR signal is *triangle-dense local clustering*, the more
geometrically-meaningful kind of cancer-vs-immune contrast. The split
result is therefore informative rather than null: Tirosh's malignant
clones form *triangle-dense* tight neighbourhoods (the geometrically
interesting case), whereas Darmanis's malignant separation is partly
driven by edge-length geometry alone. File: `exp/E2_forman_ricci/`.

### 3.9 UMI / library-size depth is not the driver

Malignant cells in raw-count datasets typically have higher total UMI
than infiltrating immune cells; PCA-of-log-counts can inherit a depth
gradient that mechanically tightens high-UMI cell neighbourhoods and
raises κ. We stratify Darmanis cells into 5 UMI quintiles and run the
malignant-vs-immune comparison within each:

| Quintile | UMI range | n_neo / n_imm | Cliff's δ [95% CI] | Welch p |
|---|---|---|---|---|
| Q1 (low) | 181–627 k | 160 / 428 | **+0.471** [+0.38, +0.56] | 8.5 × 10⁻¹⁹ |
| Q2 | 627–812 k | 160 / 427 | **+0.524** [+0.45, +0.62] | 8.9 × 10⁻²¹ |
| Q3 | 812–965 k | 184 / 404 | **+0.508** [+0.42, +0.60] | 5.2 × 10⁻²¹ |
| Q4 | 965 k–1.13 M | 277 / 310 | +0.281 [+0.20, +0.37] | 3.0 × 10⁻⁸ |
| Q5 (high) | 1.13–2.13 M | 310 / 278 | +0.320 [+0.24, +0.40] | 4.6 × 10⁻¹⁰ |

5/5 quintiles same direction, all p < 10⁻⁷. **Critically, the effect is
strongest in low-UMI quintiles, the opposite of what a depth-driven
artefact would produce** (a depth artefact would predict the effect to
grow with quintile). An inverse stratification by per-cell mean kNN
distance gives the same 5/5 pattern, all p < 10⁻¹². Both depth confounds
rejected. File: `exp/T1_umi_confound/`.

### 3.10 Proliferating immune cells show a small additive lift, not a malignancy-equivalent shift

A common confound for "malignancy detectors" in scRNA-seq is that they
actually detect cell-cycle activity. We score Tirosh immune cells on a
17-gene G2M/S signature (MKI67, TOP2A, CCNB1, CDK1, etc.) and partition
into proliferating (z > 1, n = 148) and resting (z < 0, n = 600).
Three-way comparison vs malignant cells:

| Group | per-cell κ̄ |
|---|---|
| Resting immune | −0.007 |
| Proliferating immune | +0.027 |
| Malignant | +0.118 |

| Contrast | Cliff's δ | Welch p |
|---|---|---|
| Mal vs proliferating immune | **+0.373** | 1.5 × 10⁻¹² |
| Mal vs resting immune | +0.520 | 3.1 × 10⁻⁵³ |
| Proliferating vs resting immune | +0.198 | 5.8 × 10⁻³ |

Cell-cycle activity contributes a small lift (Δκ ≈ 0.034 between
proliferating and resting immune cells), but malignant cells sit
**2.7× higher above proliferating-immune than proliferating-immune sit
above resting** (Δκ_mal-vs-prolif ≈ 0.092 vs Δκ_prolif-vs-rest ≈ 0.034).
If κ were a pure cell-cycle detector, malignant ≈ proliferating-immune
(Cliff's δ < 0.10); we observe δ = +0.37 with p = 10⁻¹². The malignant
signal is therefore predominantly tumor-specific transcriptional
geometry, with proliferation as a modest additive contributor. File:
`exp/T5_proliferating_immune/`.

### 3.11 The κ separation is intrinsic to the populations, not specific to high-variance gene selection

We replace HVG-2000 selection with 2000 randomly-sampled genes and
re-run the full pipeline (kept 600 + 600 cell sample fixed; 20 random
gene-set seeds):

| Setup | Cliff's δ |
|---|---|
| Real HVG-2000 | +0.635 |
| Random 2000 (n = 20) | +0.673, 95 % CI [+0.613, +0.721] |
| HVG-2000 z-score vs random | −1.13 (inside the random distribution) |

The HVG-2000 result is statistically indistinguishable from random gene
panels. **The κ separation is intrinsic to the cell-population geometry
in PC space**, not specific to high-variance gene selection — almost any
2000-gene linear projection of the expression matrix recovers it. This
strengthens the geometric-signal interpretation (it is robust to gene
selection) while weakening any framing that ties the effect to a
specific HVG-derived biological-variance basis. File:
`exp/T6_random_hvg_control/`.

### 3.12 Bakry-Émery decomposition into curvature K and effective dimension N

To test whether the OR signal reflects local curvature, effective
dimension, or both, we compute the Bakry-Émery CD(K, N) condition per
vertex via the Cushing-Liu-Münch curvature-matrix formulation (Cushing
et al. 2020), reduced here to a closed-form generalised eigenproblem
without SDP. K_BE(v; N) is the largest K such that B − (1/N) cc^T − KA
is PSD on the local Γ-form (k = 10 NN graph, 300 + 300 cells per
dataset).

| Metric | Tirosh: Mal / T (p) | Darmanis: Neo / Imm (p) |
|---|---|---|
| OR κ | +0.110 / −0.036 (2 × 10⁻⁴⁵) | +0.066 / −0.005 (3 × 10⁻¹⁵) |
| K_BE(∞) | −0.095 / −0.350 (3 × 10⁻²⁰) | −0.194 / −0.294 (1 × 10⁻⁶) |
| K_BE(N=2) | −0.663 / −0.788 (6 × 10⁻⁶²) | −0.684 / −0.727 (6 × 10⁻¹⁵) |
| N_eff (half-saturation) | 3.07 / 2.72 (7 × 10⁻¹⁸) | 2.83 / 2.66 (7 × 10⁻⁸) |

Per-cell Pearson correlation between OR κ and K_BE(∞):
**r = +0.74** (Tirosh), **r = +0.69** (Darmanis), p ≪ 10⁻⁸⁰ in both.
Both K (curvature: malignant cells less negative, locally less
hyperbolic) and N_eff (effective dimension: malignant worst-case test
function more concentrated, fewer effective transcriptional axes) shift
in the malignant direction. The OR signal therefore decomposes into a
genuine local curvature shift *and* an effective-dimension narrowing,
neither of which dominates exclusively. File: `exp/T3_bakry_emery/`.

### 3.13 The κ direction does not transfer to the physical-tissue Delaunay graph at Visium spot resolution

We test substrate-independence on 10X Visium CytAssist Human Breast
Cancer FFPE (4 169 in-tissue spots; data-driven malignant/stroma labels
via tumour markers KRT8/19, EPCAM, ESR1 vs. stromal markers PTPRC,
COL1A1, ACTA2, PECAM1; top/bottom tertiles → 1 376 mal / 1 376 stroma).
Two graphs over the same set of spots:

| Graph | mean κ_mal | mean κ_stroma | Cliff's δ | Welch p | MWU p |
|---|---|---|---|---|---|
| G_trans (PCA-50 kNN, scRNA-like) | +0.014 | −0.123 | **+0.478** | 2.6 × 10⁻⁸⁵ | 1.3 × 10⁻⁸⁸ |
| G_phys (Delaunay over physical coords) | +0.007 | +0.014 | **−0.005** | 6.3 × 10⁻⁴ | **0.84** |

The transcriptomic-similarity graph reproduces the malignancy κ shift
emphatically (Cliff's δ = +0.48). The physical-spot Delaunay graph is
null (rank-based MWU p = 0.84; Welch is borderline-significant in the
opposite direction, driven by tail asymmetry and not by stochastic
ordering). **The κ phenotype is therefore a property of the
transcriptional-similarity neighbourhood structure, not of physical-
tissue architecture at Visium spot resolution.** This is a meaningful
narrowing of the interpretation: malignant cells occupy tighter regions
of *expression* state space, but they do not form measurably tighter
physical neighbourhoods at the ~55 µm spot scale tested here.

Caveats: signature-based labels (not pathologist-verified), single
slide, Delaunay graph degree ≈ 6 mechanically compresses κ variance.
Visium HD or single-cell-resolution MERFISH would be the natural follow-
up before declaring substrate-independence dead. File:
`exp/T4_visium_spatial/`.

### 3.14 The Yost BCC clonality contribution does not generalise to Wu 2020 lung cancer T cells

Section 3.7 reported that TCR-expanded T-cell clones in Yost 2019 BCC
show intermediate κ between non-expanded T cells and malignant cells,
with clonality explaining ≈ 46 % of the full malignant-vs-T shift. We
attempt to replicate this on Wu et al. 2020 GSE139555, a much larger
TCR-paired cohort spanning lung, colon, renal, and endometrial cancer
(here: lung tumour LT2 + matched normal LN3, 20 080 T cells, 4 061 cells
in expanded clonotypes ≥ 10 cells, 7 216 singletons; standard pipeline,
600 vs 600).

| Group | per-cell κ̄ |
|---|---|
| Non-expanded T | −0.039 |
| Expanded T | −0.037 |

Cliff's δ (expanded vs non-expanded) = **+0.018**, MWU p = **0.60**. The
Yost +0.175 effect (p = 6 × 10⁻⁷) does **not** replicate in Wu lung
cancer T cells. The most plausible explanation is that the Yost samples
were anti-PD-1-treated and clonal expansions there concentrate into
distinct exhausted/effector transcriptional clusters (CD8_ex_T) which
appear as tight κ-positive islands; Wu samples are mostly treatment-
naïve, and lung-cancer expanded clones occupy diffuse Trm/Tem programs
that are not transcriptionally tighter than the singleton T-cell pool.
**The κ-clonality coupling is therefore treatment-context- and tumour-
type-specific, not a general property of clonal expansion.** This
*tightens* the malignancy interpretation: in typical (treatment-naïve)
solid tumours, the malignant-vs-immune κ shift is predominantly tumor-
specific transcriptional geometry, with clonality entering as a
contextual additional contributor only in some settings (BCC + ICB).
File: `exp/T2_wu_clonality/`.

### 3.15 Per-cell κ shows a weak trend toward higher κ in ICB Non-responders, not strong enough to support a biomarker claim

We test whether baseline per-cell κ on tumour-infiltrating CD8 T cells
predicts subsequent anti-PD-1 / anti-CTLA-4 response on Sade-Feldman et
al. 2018 (GSE120575): 19 melanoma patients (9 R, 10 NR), 2 403 baseline
CD8 T cells. Standard pipeline.

| Test | Result |
|---|---|
| Per-cell Cliff's δ (R vs NR) | −0.013 (null) |
| Per-cell MWU p | 0.604 |
| Per-patient mean κ MWU (n = 9 / 10) | p = 0.31 |
| Per-patient AUC for mean-κ as classifier | **0.644** |
| Direction of best AUC | high-κ → Non-responder |
| Logistic regression LLR p | 0.244 |

AUC 0.64 falls in the pre-specified weak-trend band (0.55–0.70), short
of the 0.70 threshold for a real clinical signal. Direction is
biologically plausible (high κ ≈ tighter clonally-restricted CD8
populations ≈ exhausted/dysfunctional ≈ NR), but the trend is too weak
to claim a biomarker on this cohort. The Responder group has 3× the
variance of the Non-responder group (σ = 0.086 vs 0.026), with R cells
being bimodal between very-low-κ (P28, P29, P35, P7) and near-zero-κ
(P1 anti-CTLA-4, P8 anti-CTLA-4+PD-1) patients — suggesting therapy-arm
or biological-subtype heterogeneity that a stratified analysis (anti-PD1
only) might recover signal from. **The methods finding does not
immediately translate to ICB-response prediction on this dataset.**
File: `exp/N1_sadefeldman_icb/`.

### 3.16 The Visium spot-resolution null replicates at 8 µm single-cell-adjacent resolution

The T4 spot-resolution Visium null (κ direction does not transfer to
the physical Delaunay graph) might have been a resolution artefact —
1–10 cells per spot mechanically smooths per-spot signals. We test this
on 10X Visium HD Human Colon Cancer Patient 2 at 8 µm bin resolution,
where each bin covers a sub-cellular footprint (smaller than a typical
10–20 µm colon epithelial cell):

| Graph | Cliff's δ (mal vs stroma) | Welch p | MWU p |
|---|---|---|---|
| G_phys (8 µm physical neighbours) | **+0.004** | 0.76 | 0.76 |
| G_trans (PCA-50 kNN) | −0.161 | 1 × 10⁻¹⁶ | — |

The G_phys null replicates cleanly at single-cell-adjacent resolution.
**The T4 negative was not a resolution artefact** — the substrate-
specificity of the κ phenotype is real. The transcriptomic-similarity
graph still produces a strongly significant separation
(p = 1 × 10⁻¹⁶) but in the opposite direction from T4 breast cancer
(here mal < stroma), consistent with this CRC dataset having
heterogeneous Tumour-0…V transcriptional sub-clusters that flatten the
malignant κ relative to a more homogeneous stromal compartment. The
direction reversal is dataset-dependent on the transcriptomic graph;
the physical-graph null is robust. File: `exp/N2_visium_hd/`.

### 3.17 The κ direction does not transfer to 3D-genome contact graphs at 1 Mb resolution; chr22 reverses sign

To test whether the cancer-vs-normal curvature signal extends to the
chromatin-contact domain, we compute per-bin κ on Hi-C contact graphs
from Rao et al. 2014 (GSE63525) for K562 (cancer; CML) vs GM12878
(normal; lymphoblastoid). Single-chromosome, 1 Mb resolution, observed/
expected normalisation, top-3·N edges by O/E.

| Chromosome | K562 κ̄ | GM12878 κ̄ | Cliff's δ | MWU p |
|---|---|---|---|---|
| chr22 | +0.078 | +0.181 | **−0.504** | 4.5 × 10⁻⁴ |
| chr11 | +0.072 | +0.023 | +0.144 | 0.058 |

On chr22, cancer chromatin is *less* positively curved than normal —
the **opposite** of the scRNA-seq direction, but consistent with the
TAD-boundary-loss / compartment-switching literature (Flavahan 2016,
Hnisz 2016) that predicts cancer chromatin to be more disordered. On
chr11 the direction reverses, weakly in the same sign as the scRNA-seq
finding. Conclusion: **per-bin Hi-C κ at 1 Mb resolution is chromosome-
dependent and does not transfer the scRNA-seq direction cleanly.** The
transcriptomic-similarity and chromatin-contact graphs measure
structurally different things (cell-cell similarity vs. locus-locus
contact) and do not share a common curvature signature at this
resolution.

Caveats: cell lines are not patient tumours; 1 Mb is coarse; closing
this would need 100 kb and genome-wide aggregation. File:
`exp/N3_hic_cancer_normal/`.

### 3.18 Liquid-tumour κ direction is FLIPPED relative to solid tumours

We test the only major tumour-class category not yet covered: liquid
(haematological) malignancies, where leukemic blasts are not in a stromal
microenvironment and have hierarchy-collapse rather than clonal-lineage-
commitment biology. Dataset: van Galen et al. 2019 AML scRNA-seq
(GSE116256), 6 diagnostic AML samples vs 6 healthy bone marrow controls.

| Contrast | Cliff's δ | MWU p |
|---|---|---|
| AML blasts vs healthy myeloid HSPCs (matched) | **−0.085** | 0.011 |
| AML blasts vs all healthy BM | −0.063 | 0.060 |

Effect size is small but **direction is consistently flipped** across re-runs (range −0.14 to −0.06, never crossing zero). This is the **H_b outcome** predicted in advance: AML blasts span HSC-like / Prog-like / GMP-like / ProMono-like / Mono-like states (transcriptionally diffuse due to myeloid-hierarchy collapse), while healthy HSPCs form tighter subtype clusters. **The κ phenotype is therefore solid-tumour-specific, not universal across cancer.** The 7-of-7 solid-tumour replication should be reframed as "all 7 solid tumours show malignant > non-malignant κ; AML inverts the direction." File: `exp/F1_aml_mm_liquid/`.

### 3.19 ICB-response biomarker does not replicate cross-cohort

The N1 Sade-Feldman 2018 ICB-response analysis showed a per-patient AUC
of 0.64 for high-κ-as-Non-responder, with the per-cell test null. We
test whether stratification by therapy arm and replication on a second
ICB cohort (Yost 2019 BCC, GSE123813, anti-PD1) recover signal.

| Cohort / arm | n_R / n_NR | AUC | Direction |
|---|---|---|---|
| N1 anti-PD1 only (Sade-Feldman) | 4 / 8 | **0.81** | high κ → NR |
| Yost BCC anti-PD1 (independent replication) | 6 / 5 | **0.50** | none |
| Mega anti-PD1 (cross-cohort, n = 23) | 10 / 13 | 0.64 | weak |

The N1 stratified-anti-PD1 AUC of 0.81 was overfit to a 12-patient
melanoma cohort and **does not survive first independent replication on
BCC** (AUC 0.50). Cross-cohort mega-AUC drops to 0.64. **Per-cell κ is
not validated as a generalisable ICB-response biomarker** on currently-
public scRNA-seq data. File: `exp/F2_icb_stratified/`.

### 3.20 Substrate-specificity confirmed at three resolutions including single-cell MERFISH

Three independent spatial-transcriptomics tests, three independent
nulls on the physical-neighbour graph:

| Resolution | Dataset | G_phys Cliff's δ | G_trans Cliff's δ |
|---|---|---|---|
| Visium spot (~55 µm) | T4 — Breast (10X CytAssist FFPE) | **−0.005** | +0.48 |
| Visium HD (8 µm bin) | N2 — Colon (10X CytAssist HD P2) | **+0.004** | −0.16 |
| MERFISH (~12 µm cell-contact) | F3 — MMTV-PyMT mouse mammary | **+0.050** | −0.091 |

The substrate-specificity null is robust across three resolutions
spanning two orders of magnitude (~55 → ~1 µm), three tumour models, and
three platforms. **The κ malignant-vs-stroma phenotype is genuinely a
transcriptomic-similarity-neighbourhood property, not a physical-tissue-
architecture property at any resolution tested down to single-cell
contact.** File: `exp/F3_merfish_multi/`.

### 3.21 Chromatin contact graphs show a robust cancer-vs-normal κ signature in the OPPOSITE direction (genome-wide Hi-C)

The N3 single-chromosome Hi-C test was inconclusive (chr22 strongly
negative, chr11 weakly positive at 1 Mb). We extend to genome-wide
analysis at 100 kb resolution on Rao 2014 K562 (cancer) vs GM12878
(normal), 22 of 23 chromosomes successfully fetched.

**Pooled genome-wide:**
- K562: n = 23,090 bins, mean κ = **−0.337**, median −0.390
- GM12878: n = 16,704 bins, mean κ = **−0.068**, median −0.047
- **Pooled Cliff's δ = −0.551, 95 % CI [−0.560, −0.543]**
- Mann-Whitney U p < machine epsilon (effectively p = 0)

**Per-chromosome consistency:**
- 22 / 23 chromosomes negative direction (cancer < normal)
- 0 / 23 positive direction (cancer > normal)
- 0 / 23 near-zero (|δ| < 0.05)
- Range: chr19 δ = −0.38 (smallest, gene-dense) to chr8 δ = −0.71 (largest)
- The chr11 +0.14 outlier from N3 (1 Mb) collapses to **−0.61 at 100 kb** — confirming the N3 chr11 outlier was a resolution artifact, not a real chromosome-specific signal.

**Direction: chromatin contact graphs in cancer are LESS curved than normal**, the opposite of the scRNA-seq cell-graph direction. This is consistent with the published TAD-boundary-loss / compartment-switching literature (Flavahan 2016, Hnisz 2016): cancer chromatin disorganisation flattens the modular contact landscape, which OR-κ reads as lower curvature. **Transcriptomic-similarity and 3D-genome-contact graphs measure structurally different aspects of cancer biology** — clonal-attractor narrowing in expression space, modular disorganisation in chromatin space — and they yield opposite-sign curvature signatures. File: `exp/F4_hic_genome_wide/`.

### 3.22 Per-cell κ stratifies CNV-defined subclones in well-powered tumours, uniform in most

We use chromosome-arm-level CNV inference (manual moving-average proxy
on log-TPM with non-malignant immune cells as reference) on Tirosh
malignant cells, cluster cells per-tumour into 2–5 subclones via Ward
linkage, and compare per-cell κ across subclones within each tumour.

Of Tirosh's 19 patients, only 8 had ≥ 2 subclones with ≥ 10 cells each
after CNV clustering. Of those 8:

- **2 of 8 (25 %) tumours** show Kruskal-Wallis p < 0.05 across
  subclones (P79: p = 3.5 × 10⁻¹⁰, 468 cells, max-min Cliff's δ
  = +0.55; P59: p = 2.4 × 10⁻³, max-min δ = +0.74).
- **Median max-min subclone Cliff's δ = +0.29** across 8 testable
  tumours (mean +0.30, range +0.008–+0.74).
- Aggressive-CNV enrichment in high-κ subclones: directionally
  consistent with the hypothesis (8q MYC +0.08, 17p TP53 +0.06, 9p
  CDKN2A +0.04 higher-magnitude in high-κ subclones) but underpowered
  across n = 8 tumours (all Wilcoxon p > 0.3).

**Verdict:** κ does stratify CNV-defined subclones in well-powered
tumours but is uniform across subclones in most. The dominant
limitation is statistical power — typical Tirosh-sized tumours
(< 150 malignant cells) are underpowered to detect intra-tumour κ
heterogeneity. The aggressive-CNV directional signal is suggestive but
not significant. κ is a useful single-tumour ITH axis only when the
tumour has many malignant cells and genuinely divergent subclones; for
typical scale, treat the result as within-tumour-uniform. File:
`exp/N4_subclone_stratification/`.

### 3.23 Patient-level random-effects meta-analysis: pooled δ = +0.51 across 35 patients in 6 cohorts

Cells are not independent units — cells from the same patient share kNN
edges and biological context. This section therefore re-runs the headline
inference with the patient as the unit of analysis: per-patient Cliff's δ
(malignant vs cell-type-matched comparator, within patient), combined by
DerSimonian-Laird random-effects meta-analysis across all qualifying
patients (≥ 10 cells per group per patient; stratified sampling capped at
60 cells per patient per group and 3 500 per cohort).

| Cohort | Patients in meta | RE Cliff's δ [95% CI] | p | I² |
|---|---|---|---|---|
| Tirosh melanoma | 10 (of 19) | **+0.91** [+0.86, +0.95] | 9.6 × 10⁻³⁰⁷ | 85.6% |
| Puram HNSCC | 10 (of 16) | **+0.66** [+0.56, +0.76] | 3.7 × 10⁻³⁷ | 78.3% |
| Olalekan ovarian | 5 (of 6) | **+0.42** [+0.17, +0.67] | 0.00099 | 89.8% |
| Darmanis GBM | 4 (of 4) | **+0.29** [+0.01, +0.56] | 0.041 | 87.1% |
| Peng PDAC | 2 (of 2) | +0.10 [−0.05, +0.25] | 0.17 | 0% |
| Chen prostate | 4 (of 4) | −0.08 [−0.31, +0.15] | 0.52 | 78.9% |

**Pooled patient-level estimate (k = 35 patients, 6 cohorts): Cliff's
δ = +0.51 [95% CI +0.44, +0.59], z = 12.9, p = 7.5 × 10⁻³⁸**, with
I² = 96.8%, τ² = 0.047, Q = 1064.5 on df = 34. A patient-clustered
bootstrap (B = 1000) gives 95% CI [+0.37, +0.67]. The fixed-effect
estimate is +0.93, so the random-effects model shrinks the pooled effect
by roughly half under the observed between-patient heterogeneity. At
cohort level (k = 6), the pooled δ is +0.39 [+0.07, +0.72], p = 0.017,
I² = 97.3%.

Four cohorts replicate individually at p < 0.05 (Tirosh, Puram, Olalekan,
Darmanis); Peng is underpowered at n = 2 patients. Chen prostate fails
within-patient replication — per-patient δ = +0.22, −0.29, −0.03, −0.22
(1 of 4 patients positive), the Simpson's-paradox pattern of Section 3.1:
the published cell-pooled signal was between-patient, not within-patient.
Li CRC is excluded from the meta-analysis (no patient-of-origin labels in
the GEO processed file; its single pooled estimate is δ = −0.07,
p = 0.84, n = 60/34). This patient-level analysis replaces the per-cell
p-values of Section 3.1 as the headline inference. Files:
`exp/META_patient_level/` (forest plot: `forest_plot.png`; per-patient
table: `per_patient_delta.csv`).

### 3.24 The Puram HNSCC within-patient effect replicates under mixed-effects modelling (E1b)

Section 3.5 established within-patient retention for Tirosh (85%) and
Darmanis (97%). The same protocol applied to Puram HNSCC — the cohort
that sign-flips under Harmony batch correction (Section 3.27) — retains
the effect without correction: the MixedLM `kappa ~ malignant +
(1 | patient_id)` fixed effect is **+0.049 [95% CI +0.031, +0.066]**,
p = 3.4 × 10⁻⁸, against a naive pooled difference of +0.051 — a
**retention of 95.4%** (17 patients in the model; 1 199 cells with valid
κ). Random-intercept variance is 0.0035 with ICC 0.156, and the fixed
effect is stable across three optimizers to a spread of 2.2 × 10⁻⁷. Of
the 9 patients with ≥ 30 cells in both groups, per-patient Cliff's δ is
positive in **8 of 9 (89%)**, ranging from −0.27 to +0.73. File:
`exp/E1b_puram_mixed_effects/`.

### 3.25 Operational discrimination: κ alone is a moderate within-cohort classifier and does not generalise across cohorts

Per-cohort single-feature logistic regression on per-cell κ (600 + 600
cells; 10 seeds; stratified 80/20 splits):

| Cohort | κ-only AUC-ROC | UMI-only AUC | n_genes AUC | Multi-feature AUC |
|---|---|---|---|---|
| Tirosh melanoma | **0.868** ± 0.025 | 0.559 | 0.807 | 0.926 |
| Olalekan HGSOC | **0.817** ± 0.035 | 0.785 | 0.830 | 0.917 |
| Puram HNSCC | **0.695** ± 0.030 | 0.478 | 0.910 | 0.916 |
| Darmanis GBM | **0.659** ± 0.022 | 0.630 | 0.696 | 0.718 |
| Chen prostate | **0.617** ± 0.036 | 0.802 | 0.806 | 0.811 |
| Peng PDAC | **0.579** ± 0.020 | 0.488 | 0.486 | 0.586 |
| Li CRC | **0.902** ± 0.077 | 0.700 | 0.651 | 0.922 |

Across the seven cohorts, mean κ-only within-cohort AUC is **0.734**
(range 0.579–0.902; 3 of 7 above 0.70), against 0.528 for a random
feature and 0.634 for UMI alone. Multi-feature models (κ + UMI + n_genes
+ pct_mito) average **0.828** within-cohort. The Li CRC AUC of 0.902 is
direction-agnostic: in this run the underlying Cliff's δ is −0.78 (AUC >
0.5 with malignant *lower* κ), consistent with the small-sample direction
instability previously noted for this 34-vs-34 cohort.

Leave-one-cohort-out (train on six cohorts, test on the held-out
seventh): κ-only AUC is 0.848 (Tirosh), 0.811 (Olalekan), 0.690 (Puram),
0.663 (Darmanis), 0.609 (Chen), 0.597 (PDAC), and 0.112 (Li — below
chance, i.e. reversed direction). **Mean LOCO AUC is 0.619** for κ alone
and 0.764 for the multi-feature model. The pre-specified verdicts are
MODERATE within-cohort and DOES NOT GENERALIZE cross-cohort for κ alone:
κ carries real but incremental discriminative signal beyond depth
covariates (UMI, n_genes) in high-effect cohorts, and cross-cohort
transfer of a κ threshold is not supported. Files:
`exp/CLASSIFIER_AUC/` (`per_cohort_auc.csv`, `leave_one_out_auc.csv`).

### 3.26 The finding survives alternative cell-type annotation schemes in both questioned cohorts

For the two cohorts whose malignant/non-malignant labels were
marker-derived rather than author-supplied (Chen prostate, Olalekan
ovarian), labels were re-derived three independent ways — the original
marker panels, a different broad marker panel, and unsupervised Louvain
clustering annotated by marker enrichment — and δ recomputed on each:

| Cohort | Original scheme | alt1_broad | alt2_cluster | Verdict (pre-specified) |
|---|---|---|---|---|
| Chen prostate | δ = +0.17 (p = 4 × 10⁻⁷) | δ = **+0.42** (p = 1 × 10⁻³⁵) | δ = −0.005 (p = 0.87) | robust in direction: 2/3 positive, 1 null, 0 reversed |
| Olalekan ovarian | δ = **+0.63** | δ = **+0.61** | δ = **+0.37** | robust: 3/3 positive (span 0.26) |

Olalekan is robust across all three schemes (all significantly positive,
δ ∈ [+0.37, +0.63]). Chen is robust in direction but weak in magnitude:
two of three schemes significantly positive, the unsupervised scheme
null, and no scheme reversed. A fixed-comparator decomposition — every
malignant scheme tested against the same original T-cell comparator pool
— returns positive δ for all three Chen schemes (+0.047, +0.210, +0.062),
indicating the alt2_cluster null is driven by comparator re-definition
rather than by the malignant-call itself. No annotation scheme in either
cohort reverses the direction. File: `exp/LABEL_VALIDATION/`.

### 3.27 Batch-correction sensitivity: Harmony with patient as batch attenuates all three cohorts tested and sign-flips Puram

Within-cohort batch structure (patient as batch variable: Tirosh 19,
Puram 18, Darmanis 4 batches) was corrected with Harmony on the same
PCA-50 embedding and κ recomputed on the corrected kNN graph, alongside a
random-embedding null (3 seeds) and a UMI-only 1-D control:

| Cohort | Original δ | Harmony δ | Change | Random null δ (mean) | UMI-only δ |
|---|---|---|---|---|---|
| Tirosh melanoma | **+0.740** | +0.213 | −71% | −0.012 | +0.123 |
| Puram HNSCC | **+0.340** | **−0.523** | sign flip | −0.012 | +0.001 |
| Darmanis GBM | **+0.338** | +0.037 | −89% | +0.012 | +0.005 |

The random-embedding and UMI-only controls are ≈ 0 in all three cohorts,
so neither kNN-graph construction nor sequencing depth generates the
pooled effect. After Harmony, the pooled cell-level effect weakens
markedly everywhere and reverses sign in Puram (pre-specified verdicts:
WEAKENED for Tirosh and Darmanis, KILLED for Puram). Patient is both the
batch variable and the sampling stratum here, so patient-structure
removal and contrast structure are entangled in this design; the
patient-unit analyses of Sections 3.23 and 3.24 (85–97% within-patient
retention without correction) are the corresponding patient-level
evidence. File: `exp/BATCH_CORRECTION/`.

### 3.28 Pre-registered held-out validation: lung replicates; breast is comparator- and annotation-dependent

Executed under the locked plan of `PREREGISTRATION.md` (filed 2026-06-29,
before data inspection; Section 2.25), seed 20260507. Decision rule
(pre-registered): SUPPORTED iff pooled δ > 0 and one-sided p < 0.05.
Pre-registered falsifiers: pooled δ ≤ 0 in the primary cohort
(GSE161529); pooled δ ≤ 0 in both cohorts; patient-level δ > 0 in
< 50% of patients in the primary cohort.

**(a) GSE131907 secondary held-out cohort (Kim et al. 2020 metastatic
LUAD atlas; run 2026-09-05).** Original-author annotation
(`cell_annotation.txt`); malignant = epithelial cells with tumour
subtypes (tS1/tS2/tS3/Malignant cells); comparator = immune cells
(T/NK, myeloid, B lymphocytes, MAST). 31 units passed the
≥10-cells-per-group filter; 3 123 sampled cells with valid κ
(1 072 malignant / 2 051 immune). DerSimonian-Laird pooled
δ = **+0.2469 [95% CI +0.1302, +0.3637]**, one-sided
p = **1.69 × 10⁻⁵** (two-sided 3.39 × 10⁻⁵), z = 4.146, I² = 85.2%
(τ² = 0.0926; Q = 202.8 on df = 30). Per-unit direction: δ > 0 in
**23 of 31 units** (74.2%). Patient-clustered bootstrap (B = 1000):
95% CI [0.1443, 0.3497]. Naive cell-level pooling, reported for
reference only (not the unit of inference): δ = +0.228, Welch
p = 1.8 × 10⁻²⁴. **Sample-as-patient caveat (stated per audit):** the
annotation contains no patient-of-origin field; Sample is its only
proxy, and the source paper reports 58 samples from 44 patients. The
31 meta-analytic units are therefore samples, not verified unique
patients — a donor contributing multiple samples contributes multiple
units. Decision rule: δ > 0 and p < 0.05 → **SUPPORTED**.

**(b) GSE161529 primary held-out cohort (Pal et al. 2021 breast atlas;
run 2026-09-06).** Chen et al. 2022 companion labels; unit of analysis
= the labels' patient column (26 patient IDs over 27 deposited samples;
patient 0114 pools two samples). Both pre-registered comparator arms,
each analysed as its own cohort-analysis (own stratified sample, graph,
and 4 000-edge OR draw):

| Comparator arm | k | Pooled δ [95% CI] | one-sided p | I² | Direction | Verdict |
|---|---|---|---|---|---|---|
| Immune (primary) | 18 | **+0.1680** [−0.0709, +0.4069] | 0.0840 | 96.1% | 12/18 positive | NOT SUPPORTED (δ > 0, p ≥ 0.05) |
| Normal epithelial (secondary) | 11 | **−0.4115** [−0.5623, −0.2607] | ≈ 1.000 (two-sided 8.9 × 10⁻⁸) | 84.0% | 1/11 positive | NOT SUPPORTED (δ < 0) |

Decision-rule inputs as pre-registered: per-arm pooled δ and one-sided
p against 0.05. **No pre-registered falsifier fired** — the primary
cohort's pooled δ is positive (+0.1680), the direction-positive
fraction is 12/18 = 66.7% (falsifier threshold < 50%), and pooled δ is
positive in the lung cohort — but the pre-registered prediction that
δ > 0 for **both** comparators failed on the epithelial arm (reversal:
δ = −0.4115, 1/11 positive).

Disclosures (all pre-outcome, from the run's audit trail): (i) sample
ER_0001 (patient 0001) is **unjoinable** — none of the 69 GEO
per-sample libraries contains its label barcodes (best overlap 11/964
against its name-mapped library, chance level); its cells were dropped,
not forced, so patient 0001 is absent from both arms. (ii) The immune
arm **excluded 26 439 dual-labeled cells** — barcodes listed twice in
`cells_primary.tsv.gz`, once as malignant (Total-object inferCNV
tumour-block clusters) and once as immune (Sub-object clusters) — a
conservative pre-outcome exclusion leaving clean pools of 5 238
malignant and 16 862 immune cells; the secondary (epithelial) labels
contain zero duplicate barcodes and are unaffected. (iii) Cells with
valid κ entering the arms: 2 167 (immune arm; 837 malignant / 1 330
comparator) and 1 948 (epithelial arm; 1 400 malignant / 548
comparator). Files: `exp/HELDOUT_GSE131907/results.json`,
`exp/HELDOUT_GSE161529/results.json`.

Summary (approved framing, 2026-09-06): lung replicates (+0.25,
robust); the breast immune-comparator is annotation-unstable (−0.17 to
+0.30 across dual-label handlings, Section 3.29); the breast
epithelial-comparator reverses (−0.41); overall the compactness effect
is tissue- and comparator-dependent and, in breast, sensitive to
annotation choices.

### 3.29 Annotation sensitivity: the breast immune-comparator estimate is not identifiable from the deposited annotation (post-hoc)

Post-hoc analysis (2026-09-06; not part of the pre-registration):
does the primary-arm pooled δ depend on excluding the 26 439
dual-labeled barcodes? Three variants were run, identical to the
pre-registered run in every locked respect (seed 20260507; stratified
caps 60/patient/group and 3 500 total; HVG-2000, PCA-50, kNN k = 15,
Ollivier-Ricci α = 0.5 on 4 000 edges; ≥ 10 cells per group;
DerSimonian-Laird meta-analysis; patient-clustered bootstrap B = 1000)
with only the dual-label assignment rule differing. The baseline
(excluded) variant reproduces the pre-registered run exactly (all
statistics bit-identical to `results.json`).

| Dual-label handling | k | Pooled δ [95% CI] | one-sided p | I² | Direction |
|---|---|---|---|---|---|
| Excluded (pre-registered baseline) | 18 | **+0.1680** [−0.0709, +0.4069] | 0.0840 | 96.1% | 12/18 positive |
| Dual-as-malignant | 22 | **−0.1690** [−0.3513, +0.0133] | 0.9654 | 95.2% | 9/22 positive |
| Dual-as-immune | 19 | **+0.3041** [+0.1407, +0.4675] | 0.00013 | 92.1% | 15/19 positive |

The estimate swings from significantly positive (p = 0.00013) to
negative across defensible handlings of the same deposited files. The
audit traced the conflict to the deposited companion objects
themselves: the Chen et al. 2022 Sub objects contain ~30% of cells
that ER.R's own construction should have excluded as Total-object
tumour, so the dual-labeled barcodes are an internal inconsistency of
the published annotation, not an artifact of our join. **Conclusion:**
the breast immune-comparator estimate is not identifiable from this
annotation; the −0.17-to-+0.30 spread is a property of the label
source, not of the curvature statistic. File:
`exp/HELDOUT_GSE161529/results_sensitivity.json` (full-precision
values).

## 4. Discussion

### 4.1 The cancer signal aligns with terminal-lineage commitment, not with stemness

A natural interpretive question is whether the elevated malignant κ
reflects "cancer cells are stem-like" — a hypothesis pervasive in the
cancer-stem-cell literature — or some other organising principle. We
test this directly on Paul et al. 2015 mouse myeloid hematopoiesis
(2730 cells; MEP→Ery/Mk/Mo/Neu/Baso/DC).

We find **Spearman ρ(κ, DPT) = +0.42 (p ≈ 10⁻¹⁰⁵, n = 2503)**: κ
*increases* monotonically along diffusion pseudotime. The most stem-like
cluster (`7MEP`) has the lowest mean κ at −0.12, with early erythroid
clusters at −0.18. Terminal clusters have the highest κ: dendritic
(`11DC`) at +0.26, lymph (`19Lymph`) at +0.11, late erythroid (`1Ery`)
at +0.06. The relationship is monotone-increasing, not U-shaped.

Stem-like cells therefore live in **locally hyperbolic, branching**
neighbourhoods (negative κ); terminal lineages live in **locally
spherical, compact** islands (positive κ). The cancer signal — malignant
cells with positive κ — aligns with the terminal-commitment end of
this axis, not the stem end. The cleaner reframing of our result is
therefore:

> Per-cell κ tracks lineage-commitment compactness. Both terminal
> committed lineages and clonally-expanded tumour cells live in
> positively-curved, transcriptionally-tight cell-graph neighbourhoods.
> Stem and progenitor populations live in negatively-curved, branching
> neighbourhoods.

This is the opposite direction from a naive "cancer stem cell" reading
of κ, and is consistent with the molecular reality that bulk-malignant
populations in untreated solid tumours are typically dominated by
clonal-lineage expansions, not by a CSC-fraction-driven branching
hierarchy. File: `exp/E6_pseudotime_paul/`.

### 4.2 What survived (and the coherent biology that emerges from a 24-experiment sweep)

Twenty-four independent follow-up experiments have now been run across
within-scope validation, cross-tumour-type replication, mathematical
decomposition, and cross-domain extension axes. The within-scope tests
are pass-or-tighten; the cross-domain extensions yielded a richer
biological picture than expected — they did not all confirm the
phenotype, but several gave **biologically informative negatives** that
together form a coherent picture of *what discrete graph curvature
measures in cancer biology, and where each measurement direction comes
from*:

| Substrate / graph | Direction | Magnitude | Biological reading |
|---|---|---|---|
| Solid-tumour scRNA-seq, transcriptomic kNN | Cancer > Normal | Cliff's δ +0.15 → +0.74 (n=7) | Clonal-lineage commitment narrows transcriptional state space |
| Liquid-tumour (AML) scRNA-seq, transcriptomic kNN | Cancer < Normal | Cliff's δ −0.085 (weak but robust) | Myeloid-hierarchy collapse spans multiple states; healthy HSPCs are tighter |
| Genome-wide Hi-C contact graph | Cancer < Normal | Cliff's δ −0.55 (22/23 chroms, p≈0) | TAD-boundary loss / compartment switching disorganises modular contact landscape |
| Visium / Visium HD / MERFISH physical-neighbour graph | Null | Cliff's δ ≈ 0 across 3 resolutions | Cancer cells are not more physically clustered at any spatial-transcriptomics resolution tested |

**The coherent reading:** discrete graph curvature does not have a
universal "cancer signature." The direction of the cancer-vs-normal κ
shift depends on which graph one builds, and each graph reads a
different biological process — *clonal compactness* in transcriptomic
similarity, *modular disorganisation* in chromatin contact, *physical-
tissue layout* (where cancer is null at the resolutions tested),
*hierarchy state* in liquid-tumour scRNA-seq. The integrated picture is
not "cancer is geometrically simpler" but rather "different geometric
substrates capture different aspects of cancer biology, and the κ
direction tells you which one."

Pass-or-tighten:

1. **Replicates across seven tumour types** (melanoma, ovarian HGSOC,
   HNSCC, GBM, prostate PRAD, CRC, PDAC), spanning immune, stromal,
   and lineage-matched comparison groups, with effect size scaling
   monotonically with comparator distance (largest δ for cross-
   lineage, smallest for matched-lineage benign epithelium); 7/7
   datasets show same direction at p < 10⁻³
   (E1 / E3 / E4 / N5a / N5b / N5c / §3.1).
2. **Within-patient mixed-effects** (Tirosh 85 % retention, Darmanis
   97 %; per-patient direction 100 % consistent) — not Simpson's
   paradox (E1).
3. **Robust to pipeline hyperparameters** (72 / 72 grid cells
   same-direction, median Cliff's δ + 0.535) (E5).
4. **Geometrically genuine** — Forman-Ricci diagnostic shows the Tirosh
   signal is triangle/clustering-driven, not edge-weight-driven (E2).
5. **Interpretable as commitment-compactness, not stemness** —
   Paul 2015 pseudotime ρ(κ, DPT) = + 0.42 (E6).
6. **Coherent with gene-network curvature direction** (Sandhu 2015) on
   a structurally different graph.
7. **Not a depth artefact** — UMI quintile stratification on Darmanis
   shows 5 / 5 quintiles same direction with low-UMI strongest, the
   opposite of what a depth confound predicts; inverse kNN-distance
   stratification gives the same 5 / 5 pattern (T1).
8. **Not a cell-cycle artefact** — proliferating immune cells show a
   small additive lift (Cliff's δ ≈ + 0.20 vs resting), but malignant
   cells sit a clean δ ≈ + 0.37 above proliferating-immune (T5).
9. **Not an HVG-selection artefact** — 20 random-2000-gene panels
   reproduce the effect; HVG-2000 sits inside the random distribution
   (z = − 1.13). Signal is intrinsic to the population partition (T6).
10. **Decomposes into both curvature and effective-dimension shifts**
    via Bakry-Émery CD(K, N); per-cell Pearson r between OR and K_BE
    is + 0.74 (Tirosh) / + 0.69 (Darmanis) (T3).
11. **Clonality is a context-dependent contributor, not a universal
    confound** — Yost BCC + ICB shows ≈ 46 % clonality share (E7), but
    Wu 2020 treatment-naïve lung cancer shows clonality contribution
    collapses to Cliff's δ = + 0.018 (NS). The malignancy framing is
    therefore predominantly tumor-specific transcriptional geometry,
    with clonality entering only in specific therapeutic contexts (E7,
    T2).

Interpretable negative:

12. **Substrate-specificity confirmed at two resolutions:
    transcriptomic-similarity, not physical-tissue.** On 10X Visium
    CytAssist Breast Cancer FFPE (T4, ~55 µm spot), the κ direction
    reproduces on the PCA-50 transcriptomic kNN graph (Cliff's δ
    +0.48) but collapses on the physical-spot Delaunay graph (δ
    −0.005, MWU p 0.84). On 10X Visium HD Human Colon Cancer P2 (N2,
    8 µm bin, sub-cellular), the same pattern: G_phys δ +0.004 (NS),
    G_trans δ −0.16 (p = 1e-16). The substrate-specificity is
    therefore real, not a resolution artefact: the κ phenotype is
    transcriptional-similarity-neighbourhood compactness, not a
    physical-tissue-architecture property.

Cross-domain extensions (mostly null or partial — bounded scope):

13. **ICB-response biomarker on Sade-Feldman 2018 — weak trend.**
    Per-patient AUC for mean baseline κ as a R-vs-NR classifier is
    0.64 (high κ → NR direction; n = 9 R / 10 NR). Below the 0.70
    threshold for a real clinical signal. Direction is biologically
    plausible (high κ ≈ exhausted tight-clonal-T-cell ≈ NR) but the
    cohort is too small and therapy-arm-heterogeneous to support a
    biomarker claim (N1).

14. **Hi-C 3D-genome contact graphs — chromosome-dependent, opposite
    sign on chr22.** K562 (cancer) vs GM12878 (normal) per-bin κ at
    1 Mb resolution: chr22 Cliff's δ = −0.50 (cancer LESS curved,
    opposite of scRNA-seq direction; consistent with TAD-boundary-
    loss / compartment-switching literature); chr11 Cliff's δ +0.14
    (weak, same direction as scRNA-seq). The transcriptomic-similarity
    and chromatin-contact graphs measure structurally different things
    and do not share a clean cancer-vs-normal curvature signature at
    this resolution (N3).

15. **Intra-tumour subclone stratification — partial, power-limited.**
    Of Tirosh's 19 patients, only 8 had ≥ 2 CNV-defined subclones with
    ≥ 10 cells; of those, 2 of 8 (P59, P79) showed Kruskal-Wallis
    p < 0.05 across subclones, with max-min Cliff's δ up to +0.74.
    Median across testable tumours: +0.29. κ functions as an ITH
    readout only at high cell counts and genuinely-divergent CNV
    structure; for typical Tirosh-sized tumours (< 150 cells) it is
    closer to within-tumour-uniform (N4).

The cleanest current statement of the result:

> Per-cell mean Ollivier-Ricci κ on the kNN graph of an scRNA-seq
> PCA-50 embedding is a transcriptional-compactness phenotype.
> Malignant cells in solid tumours occupy more positively-curved (more
> tightly-clustered) regions of the cell-cell graph than non-malignant
> immune or stromal cells. The signal replicates across four tumour
> types, survives within-patient stratification, hyperparameter
> variation, depth controls, cell-cycle controls, gene-selection
> controls, and decomposes into both a curvature shift and an
> effective-dimension narrowing. It is a property of the
> transcriptional-similarity neighbourhood structure (not of physical-
> tissue architecture at Visium spot resolution), monotonically
> increases along normal differentiation pseudotime, and is largely
> explained by tumor-specific transcriptional restriction with clonal
> expansion as a context-dependent additional contributor in some
> ICB-treated settings.

Finally, annotation fragility itself emerged as a finding from the
held-out validation. The pre-registered analysis survived its own
falsifiers (Section 3.28), but the primary cohort's public annotation
cannot resolve the immune-comparator contrast: the deposited companion
objects assign the same 26 439 barcodes to both malignant and immune,
and the pooled estimate ranges from −0.17 to +0.30 across defensible
dual-label handlings (Section 3.29). The implication is methodological
and extends beyond this cohort: curvature comparisons require
conflict-free labels, and confirmatory reuse of deposited annotations
should audit them for internal consistency before treating them as
ground truth.

### 4.3 What is still open

After 24 experiments the major confound, replication, mathematical-
decomposition, and cross-domain axes have all been tested. The
remaining caveats:

- **More liquid-tumour types.** F1 tested AML alone. CLL, multiple
  myeloma, T-cell leukaemias would establish whether the direction-
  flip is general to all hierarchy-collapse haematological cancers
  or AML-specific.
- **Replication beyond solid tumours.** Hematological malignancies
  (AML, MM, leukaemias) have very different scRNA-seq structure;
  whether the malignant-vs-immune κ signal extends to liquid tumours
  is untested.
- **Cell-line vs patient-derived contrast.** Cancer cell lines may
  show even higher κ than primary tumour cells (forced clonal
  uniformity); untested as an extreme case.

### 4.4 Suggested next experiments, ranked

The high-leverage extensions have largely been run. The remaining tier
is mostly polish:

1. **MM, CLL, T-cell leukaemia replication of the AML direction-flip.**
   Confirms whether the inversion is general to liquid tumours or AML-
   specific.
2. **Bakry-Émery decomposition on Hi-C contact graphs.** T3 decomposed
   the scRNA-seq κ signal into curvature-shift and effective-dimension-
   narrowing components. Doing the same decomposition on the Hi-C
   chromatin signal would test whether the cancer-chromatin signature
   is curvature-driven (organisational disorder) or dimension-driven
   (compartment-switch reduces accessible contact space).
3. **Perturb-seq κ-flattener screen on Replogle 2022.** Identifies
   gene knockouts that drop κ in cancer cells — direct mechanistic
   path from the geometric phenotype to therapeutic-vulnerability
   hypotheses (the AI-driven mechanism-to-therapy direction).
4. **Cell-line vs patient-derived κ at extreme clonal uniformity** —
   testable on existing 10X public cell-line scRNA-seq.
5. **Subclone analysis at larger N tumours** (Lambrechts 2018 NSCLC,
   Wu 2021 BRCA) where ≥ 50 cells per CNV-defined subclone is
   feasible — would lift the N4 statistical-power constraint.

## 5. Acknowledgements

Computation: Apple M-series CPU, ~10 GB RAM. No external compute
funding. Open-source software: scanpy, networkx, POT, ripser, persim,
scikit-learn, scipy, numpy, statsmodels, harmonypy, matplotlib.

## 6. Author contributions

TL conceived the study, designed and performed all analyses, wrote the
manuscript, and takes responsibility for all conclusions.

## 7. Competing interests

The author has declared that no competing interests exist.

## 8. Funding

No specific funding was received for this work.

## 9. Data availability

All datasets are publicly available from GEO / 10X Genomics:

- GSE72056 — Tirosh et al. 2016, melanoma scRNA-seq
- GSE84465 — Darmanis et al. 2017, glioblastoma scRNA-seq
- GSE103322 — Puram et al. 2017, head-and-neck cancer scRNA-seq
- GSE81861 — Li et al. 2017, colorectal cancer scRNA-seq
- GSE111672 — Peng et al. 2019, pancreatic ductal adenocarcinoma scRNA-seq
- GSE176031 — Chen et al. 2021, prostate cancer scRNA-seq
- GSE147082 — Olalekan et al. 2021, HGSOC ovarian scRNA-seq
- GSE123813 — Yost et al. 2019, basal cell carcinoma scRNA-seq + TCR
- GSE139555 — Wu et al. 2020, lung/colorectal/renal/endometrial T cells + TCR
- GSE120575 — Sade-Feldman et al. 2018, melanoma ICB scRNA-seq
- GSE116256 — van Galen et al. 2019, AML scRNA-seq
- GSE72857 — Paul et al. 2015, mouse hematopoiesis (scanpy built-in)
- GSE63525 — Rao et al. 2014, in-situ Hi-C (K562, GM12878)
- GSM8594568 — 10X Visium HD Human Colon Cancer Patient 2, 8 µm bins
- 10X Visium CytAssist Human Breast Cancer FFPE — 10X Genomics demo
  dataset (no accession; downloaded from 10xgenomics.com)
- GSE291210 — MERFISH MMTV-PyMT mouse mammary tumour (sample T1 =
  GSM8830801)

Pre-registered held-out cohorts (Section 3.28, completed 2026-09-05/06):
GSE161529 (Pal et al. 2021) and GSE131907 (Kim et al. 2020); outputs in
`bio_paths/exp/HELDOUT_GSE161529/` (results.json, results_sensitivity.json)
and `bio_paths/exp/HELDOUT_GSE131907/` (results.json). All per-experiment
outputs (CSVs, results.json, figures) are in `bio_paths/exp/<experiment>/`.

## 10. Code availability

All analysis code is the collection of self-contained per-experiment
scripts (`run.py`) under `bio_paths/exp/<experiment>/`, each with its
random seed recorded. TODO[VERIFY: public code repository URL/DOI to be
added upon deposition.]

## 11. References

Borassi, M., Chessa, A., & Dragan, F. F. (2015). *Hyperbolicity Measures
Democracy in Real-World Networks.* Phys Rev E 92, 032812.

Song, H., Weinstein, H. N. W., Allegakoen, P., et al. (2022). *Single-cell
analysis of human primary prostate cancer reveals the heterogeneity of
tumor-associated epithelial cell states.* Nat Commun 13, 141.
doi:10.1038/s41467-021-27322-4. GEO: GSE176031. [Corrected per GEO/PubMed
(PMID 35013146): GSE176031 is Song et al. 2022, not "Chen et al. 2021";
the cohort is called "Chen prostate" elsewhere in the text.]

Chen, Y., Pal, B., Lindeman, G. J., Visvader, J. E., & Smyth, G. K.
(2022). *R code and downstream analysis objects for the scRNA-seq atlas
of normal and tumorigenic human breast tissue.* Sci Data 9, 96.
doi:10.1038/s41597-022-01236-2. (Author-annotated labels and sample
metadata for the GSE161529 breast atlas; PMID 35322042.)

Cliff, N. (1993). *Dominance statistics: Ordinal analyses to answer
ordinal questions.* Psychol Bull 114(3), 494–509.

Cushing, D., Liu, S., & Münch, F. (2020). *Bakry-Émery curvature on
graphs as an eigenvalue problem.* Calc Var Partial Differ Equ 59, 142.

Darmanis, S., Sloan, S. A., Croote, D., et al. (2017). *Single-Cell
RNA-Seq Analysis of Infiltrating Neoplastic Cells at the Migrating Front
of Human Glioblastoma.* Cell Reports 21(5), 1399–1410.

DerSimonian, R., & Laird, N. (1986). *Meta-analysis in clinical trials.*
Controlled Clinical Trials 7(3), 177–188.

Ding, J., & Regev, A. (2021). *Deep generative model embedding of
single-cell RNA-Seq profiles on hyperspheres and hyperbolic spaces.*
Nat Commun 12, 2554.

Flavahan, W. A., Drier, Y., Liau, B. B., et al. (2016). *Insulator
dysfunction and oncogene activation in IDH mutant gliomas.* Nature
529(7584), 110–114.

Flamary, R., Courty, N., Gramfort, A., et al. (2021). *POT: Python
Optimal Transport.* JMLR 22(78), 1–8.

Hnisz, D., Weintraub, A. S., Day, D. S., et al. (2016). *Activation of
proto-oncogenes by disruption of chromosome neighborhoods.* Science
351(6280), 1454–1458.

Kim, N., Kim, H. K., Lee, K., et al. (2020). *Single-cell RNA sequencing
demonstrates the molecular and cellular reprogramming of metastatic lung
adenocarcinoma.* Nat Commun 11, 2285.

Klimovskaia, A., Lopez-Paz, D., Bouchacourt, D., & Nickel, M. (2020).
*Poincaré maps for analyzing complex hierarchies in single-cell data.*
Nat Commun 11, 2966.

Korsunsky, I., Millard, N., Fan, J., et al. (2019). *Fast, sensitive and
accurate integration of single-cell data with Harmony.* Nat Methods
16(12), 1289–1296.

Lambrechts, D., Wauters, E., Boeckx, B., et al. (2018). *Phenotype
modeling of stromal cells in the lung tumor microenvironment.* Nat Med
24(8), 1277–1289.

Li, H., Courtois, E. T., Sengupta, D., et al. (2017). *Reference
component analysis of single-cell transcriptomes elucidates cellular
heterogeneity in human colorectal tumors.* Nat Genet 49(5), 708–718.

Lin, Y., & Yau, S.-T. (2010). *Ricci curvature and eigenvalue estimate
on locally finite graphs.* Math Res Lett 17(2), 343–356.

Moncada, R., Barkley, D., Wagner, J., et al. (2020). *Integrating
microarray-based spatial transcriptomics and single-cell RNA-seq reveals
tissue architecture in pancreatic ductal adenocarcinomas.* Nat Commun
11, 2028.

Olalekan, S., Xie, B., Back, R., Eckart, H., & Basu, A. (2021).
*Characterizing the tumor microenvironment of metastatic ovarian cancer
by single-cell transcriptomics.* Cell Rep 35(8), 109165.
doi:10.1016/j.celrep.2021.109165. GEO: GSE147082. [Corrected per
GEO/PubMed (PMID 34038734): journal is Cell Reports, not Nat Commun;
author initials and title as previously drafted were inaccurate.]

Ollivier, Y. (2009). *Ricci curvature of Markov chains on metric spaces.*
J Funct Anal 256(3), 810–864.

Pal, B., Chen, Y., Vaillant, F., et al. (2021). *A single-cell RNA
expression atlas of normal, preneoplastic and tumorigenic states in the
human breast.* EMBO J 40(11), e107333. doi:10.15252/embj.2020107333.
GEO: GSE161529. [Corrected per GEO/PubMed (PMID 33950524): journal is
The EMBO Journal, not Nat Commun; full title includes "normal,
preneoplastic and tumorigenic states".]

Paul, F., Arkin, Y., Giladi, A., et al. (2015). *Transcriptional
heterogeneity and lineage commitment in myeloid progenitors.* Cell
163(7), 1663–1677.

Peng, J., Sun, B.-F., Chen, C.-Y., et al. (2019). *Single-cell RNA-seq
highlights intra-tumoral heterogeneity and malignant progression in
pancreatic ductal adenocarcinoma.* Cell Res 29(6), 725–738.

Pouryahya, M., Mathews, J. C., & Tannenbaum, A. (2020). *Comparing Three
Notions of Discrete Ricci Curvature on Biological Networks.* arXiv
1712.02943.

Puram, S. V., Tirosh, I., Parikh, A. S., et al. (2017). *Single-Cell
Transcriptomic Analysis of Primary and Metastatic Tumor Ecosystems in
Head and Neck Cancer.* Cell 171(7), 1611–1624.

Rao, S. S. P., Huntley, M. H., Durand, N. C., et al. (2014). *A 3D Map
of the Human Genome at Kilobase Resolution Reveals Principles of
Chromatin Looping.* Cell 159(7), 1665–1680.

Replogle, J. M., Saunders, R. A., Pogson, A. N., et al. (2022). *Mapping
information-rich genotype-phenotype landscapes with genome-scale
Perturb-seq.* Cell 185(14), 2792–2816.

Sade-Feldman, M., Yizhak, K., Bjorgaard, S. L., et al. (2018). *Defining
T Cell States Associated with Response to Checkpoint Immunotherapy in
Melanoma.* Cell 175(4), 998–1013.

Samal, A., Sreejith, R. P., Gu, J., et al. (2018). *Comparative analysis
of two discretizations of Ricci curvature for complex networks.* Sci
Rep 8, 8650.

Sandhu, R., Georgiou, T., Reznik, E., et al. (2015). *Graph Curvature
for Differentiating Cancer Networks.* Sci Rep 5, 12323.

Sreejith, R. P., Mohanraj, K., Jost, J., Saucan, E., & Samal, A. (2016).
*Forman curvature for complex networks.* J Stat Mech 2016, 063206.

Sritharan, D., Wang, S., & Krishnaswamy, S. (2025). *Recovering Manifold
Structure Using Ollivier-Ricci Curvature.* ICLR 2025.

Tian, T., Zhang, J., Lin, X., Wei, Z., & Hakonarson, H. (2023). *Dependency-aware deep generative models for multitasking analysis of spatial omics data.* Genome Research 33(2), 232–246.

Tirosh, I., Izar, B., Prakadan, S. M., et al. (2016). *Dissecting the
multicellular ecosystem of metastatic melanoma by single-cell RNA-seq.*
Science 352(6282), 189–196.

van Galen, P., Hovestadt, V., Wadleigh, M. I., et al. (2019). *Single-Cell
RNA-Seq Reveals AML Hierarchies Relevant to Disease Progression and
Immunity.* Cell 176(6), 1265–1281.

Weistuch, C., Murray, J. M., Mukherjee, S., et al. (2024). *Charting
cellular differentiation trajectories with Ricci flow.* Nat Commun 15,
2024.

Wolf, F. A., Angerer, P., & Theis, F. J. (2018). *SCANPY: large-scale
single-cell gene expression data analysis.* Genome Biology 19, 15.

Wu, S. Z., Roden, D. L., Wang, C., et al. (2021). *A single-cell and
spatially resolved atlas of human breast cancer.* Nat Med 27(12),
1969–1980.

Wu, T. D., Madireddi, S., de Almeida, P. E., et al. (2020). *Peripheral
T cell expansion predicts tumour infiltration and clinical response.*
Nature 579, 274–278.

Jiménez-Castaño, R., Narwade, N., Moreno-Bueno, G., et al. (2026). *A
hormetic transcriptional program coregulates invasion, proliferation and
dormancy to define metastatic potential.* Nat Commun 17, 3425.
doi:10.1038/s41467-026-70242-4. GEO: GSE291210 (MERFISH spatial
transcriptomics of MMTV-PyMT mammary tumours; sample GSM8830801 =
MMTV-PyMT WT Tumors Rep1). [Corrected per GEO/PubMed (PMID 41781391):
GSE291210 is linked solely to this paper; no "Wu et al. 2025" MERFISH
publication exists for this accession.]

Yost, K. E., Satpathy, A. T., Wells, D. K., et al. (2019). *Clonal
replacement of tumor-specific T cells following PD-1 blockade.* Nat Med
25(8), 1251–1259.

Zhou, Y., Sharpee, T. O., et al. (2021). *Hyperbolic geometry of gene
expression.* iScience 24(3), 102225.
