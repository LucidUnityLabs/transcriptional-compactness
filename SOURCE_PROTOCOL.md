# Public source and corrected protocol

Historical scripts, results, figures, and `manuscript/MANUSCRIPT.md` are retained as a baseline. They used restricted endpoint-neighborhood paths and cannot establish corrected full-graph Ollivier–Ricci results. Corrected results use TC-1 and separate output paths.

## Source identities

GEO processed expression and metadata are acquired from the supplementary URLs in the actual [GEO records](https://www.ncbi.nlm.nih.gov/geo/). `tools/acquire_public.py` records observed SHA256, byte count, URL, and accession; `config/data.lock.json` separates downloaded source bytes from derived artifacts. Observed checksums establish reproducible byte identity, not scientific correctness or undocumented identity to a historical download.

The breast objects are the deposited [Figshare article 17058077](https://doi.org/10.6084/m9.figshare.17058077). Publisher MD5 and size are checked in addition to observed SHA256. Companion author code and annotations are pinned to [HumanBreast10X commit65cc0fe3fa1a72af8cfff0d3ba32c7545091cc4f](https://github.com/yunshun/HumanBreast10X/tree/65cc0fe3fa1a72af8cfff0d3ba32c7545091cc4f).

Native R validates the complete named RNA cell/gene axes against metadata/features. The deposited `seurat_clusters` is a factor: the author's `as.integer(factor)` means level position, while printed cluster labels are recorded separately. Corrected extraction validates complete Total/Sub barcode inclusion and epithelial exclusion. Marker panels with no measured genes are unavailable and excluded from ranking; they are never assigned fabricated measurements. Immune labels remain pipeline-derived marker calls, with the explicit panel/margin/override rule recorded.

Breast specimen identity uses exact, injective supplementary mappings. The single explicit alias `ER_0040_T` → GSM4909307 is supported by the author's primary tumor sample list and GEO patient0040/Total metadata; the separately deposited lymph-node specimen GSM4909308 is distinct. The alias record lives in `config/GSE161529_sample_aliases.json`; general suffix matching remains strict. GEO donor IDs retain leading zeros. Raw feature IDs identify matrix rows; duplicate display symbols do not imply duplicate feature axes.

The public [GSE131907 workbook](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907) maps58 specimens to44 donors. `tools/lung_donor_crosswalk.py` checks the committed crosswalk directly against FeatureSummary columnsB/C, rows4–61, using OOXML identities. Its donor grouping replaces sample-name proxies; shared graph dependence remains a limitation.

GSE111672 is the Moncada PDAC deposit. The legacy artifact key `Peng_PDAC` is retained for comparison and is not source attribution. The selected 10X Visium input is [CytAssist_FFPE_Human_Breast_Cancer, Space Ranger2.0.0](https://github.com/10XGenomics/janesick_nature_comms_2023_companion); its identity to undocumented historical bytes is not asserted.

## Protocol changes and inference limits

- Full-graph shortest paths, checked POT transport plans, positive metric weights, and measured-edge incidence replace the defective restricted metric.
- Float64, explicit full SVD, imported-code/environment/normalized-input/identity-bound caches, strict integer readers, and complete identity checks are part of the method.
- Lung and breast select bounded qualifying donor identities before allocating dense counts. Gene detection retains the complete eligible label universe; normalization retains the full raw-library denominator before gene filtering. Compressed matrices stream to EOF without raw expansion on disk.
- Author breast `RNA@counts` contains integer counts but follows upstream gene filtering, official-symbol selection, and deduplication. It does not replace the original complete raw matrix or its denominator.
- Cells lacking an inferCNV malignant call establish an exploratory epithelial comparator, not affirmative normal-luminal identity. The preregistered luminal endpoint requires a source-backed positive per-cell annotation table and an exact donor/specimen join. Public normal and paired objects represent distinct cell universes and cannot silently provide those annotations for this tumor-cell universe.
- Corrected exploratory adapters preserve each historical assay scale and design. Dependent cell-level tests are descriptive. Class-balanced, cohort-transductive cell splits do not establish independent-patient deployment performance; split standard deviations are not donor confidence intervals.
- Patient-group effect summaries and row bootstrap uncertainty are conditional on the shared graph and inferred labels. Source-backed donor identities alone do not make graph-derived cells independent or establish causal effects.

## Reproduction

Use the tested interpreter/platform lock corresponding to the actual host; no untested Linux equivalence is claimed. The exact source-built POT wheel under `requirements/wheels/` matches the SHA256 in the macOS arm64 core lock. Download other locked wheels from their distributions and verify hashes. Set numerical thread controls before interpreter startup, as in `.github/workflows/tests.yml`.

```sh
python tools/acquire_public.py --lock config/data.lock.json
python tools/lung_donor_crosswalk.py
python tools/extract_breast_labels_corrected.py
python experiments/HELDOUT_GSE161529/download_corrected.py --plan --update
python experiments/HELDOUT_GSE161529/download_corrected.py --update
python tools/data_lock.py --verify
```

For a new breast bootstrap before a final lock exists, acquire `breast-bootstrap breast-rds`, extract labels, then plan/download matrices. The downloader requires all remaining compressed bytes plus3.5GiB host reserve and rechecks individual transfers. A refused acquisition is a real capacity failure, never a successful scientific result. Dedicated corrected drivers and `tools/rerun_extensions.py EXPERIMENT` write separately identified results; downstream figures must read validated corrected artifacts.

Yost response curation uses the publisher's [Supplementary Tables 1–3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-019-0522-3/MediaObjects/41591_2019_522_MOESM2_ESM.xlsx), SHA256 `1e55fd00e2c0bce62653b3909187005994696c6b7609fce0fe7642b3c65bcce9`. `tools/yost_response_crosswalk.py` reads SuppTable1 Patient/Tumor Type/Response columns and selects the 11 BCC rows. It preserves author-reported Yes/No, including su002's clinical-response footnote, rather than inventing a RECIST threshold. SCC lesion su010-S remains distinct from BCC su010.

For range-read Hi-C sources, the corrected adapter preserves the actual consumed observed/KR contact-matrix slices and their SHA256, request coordinates, source URL, HTTP object identity headers and hic-straw version. These receipts identify the consumed matrix bytes; they do not assert a SHA256 for the entire remote `.hic` asset. Cell-line contrasts remain exploratory and confounded by lineage and culture conditions.

The acquired GSE161529 raw files expose a real same-cell coverage gap for `ER_0001` (GSM4909296): only 59 of 5,462 primary labels and 58 of 5,303 secondary labels occur among the 7,375 deposited raw barcodes. The remaining 26 specimens have complete label/barcode coverage. `config/GSE161529_source_exclusions.json` binds the exact raw barcode and full label hashes and explicitly excludes the entire ER_0001 specimen, including coincidental barcode overlaps. Its exclusion is reported before sampling and changes the analyzed cohort; the resulting analysis has partial source coverage and cannot establish complete preregistered replication. Restoring this specimen requires author-corrected full raw counts with exact gene/cell axes, or an authenticated same-cell crosswalk. The author-filtered RDS RNA assay does not supply the original full raw-library denominator.

## Reviewed bounded breast evidence and successor source

The completed primary/excluded run is conditional numerical evidence for a source-covered subset; its original full cohort endpoint remains incomplete. The per-donor contrast is a Cliff rank contrast of sampled incident-edge mean curvature, conditional on the fitted shared graph, selected cells and edges, inferred immune labels, and whole-source exclusion. High heterogeneity, shared graph dependence, and unfinished calibration/repeats prevent interpreting its nominal directional test as independent-donor causal evidence. The original accepted tables, cache, method receipt, and source freeze remain unchanged; `config/GSE161529_completed_primary_excluded.json` binds their identities. New successor source bytes require new review and frozen repeats before scientific acceptance.

Harmony sensitivities attenuate melanoma and GBM and reverse HNSCC. These descriptive pipeline responses must accompany any positive discovery/held-out narrative. Dependent Welch/MWU tests and transductive cell splits do not establish independent-donor significance or clinical deployment. Remaining discovery/lung/LABEL repeats, breast arms/variants, independent controls, authenticated gene-position reference, closed controls environment, required remote ARM64 CI, figures, and manuscript reconciliation are open. F4's completed first pass is superseded pending an edge-budget and consumed-slice-bound rerun; T5's failed artifact is historical diagnostic evidence.

Six `axes_corrected/*_source_v2.json` receipts bind all emitted cells, cluster crosswalks, and Sub marker-score TSVs. `labels/LABELS_MANIFEST_CORRECTED_V2.json` binds these complete receipts and the retained labels. Cached extraction rejects missing/tampered outputs before label work. Existing reviewed axes may be bound only against a preserved byte-identical reviewed repository snapshot using `--bind-reviewed-axes`; that operation records prior native-R evidence and does not claim a fresh RDS parse. Fresh extraction uses a separate `labels_v2/` destination and requires a separately reviewed coverage binding before its adoption. Legacy receipts are never upgraded in place.

Breast publication now stages each calculation, per-donor table, result JSON, used cache pair, and provenance in `experiments/HELDOUT_GSE161529/attempts/<result-stem>/<attempt-id>/`. Only after complete-field reproduction and artifact validation does a single `results*_corrected.accepted.json` pointer select the complete immutable set. Flat historical files stay unchanged. Python readers use `lib.publication.read_set` / `load_result`; table readers use `resolve_table_result` to resolve one consistent snapshot. `tools/verify_outputs.py` accepts the existing flat-path CLI and resolves an accepted pointer automatically. Failed calculation, cache, or acceptance gates retain attempt diagnostics without publishing. An error at or after pointer rename is explicitly reported as uncertain; inspect the pointer before retrying. Never infer acceptance from an orphan attempt directory or fall back to stale flat files after a pointer validation failure.

A future separately authorized numerical slot must budget at least the observed 4,446,408,952-byte peak footprint plus explicit headroom. The earlier estimated three-GiB ceiling was exceeded and is not a tested bound. Source repair and small regression tests do not authorize a new cohort calculation.
