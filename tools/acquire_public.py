"""Reproducible acquisition of author-deposited GEO/figshare inputs.

URLs are discovered from GEO text records / the figshare article API, rather
than guessed sample ranges. Data and receipts stay in ignored data/. A
content lock can be reproduced with --lock config/data.lock.json.
"""
import argparse
import gzip
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from lib.acquisition import atomic_download
from lib.runner import atomic_json

SELECT = {
    "GSE72056": ["GSE72056_melanoma_single_cell_revised_v2.txt.gz"],
    "GSE84465": ["GSE84465_GBM_All_data.csv.gz"],
    "GSE103322": ["GSE103322_HNSCC_all_data.txt.gz"],
    "GSE111672": ["GSE111672_PDAC-A-indrop-filtered-expMat.txt.gz",
                  "GSE111672_PDAC-B-indrop-filtered-expMat.txt.gz"],
    "GSE176031": ["GSE176031_RAW.tar"],
    "GSE147082": ["GSE147082_RAW.tar"],
    "GSE131907": ["GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz",
                  "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
                  "GSE131907_Lung_Cancer_Feature_Summary.xlsx"],
    "GSE81861": ["GSE81861_CRC_tumor_all_cells_FPKM.csv.gz"],
    "GSE120575": ["GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
                  "GSE120575_patient_ID_single_cells.txt.gz"],
    "GSE123813": ["GSE123813_bcc_all_metadata.txt.gz", "GSE123813_bcc_scRNA_counts.txt.gz",
                  "GSE123813_bcc_tcr.txt.gz"],
    "GSE116256": ["GSE116256_RAW.tar"],
    "GSE139555": ["GSE139555_tcell_metadata.txt.gz"],
    "GSM4143657": ["GSM4143657_SAM24348188-lt2.barcodes.tsv.gz", "GSM4143657_SAM24348188-lt2.genes.tsv.gz", "GSM4143657_SAM24348188-lt2.matrix.mtx.gz"],
    "GSM4143660": ["GSM4143660_SAM24349906-ln3.barcodes.tsv.gz", "GSM4143660_SAM24349906-ln3.genes.tsv.gz", "GSM4143660_SAM24349906-ln3.matrix.mtx.gz"],
    "GSM8594568": ["GSM8594568_P2CRC_Metadata.parquet.gz", "GSM8594568_P2CRC_filtered_feature_bc_matrix.h5", "GSM8594568_P2CRC_scalefactors_json.json.gz", "GSM8594568_P2CRC_tissue_positions.parquet.gz"],
    "GSM8830801": ["GSM8830801_T1_cell_by_gene.csv.gz", "GSM8830801_T1_cell_metadata.csv.gz"],
}
ALIASES = {"GSE72056_melanoma_single_cell_revised_v2.txt.gz": "GSE72056_melanoma.txt",
           "GSE84465_GBM_All_data.csv.gz": "GSE84465_GBM.csv",
           "GSE103322_HNSCC_all_data.txt.gz": "GSE103322_HNSCC.txt",
           "GSE81861_CRC_tumor_all_cells_FPKM.csv.gz": "GSE81861_CRC_tumor_FPKM.csv"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan(accessions):
    entries = []
    for acc in accessions:
        if acc == "breast-bootstrap":
            commit = "65cc0fe3fa1a72af8cfff0d3ba32c7545091cc4f"
            source = f"https://github.com/yunshun/HumanBreast10X/tree/{commit}"
            files = [f"RCode/{n}.R" for n in ["BRCA1Tum", "BulkRNAseq", "ER", "HER2", "Male", "NormBRCA1", "NormEpi", "NormTotal", "PairedER", "QC", "TNBC", "TumLN", "fig_label"]] + ["Tables/InferCNV-Annotation.txt", "Signatures/ImmuneMarkers2.txt"]
            entries.extend({"url": f"https://raw.githubusercontent.com/yunshun/HumanBreast10X/{commit}/{name}", "path": "GSE161529/author_sources/" + name, "source": source} for name in files)
            entries.extend([{ "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161529/soft/GSE161529_family.soft.gz", "path": "GSE161529_family.soft.gz", "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161529"}, {"url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161529/suppl/GSE161529_features.tsv.gz", "path": "GSE161529/samples/GSE161529_features.tsv.gz", "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161529"}])
            continue
        if acc == "visium10x":
            base = "https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer"
            entries.extend({"url": base + suffix, "path": "visium/" + name, "source": "https://github.com/10XGenomics/janesick_nature_comms_2023_companion"} for suffix, name in [("_filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"), ("_spatial.tar.gz", "spatial.tar.gz")])
            continue
        if acc == "breast-rds":
            api = "https://api.figshare.com/v2/articles/17058077"
            article = json.loads(urllib.request.urlopen(api, timeout=60).read())
            required = {f"SeuratObject_{s}.rds" for s in
                        ["ERTotal", "HER2", "TNBC", "ERTotalSub", "HER2Sub", "TNBCSub"]}
            files = [f for f in article["files"] if f["name"] in required]
            if {f["name"] for f in files} != required:
                raise ValueError("figshare required object set changed")
            entries.extend({"url": f["download_url"], "path": "GSE161529/figshare_tmp/" + f["name"],
                            "source": api, "expected_bytes": f["size"],
                            "publisher_md5": f["computed_md5"]} for f in files)
            continue
        source = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=self&form=text&view=full"
        raw = urllib.request.urlopen(source, timeout=60).read()
        urls = [l.split(" = ", 1)[1].replace("ftp://", "https://")
                for l in raw.decode().splitlines() if l.startswith("!Series_supplementary_file = ")
                or l.startswith("!Sample_supplementary_file")]
        for name in SELECT[acc]:
            matches = [u for u in urls if u.rsplit("/", 1)[-1] == name]
            if len(matches) != 1:
                raise ValueError(f"{acc}/{name}: not exactly one author-deposited URL")
            subdir = {"GSE131907": "GSE131907_kim_nsclc/", "GSE123813": "E7_clonality/",
                      "GSE120575": "", "GSE139555": "T2_wu_clonality/",
                      "GSM4143657": "T2_wu_clonality/", "GSM4143660": "T2_wu_clonality/",
                      "GSM8594568": "visium_hd/", "GSM8830801": "merfish/"}.get(acc, "sources/")
            entries.append({"url": matches[0], "path": subdir + name, "source": source})
        if acc == "GSE84465":
            entries.append({"url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84465/matrix/GSE84465_series_matrix.txt.gz",
                            "path": "sources/GSE84465_series_matrix.txt.gz", "source": source})
    return entries


def materialize(data, entry):
    path = data / entry["path"]
    name = path.name
    if name == "GSE161529_family.soft.gz" and path.parent == data:
        import shutil
        (data / "GSE161529").mkdir(exist_ok=True)
        shutil.copyfile(path, data / "GSE161529" / name)
    if entry["path"] == "visium/spatial.tar.gz":
        with tarfile.open(path) as tar:
            tar.extractall(data / "visium", filter="data")
    if name in ("GSM8830801_T1_cell_by_gene.csv.gz", "GSM8830801_T1_cell_metadata.csv.gz"):
        import shutil
        alias = "T1_cbg.csv.gz" if "by_gene" in name else "T1_meta.csv.gz"
        shutil.copyfile(path, path.parent / alias)
    if name.startswith("GSE111672_PDAC-"):
        import shutil
        shutil.copyfile(path, data / name)
    if name.startswith("GSM8594568_") and name.endswith(".gz"):
        with gzip.open(path, "rb") as src, open(path.with_suffix(""), "wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
    if name in ALIASES or name == "GSE84465_series_matrix.txt.gz":
        out = data / ALIASES.get(name, "GSE84465_meta.txt")
        with gzip.open(path, "rb") as src, open(out, "wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
    if name in ("GSE176031_RAW.tar", "GSE147082_RAW.tar", "GSE116256_RAW.tar"):
        folder = {"GSE176031_RAW.tar": "GSE176031_chen", "GSE147082_RAW.tar": "GSE147082", "GSE116256_RAW.tar": "GSE116256_galen"}[name]
        with tarfile.open(path) as tar:
            tar.extractall(data / folder, filter="data")
        if folder == "GSE147082":
            for gz in sorted((data / folder).glob("*.csv.gz")):
                with gzip.open(gz, "rb") as src, open(gz.with_suffix(""), "wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("accessions", nargs="*")
    p.add_argument("--lock", type=Path)
    args = p.parse_args()
    unknown = set(args.accessions) - (set(SELECT) | {"breast-rds", "breast-bootstrap", "visium10x"})
    if unknown:
        p.error(f'unknown source selections: {sorted(unknown)}')
    if not args.lock and not args.accessions:
        p.error('provide source selections or --lock')
    data = ROOT / "data"
    entries = json.loads(args.lock.read_text())["files"] if args.lock else plan(args.accessions)
    receipts = []
    receipt_path = data / ("acquisition_receipts_" + "_".join(args.accessions or ["locked"]) + ".json")
    for entry in entries:
        path = data / entry["path"]
        expected = entry.get("sha256")
        if path.exists() and (not expected or sha(path) == expected):
            receipt = {"cached": True}
        else:
            print("Downloading", entry["path"], flush=True)
            receipt = atomic_download(entry["url"], path, sha256=expected,
                                      max_bytes=8 << 30, timeout_s=7200)
        if entry.get("expected_bytes") and path.stat().st_size != entry["expected_bytes"]:
            raise ValueError(f"size mismatch {path}")
        if entry.get("publisher_md5"):
            h = hashlib.md5(usedforsecurity=False)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != entry["publisher_md5"]:
                raise ValueError(f"publisher content checksum mismatch {path}")
        receipt.update(entry)
        receipt.update(sha256=sha(path), bytes=path.stat().st_size)
        materialize(data, entry)
        receipts.append(receipt)
        atomic_json(receipt_path, {"status": "in_progress", "files": receipts})
        print("Verified", entry["path"], receipt["sha256"], flush=True)
    atomic_json(receipt_path, {"status": "ok", "files": receipts})


if __name__ == "__main__":
    main()
