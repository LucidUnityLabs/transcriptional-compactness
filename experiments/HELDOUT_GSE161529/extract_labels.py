"""
HELDOUT_GSE161529 -- label extraction (PREREGISTRATION.md 2026-06-29, LOCKED).

Original-author annotation sources:
  - Counts + per-sample metadata: GEO GSE161529 (Pal et al. 2021, EMBO J 40:e107333).
  - Cell cluster identities: Chen et al. 2022 Sci Data 9:96 companion deposit
    (figshare 10.6084/m9.figshare.17058077) -- Seurat objects
    SeuratObject_{ERTotal,HER2,TNBC,ERTotalSub,HER2Sub,TNBCSub}.rds, meta.data
    (cell barcode, group=sample, seurat_clusters).
  - Malignant calls: Tables/InferCNV-Annotation.txt from the companion repo
    (github.com/yunshun/HumanBreast10X) -- inferCNV tumor column-blocks per
    (cluster, sample). Annotation cluster k == seurat_clusters k+1 (validated:
    annotation tumor sets {0,4,5,6}/{0,3,7}/{0,2} equal the repo code's
    epithelial/microenvironment split sets {1,5,6,7}/{1,4,8}/{1,3} shifted by +1,
    for ER/HER2/TNBC respectively, 3/3 groups).
  - Immune identity: authors' marker panel Signatures/ImmuneMarkers2.txt applied
    to the authors' Sub (microenvironment) clusters, anchored to author-pinned
    assignments: T cells = ERTotalSub {1,8}, TNBCSub {1,5}, HER2Sub {2,7}
    (companion code ER.R/TNBC.R/HER2.R and EV4 legend); ERTotalSub cluster 7 =
    cycling TAM (primary paper text); HER2Sub cluster 9 myeloid/luminal
    (primary paper Fig 7 legend).

Scope decision (recorded): TumLN and Male analyses excluded -- original-author
tumour/immune calls for those analyses exist only as figure annotations
(primary paper Fig 9A / EV5), not machine-readable labels.

Outputs (data/GSE161529/labels/):
  cells_primary.tsv.gz     label in {malignant, immune}
  cells_secondary.tsv.gz   label in {malignant, normal_epithelial}
  cluster_marker_scores.tsv
  cluster_sizes.tsv
  cluster_sizes_sub.tsv
"""

import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

warnings.filterwarnings("ignore")
import rdata

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
CO = DATA / "GSE161529"
TMP = CO / "figshare_tmp"
LAB = CO / "labels"
LAB.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

FIGSHARE = {
    "ERTotal": "https://ndownloader.figshare.com/files/31545107",
    "HER2": "https://ndownloader.figshare.com/files/31544843",
    "TNBC": "https://ndownloader.figshare.com/files/31546307",
    "ERTotalSub": "https://ndownloader.figshare.com/files/31545113",
    "HER2Sub": "https://ndownloader.figshare.com/files/31544858",
    "TNBCSub": "https://ndownloader.figshare.com/files/31546385",
}

# authors' InferCNV annotation (Tables/InferCNV-Annotation.txt);
# annotation cluster k -> seurat_clusters k+1
TUMOR_BLOCKS = {
    "ERTotal": {
        1: ["ER_0319", "ER_0001", "ER_0025", "ER_0032", "ER_0040_T", "ER_0042",
            "ER_0043_T", "ER_0125", "ER_0151", "ER_0163", "ER_0167_T",
            "ER_0360", "ER_0114_T3"],
        5: ["ER_0319", "ER_0001", "ER_0040_T", "ER_0042", "ER_0043_T",
            "ER_0125", "ER_0163", "ER_0167_T", "ER_0360"],
        6: ["ER_0001", "ER_0032", "ER_0043_T", "ER_0114_T3"],
        7: ["ER_0043_T", "ER_0114_T3"],
    },
    "HER2": {
        1: ["HER2_0308", "HER2_0031", "HER2_0069", "HER2_0161", "HER2_0176",
            "HER2_0337"],
        4: ["HER2_0308", "HER2_0031", "HER2_0161", "HER2_0176", "HER2_0337"],
        8: ["HER2_0308", "HER2_0161", "HER2_0176"],
    },
    "TNBC": {
        1: ["TN_0126", "TN_B1_0131", "TN_0135", "TN_B1_0177", "TN_B1_4031",
            "TN_0114_T2", "TN_B1_0554"],
        3: ["TN_0126", "TN_B1_0131", "TN_0135", "TN_B1_0177", "TN_B1_4031",
            "TN_B1_0554"],
    },
}

# epithelial compartments (repo code microenvironment exclusions)
EPI_CLUSTERS = {"ERTotal": {1, 5, 6, 7}, "HER2": {1, 4, 8}, "TNBC": {1, 3}}

PINNED_T = {"ERTotalSub": {1, 8}, "TNBCSub": {1, 5}, "HER2Sub": {2, 7}}
PINNED_EXTRA = {
    "ERTotalSub": {7: "Macro(cycling TAM, primary paper text)"},
    "HER2Sub": {9: "Macro(myeloid/luminal, primary paper Fig 7 legend)"},
}

MARKERS = {
    "BCell": ["CD19", "PTPRC", "CD40LG", "CD79A", "MS4A1"],
    "TCell": ["CD4", "CD8A", "CD8B", "FOXP3", "IL2RA", "SELL", "ICOS",
              "PDCD1", "CTLA4", "NRP1", "CCR10", "CLEC4C", "ID3"],
    "TCell2": ["CD74", "ICAM1", "IL7R", "CCR7", "CD69", "CD3E", "CD3G"],
    "NK": ["NCR1", "KLRB1", "NCAM1", "KLRD1", "XCL1", "NCR3", "CCR5", "GZMH",
           "SIGLEC7", "GNLY", "NKG7"],
    "DC": ["ITGAX", "ITGAE", "CD40", "LY75", "BST2", "LGALS3", "FCER1A",
           "CST3"],
    "Macro": ["ITGAM", "ITGAX", "CD68", "FCGR3A", "ADGRE1", "HLA-DPB1",
              "HLA-DMA", "HLA-DOA", "HLA-DOB", "HLA-DQA1", "HLA-DRA",
              "TMEM119", "VCAM1", "CD74", "CX3CR1"],
    "Endo": ["PECAM1", "CD34", "PROCR", "EPAS1"],
    "Mega": ["PF4", "PTCRA"],
    "Fibro": ["PDGFRA", "PDGFRB", "PDPN", "COL1A2", "COL3A1", "TIMP2",
              "TIMP3", "MMP2", "POSTN", "ACTN2", "SDC1", "S100A4", "ACTA2",
              "KIT", "CEBPB"],
    "Fibro2": ["DLK1", "ACTA1", "DES", "RGS5", "CSPG4", "MCAM", "FAP",
               "FABP4", "FN1", "COL1A1", "COL1A3", "COL6A3", "MMP23",
               "HSPA1A", "GADD45B"],
}
IMMUNE_LINEAGES = {"BCell", "TCell", "TCell2", "NK", "DC", "Macro"}


def patient_of(sample_comb):
    s = sample_comb
    for pre in ("mER_", "ER_", "HER2_", "TN_B1_", "TN_", "N_", "B1_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.split("_")[0]


def load_obj(path):
    parsed = rdata.parser.parse_file(str(path))
    conv = rdata.conversion.convert(parsed)
    md = getattr(conv, "meta.data").copy()
    md.index = pd.Index([str(x) for x in md.index])
    rna = getattr(conv, "assays", {}).get("RNA", None)
    return conv, md, rna


def dgC_to_csr(ns):
    i = np.asarray(ns.i, dtype=np.int32)
    p = np.asarray(ns.p, dtype=np.int64)
    x = np.asarray(ns.x, dtype=np.float64)
    dim = np.asarray(ns.Dim, dtype=np.int64)
    return csc_matrix((x, i, p), shape=(dim[0], dim[1])).tocsr()


def main():
    t0 = time.time()
    frames, sub_frames, cluster_counts, sub_counts, score_rows = (
        [], [], [], [], [])

    for obj_name in ["ERTotal", "HER2", "TNBC"]:
        path = TMP / f"SeuratObject_{obj_name}.rds"
        print(f"[{obj_name}] {path.name}", flush=True)
        conv, md, _rna = load_obj(path)
        cl = md["seurat_clusters"].astype(int).values
        grp = md["group"].astype(str).values
        tb = {(s, c): True
              for c, samples in TUMOR_BLOCKS[obj_name].items()
              for s in samples}
        is_mal = np.array([tb.get((g, c), False) for g, c in zip(grp, cl)])
        is_epi = np.array([c in EPI_CLUSTERS[obj_name] for c in cl])
        label = np.where(is_mal, "malignant",
                         np.where(is_epi, "normal_epithelial", "other"))
        df = pd.DataFrame({"barcode": md.index, "sample": grp,
                           "patient": [patient_of(g) for g in grp],
                           "object": obj_name, "cluster": cl, "label": label})
        frames.append(df)
        print(f"  cells={len(df)} malignant={int((label == 'malignant').sum())}"
              f" normal_epi={int((label == 'normal_epithelial').sum())}"
              f" samples={df['sample'].nunique()}", flush=True)
        del conv, md
        gc.collect()

    for obj_name in ["ERTotalSub", "HER2Sub", "TNBCSub"]:
        path = TMP / f"SeuratObject_{obj_name}.rds"
        print(f"[{obj_name}] {path.name}", flush=True)
        conv, md, rna = load_obj(path)
        cl = md["seurat_clusters"].astype(int).values
        grp = md["group"].astype(str).values

        M = dgC_to_csr(getattr(rna, "counts"))
        mf = getattr(rna, "meta.features")
        gene_names = [str(x) for x in mf.index]
        assert M.shape[0] == len(gene_names)
        print(f"  matrix {M.shape} nnz={M.nnz}", flush=True)

        lib = np.asarray(M.sum(axis=0)).ravel()
        lib[lib == 0] = 1.0
        Mc = M.multiply(1e4 / lib).tocsr()
        Mc.data = np.log1p(Mc.data)

        gidx = {g: i for i, g in enumerate(gene_names)}
        assignments = {}
        for c in sorted(pd.unique(cl)):
            sub = Mc[:, cl == c]
            row = {"object": obj_name, "cluster": int(c),
                   "n_cells": int(sub.shape[1])}
            means = {}
            for lin, genes in MARKERS.items():
                hit = [gidx[g] for g in genes if g in gidx]
                means[lin] = (float(np.mean(
                    np.asarray(sub[hit, :].mean(axis=1)).ravel()))
                    if hit else np.nan)
                row[f"score_{lin}"] = means[lin]
                row[f"n_markers_{lin}"] = len(hit)
            valid = {k: v for k, v in means.items() if not np.isnan(v)}
            ranked = sorted(valid.items(), key=lambda kv: -kv[1])
            top, topv = ranked[0]
            best_ni = max((v for k, v in valid.items()
                           if k not in IMMUNE_LINEAGES), default=-np.inf)
            immune_panel = (top in IMMUNE_LINEAGES
                            and topv - best_ni > 0.1)
            row["argmax_lineage"] = top
            row["immune_by_panel"] = bool(immune_panel)
            note = ""
            if c in PINNED_T.get(obj_name, set()):
                row["immune_final"], row["identity"] = True, \
                    f"TCell(pinned {sorted(PINNED_T[obj_name])})"
                note = "pinned"
            elif c in PINNED_EXTRA.get(obj_name, {}):
                row["immune_final"], row["identity"] = True, \
                    PINNED_EXTRA[obj_name][c]
                note = "pinned"
            else:
                row["immune_final"] = bool(immune_panel)
                row["identity"] = top if immune_panel else "non-immune"
                note = "panel"
            score_rows.append(row)
            if row["immune_final"]:
                assignments[c] = row["identity"]
            print(f"  cluster {c}: n={row['n_cells']} argmax={top} "
                  f"immune_final={row['immune_final']} ({note})", flush=True)

        is_imm = np.array([c in assignments for c in cl])
        df = pd.DataFrame({"barcode": md.index, "sample": grp,
                           "patient": [patient_of(g) for g in grp],
                           "object": obj_name, "cluster": cl,
                           "label": np.where(is_imm, "immune", "other")})
        sub_frames.append(df)
        print(f"  immune cells={int(is_imm.sum())} of {len(df)}", flush=True)
        del conv, md, Mc, M
        gc.collect()

    full = pd.concat(frames, ignore_index=True)
    suball = pd.concat(sub_frames, ignore_index=True)
    imm = suball[suball.label == "immune"]

    prim_mal = full[full.label == "malignant"]
    prim = pd.concat([prim_mal, imm], ignore_index=True)
    prim.to_csv(LAB / "cells_primary.tsv.gz", sep="\t", index=False)

    sec = full[full.label.isin(["malignant", "normal_epithelial"])]
    sec.to_csv(LAB / "cells_secondary.tsv.gz", sep="\t", index=False)

    pd.DataFrame(score_rows).to_csv(LAB / "cluster_marker_scores.tsv",
                                    sep="\t", index=False)
    full.groupby(["object", "cluster", "label"]).size().rename(
        "n").reset_index().to_csv(LAB / "cluster_sizes.tsv", sep="\t",
                                  index=False)
    suball.groupby(["object", "cluster", "label"]).size().rename(
        "n").reset_index().to_csv(LAB / "cluster_sizes_sub.tsv", sep="\t",
                                  index=False)

    print("\nPRIMARY universe:", prim.groupby("label").size().to_dict())
    q = (prim.groupby(["patient", "label"]).size().unstack(fill_value=0))
    print("patients >=10/10 (mal/immune) pre-sampling:",
          int(((q.get("malignant", 0) >= 10) & (q.get("immune", 0) >= 10))
              .sum()), "of", len(q))
    print("SECONDARY universe:", sec.groupby("label").size().to_dict())
    q2 = (sec.groupby(["patient", "label"]).size().unstack(fill_value=0))
    print("patients >=10/10 (mal/normal_epi) pre-sampling:",
          int(((q2.get("malignant", 0) >= 10)
               & (q2.get("normal_epithelial", 0) >= 10)).sum()), "of", len(q2))
    print(f"wallclock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
