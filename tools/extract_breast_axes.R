# Read author-deposited Seurat objects without rerunning clustering.
# Base R/Matrix retain factor levels and named RNA axes; no Seurat install
# or heuristic reconstruction is needed. Called one object at a time.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop("usage: Rscript tools/extract_breast_axes.R OBJECT.rds OUTPUT_DIR")
library(Matrix)
input <- args[1]
output <- args[2]
dir.create(output, recursive=TRUE, showWarnings=FALSE)
object <- readRDS(input)
md <- slot(object, "meta.data")
rna <- slot(object, "assays")[["RNA"]]
counts <- slot(rna, "counts")
features <- slot(rna, "meta.features")
validObject(counts)
if (anyDuplicated(rownames(counts)) || anyDuplicated(colnames(counts)) ||
    anyDuplicated(rownames(md)) || anyDuplicated(rownames(features))) stop("duplicate RNA axis identity")
if (!setequal(colnames(counts), rownames(md))) stop("RNA/metadata cell-axis mismatch")
if (!setequal(rownames(counts), rownames(features))) stop("RNA/feature gene-axis mismatch")
if (any(!is.finite(counts@x)) || any(counts@x < 0)) stop("invalid RNA counts")
counts <- counts[, match(rownames(md), colnames(counts)), drop=FALSE]
name <- sub("\\.rds$", "", sub("^SeuratObject_", "", basename(input)))
clusters <- md$seurat_clusters
# The companion R scripts use as.integer(seurat_clusters), which converts
# FACTORS to level positions, not their printed numeric level labels.
if (!is.factor(clusters)) stop("author cluster coding requires a reviewed non-factor mapping")
cells <- data.frame(barcode=rownames(md), sample=as.character(md$group),
                    object=name, cluster_stored=as.character(clusters),
                    cluster=as.integer(clusters), stringsAsFactors=FALSE)
write.table(cells, file=file.path(output, paste0(name, "_cells.tsv")), sep="\t", row.names=FALSE, quote=FALSE)
crosswalk <- unique(cells[, c("sample", "object", "cluster_stored", "cluster")])
crosswalk <- crosswalk[order(crosswalk$sample, crosswalk$cluster), ]
write.table(crosswalk, file=file.path(output, paste0(name, "_cluster_crosswalk.tsv")), sep="\t", row.names=FALSE, quote=FALSE)
if (grepl("Sub$", name)) {
    # Panel means of log1p-CP10K use the complete raw library denominator.
    panel_file <- file.path(dirname(output), "author_sources", "Signatures", "ImmuneMarkers2.txt")
    panel_table <- read.delim(panel_file, check.names=FALSE)
    panels <- split(panel_table$Signatures, panel_table$CellType)
    library_sizes <- colSums(counts)
    if (any(library_sizes <= 0)) stop("empty raw library")
    scores <- list()
    for (panel in names(panels)) {
        markers <- unique(as.character(panels[[panel]]))
        markers <- markers[!is.na(markers) & nzchar(markers)]
        hit <- intersect(markers, rownames(counts))
        if (!length(hit)) {
            message(paste("Excluded panel with no measured genes:", panel))
            next
        }
        normalized <- as.matrix(counts[hit, , drop=FALSE])
        normalized <- log1p(sweep(normalized, 2, library_sizes, "/") * 1e4)
        cell_score <- colMeans(normalized)
        means <- tapply(cell_score, cells$cluster, mean)
        scores[[panel]] <- data.frame(object=name, cluster=as.integer(names(means)),
                                     panel=panel, score=as.numeric(means), n_markers=length(hit))
    }
    write.table(do.call(rbind, scores), file=file.path(output, paste0(name, "_marker_scores.tsv")),
                sep="\t", row.names=FALSE, quote=FALSE)
}
cat(name, nrow(cells), "cells; named RNA axes verified;", nrow(crosswalk), "sample/cluster mappings\n")
