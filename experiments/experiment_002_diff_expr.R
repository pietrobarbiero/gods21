# Title     : TODO
# Objective : TODO
# Created by: pietr
# Created on: 05/09/2021
# References:
# http://www.bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html

library("tximport")
library("readr")
library("tximportData")
library("data.table")
library("TxDb.Hsapiens.UCSC.hg19.knownGene")
library("DESeq2")

# Load data
dir <- "./data"
phenotype <- read.csv(file.path(dir,"gods21_phenotype.csv"), header=TRUE)
phenotype <- phenotype[phenotype$Project == "DownS",]
colnames(phenotype)[1] <- "SampleName"
conditionID <- c()
for (sid in phenotype$SampleID){
  if (grepl('CTRL', sid))
    conditionID <- c(conditionID, 'CTRL')
  else
    conditionID <- c(conditionID, 'DS')
}
phenotype$condition <- factor(conditionID)
samples <- phenotype$SampleName

files <- file.path(paste('./data/Sample_', samples, sep=''), 'quant', paste('Sample_', samples, '_quant', sep=''), "quant.sf")
# dir <- system.file("extdata", package = "tximportData")
# samples <- read.table(file.path(dir, "samples.txt"), header = TRUE)
# files <- file.path(dir, "salmon", samples$run, "quant.sf.gz")
# files <- file.path("/data/", samples$run, "quant.sf.gz")
# names(files) <- paste0("sample", 1:6)
all(file.exists(files))


# creating table
# if (file.exists("./data/reference/gencode.v38.annotation.gtf.gz") == FALSE)
txdb <- makeTxDbFromGFF(file="./data/reference/gencode.v38.annotation.gtf.gz")
saveDb(x=txdb, file = "./data/reference/gencode.v38.annotation.TxDb")
k <- keys(txdb, keytype = "TXNAME")
tx2gene <- select(txdb, k, "GENEID", "TXNAME")
head(tx2gene)

txi <- tximport(files, type = "salmon", tx2gene = tx2gene, ignoreAfterBar = TRUE)
names(txi)

dds <- DESeqDataSetFromTximport(txi, colData = phenotype, design = ~ condition)

# filtering
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep,]
# set reference level
dds$condition <- relevel(dds$condition, ref = "CTRL")

# differential expression analysis
dds <- DESeq(dds)
res <- results(dds)
res

# Shrinkage of effect size (LFC estimates) is useful for visualization and ranking of genes
resultsNames(dds)
resLFC <- lfcShrink(dds, coef="condition_DS_vs_CTRL", type="apeglm")
resLFC

# We can order our results table by the smallest p value
resOrdered <- res[order(res$pvalue),]

# How many adjusted p-values were less than 0.1?
sum(res$padj < 0.1, na.rm=TRUE)

# plotMA shows the log2 fold changes attributable to a given variable over the mean of normalized counts for all the samples
png("./experiments/results/diff_expr/plotma.png")
plotMA(res, ylim=c(-2,2))
dev.off()
png("./experiments/results/diff_expr/plotma_adj.png")
plotMA(resLFC, ylim=c(-2,2))
dev.off()

# It can also be useful to examine the counts of reads for a single gene across the groups.
# A simple function for making this plot is plotCounts, which normalizes counts by the estimated size factors (or normalization factors if these were used) and adds a pseudocount of 1/2 to allow for log scale plotting
# Here we specify the gene which had the smallest p value from the results table created above
png("./experiments/results/diff_expr/plot_counts.png")
plotCounts(dds, gene=which.min(res$padj), intgroup="condition")
dev.off()

resSig <- subset(resOrdered, padj < 0.1)
resSig

write.csv(as.data.frame(resOrdered),
          file="./experiments/results/diff_expr/ordered_all_results.csv")

write.csv(as.data.frame(resSig),
          file="./experiments/results/diff_expr/ordered_filtered_results.csv")