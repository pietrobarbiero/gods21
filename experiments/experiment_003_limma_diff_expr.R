library(tidyverse)
library(stringr)
library(limma)
data<-read.csv("./gods21_data_salmon.csv")
data$Name<-str_split(data$Name,"\\|",simplify = T)[,6]

# Transcripts corresponding to the same gene are averaged 
# to measure the expression of the gene. 
# Total of 59,059 gene pairs
data<-avereps(data[,-1],ID=data$Name)
dim(data)

# Extract the corresponding DS samples and phenotype
phenotype<-read.csv("./gods21_phenotype.csv")
phenotype_DS<-phenotype[7:12,]

data_DS<- data[, match(paste0("Sample_",phenotype_DS$SampleName),colnames(data) )]   
colnames(data_DS)<-phenotype_DS$SampleID

# Filter Low-expressed data 
data_DS_new=data_DS[which(apply(data_DS,1,function(x){return( length(which(x == 0)))}) < ncol(data_DS)*0.2) ,] 

data_DS_new<-log2(data_DS_new)

PCA_new <- function(expr, ntop = 500, group, show_name = F){
  library(ggplot2)
  library(ggrepel)
  object <- expr
  rv <- genefilter::rowVars(object)
  select <- order(rv, decreasing = TRUE)[seq_len(min(ntop, length(rv)))]
  pca <- prcomp(t(object[select, ]))
  percentVar <- pca$sdev^2/sum(pca$sdev^2)
  d <- data.frame(PC1 = pca$x[, 1], 
                  PC2 = pca$x[, 2], 
                  group = group, 
                  name = colnames(object))
  attr(d, "percentVar") <- percentVar[1:2]
  if (show_name) {
    ggplot(data = d, aes_string(x = "PC1", y = "PC2", color = "group")) + 
      geom_point(size = 2) +
      xlab(paste0("PC1: ", round(percentVar[1] * 100), "% variance")) + 
      ylab(paste0("PC2: ", round(percentVar[2] * 100), "% variance")) +
      geom_text_repel(aes(label = name),
                      size = 3,
                      segment.color = "black",
                      show.legend = FALSE )
  } else {
    ggplot(data = d, aes_string(x = "PC1", y = "PC2",color = "group")) + 
      geom_point(size = 2) +
      xlab(paste0("PC1: ", round(percentVar[1] * 100), "% variance")) + 
      ylab(paste0("PC2: ", round(percentVar[2] * 100), "% variance"))
  }
}

# PCA with the gene expression values
PCA_new(data_DS_new, 
        ntop = 500,
        group = c(rep("DS",3),rep("Ctr",3)),
        show_name = T)

# Differential expression analysis
colnames(data_DS_new)<-c(rep("DS",3),rep("DS_CTRL",3))
group_list<-factor(c(rep("DS",3),rep("DS_CTRL",3)) ,
                   level = c("DS_CTRL","DS"))  #control, treat


design <- model.matrix(~0+factor(group_list))
colnames(design)=levels(factor(group_list))
rownames(design)=colnames(data_DS_new)

contrast.matrix<-makeContrasts("DS-DS_CTRL",levels = design)   #treat, control


# Fit a linear model
fit <- lmFit(data_DS_new,design)
# Calculate the difference according to the contrast model
fit2 <- contrasts.fit(fit, contrast.matrix) 
# Bayesian test
fit2 <- eBayes(fit2) 
# Generate results
DEG<-topTable(fit2, coef=1, n=Inf) %>% na.omit()
# Filter with pvalue, get all DEGs
expr<-data_DS_new[match(rownames(DEG),rownames(data_DS_new)),]

DEG_exp<-as.data.frame(cbind(DEG,data_DS_new))
write.table(DEG_exp,"DEG_exp.txt",sep = "\t",quote = F)

All_diff_padj<-na.omit(DEG_exp[DEG_exp$P.Value <=0.05,])
# down

DEGs.res_down<-All_diff_padj[All_diff_padj$logFC< -1,]
write.table(DEGs.res_down,"DEGs.res_down_padj005.txt",sep = "\t",quote = F)

# up
DEGs.res_up<-All_diff_padj[All_diff_padj$logFC > 1,]
write.table(DEGs.res_up,"DEGs.res_up_padj005.txt",sep = "\t",quote = F)

down<-data.frame(Gene = rownames(DEGs.res_down),Regulate = "Down")

up<-data.frame(Gene = rownames(DEGs.res_up),Regulate = "Up")
# All DEGs
DEGs_gene<-rbind(down,up)
write.table(DEGs_gene,"DEGs_gene.txt",sep = "\t",quote = F,row.names = F)











