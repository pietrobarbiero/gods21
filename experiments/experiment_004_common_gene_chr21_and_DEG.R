library(data.table)
data<-fread("./annotation_gene_Homo_sapiens.GRCh38.txt",header = T,data.table = F)
data<- data[-which(data$Gene_name==""),]
write.table(data,"All_Chr.txt",quote = F,sep = "\t",col.names = T,row.names = F)


data_21<-data[which(data$Chr == "Chr21"),]
write.table(data_inter_21,"All_21.txt",quote = F,sep = "\t",col.names = T,row.names = F)


DEGs_gene<-fread("./DEGs_gene.txt",header = T,data.table = F)

intersect_gene<-intersect(data$Gene_name,DEGs_gene$Gene)


data_inter<-data[match(intersect_gene,data$Gene_name),]
data_inter_21<-data_inter[which(data_inter$Chr == "Chr21"),]

write.table(data_inter_21,"DEG_21.txt",quote = F,sep = "\t",col.names = T,row.names = F)
