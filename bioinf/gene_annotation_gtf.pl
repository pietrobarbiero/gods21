#!/usr/bin/perl
open (IN,"<","Homo_sapiens.GRCh38.104.chr.gtf")||die "can't open it $!\n";
open (OUT,">","annotation_gene_Homo_sapiens.GRCh38.txt")||die "can't open it $!\n";


print OUT "Chr\tStart\tEnd\tStrand\tGene_id\tGene_name\n";   #11 columns

while(defined($line = <IN>))
{
	chomp $line;
	@arr = split(/\t/,$line);
             if ($arr[2] eq "gene") 
             {
             
             	$arr[8] =~ /gene_id\s+"(\S+?)".*gene_name\s+"(\S+?)";/;
             	print OUT "Chr$arr[0]\t$arr[3]\t$arr[4]\t$arr[6]\t$1\t$2\n";
             
             }
}




close(IN);
close(OUT);
