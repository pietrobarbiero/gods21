import glob
import json
import os
import subprocess
import pandas as pd


def main():
    _DATA_DIR = '/local/scratch-3/pb737/gods21_pilot/all_data/'
    dataset_dir = os.path.join(_DATA_DIR, 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    sample_dir_list = glob.glob(os.path.join(_DATA_DIR, 'Sample_*'), recursive=True)
    sample_dict = {sample_dir.split('/')[-1]: {'dir': sample_dir} for sample_dir in sample_dir_list}
    for sample in sample_dict.keys():
        fastp_dir = os.path.join(sample_dict[sample]['dir'], 'fastp')
        os.makedirs(fastp_dir, exist_ok=True)
        quant_dir = os.path.join(sample_dict[sample]['dir'], 'quant')
        os.makedirs(quant_dir, exist_ok=True)
        sample_dict[sample]['quant'] = os.path.join(quant_dir, sample + '_quant')
        sample_dict[sample]['fastp_R1'] = os.path.join(fastp_dir, sample + '_R1_fastp.fastq.gz')
        sample_dict[sample]['fastp_R2'] = os.path.join(fastp_dir, sample + '_R2_fastp.fastq.gz')
        sample_dict[sample]['fastp_html'] = os.path.join(fastp_dir, sample + '_fastp.html')
        sample_dict[sample]['R1'] = sample_dict[sample]['dir'] + '/' + sample + '_R1.fastq.gz'
        sample_dict[sample]['R2'] = sample_dict[sample]['dir'] + '/' + sample + '_R2.fastq.gz'
        sample_dict[sample]['L001_R1'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L001_R1_001.fastq.gz'))[0]
        sample_dict[sample]['L002_R1'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L002_R1_001.fastq.gz'))[0]
        sample_dict[sample]['L001_R2'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L001_R2_001.fastq.gz'))[0]
        sample_dict[sample]['L002_R2'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L002_R2_001.fastq.gz'))[0]

    reference_dir = os.path.join(_DATA_DIR, 'reference')
    reference_fasta = glob.glob(os.path.join(reference_dir, '*.fa.gz'), recursive=True)[0]
    reference_gtf = glob.glob(os.path.join(reference_dir, '*.gtf.gz'), recursive=True)[0]
    reference_dict = {
        'name': reference_fasta.split('/')[-1].split('.fa')[0],
        'dir': reference_dir,
        'fasta': reference_fasta,
        'gtf': reference_gtf,
        'index': reference_dir + '/' + reference_fasta.split('/')[-1].split('.fa')[0]+'_index',
    }

    sample_data_list = []
    for sample in sample_dict.keys():
        # https://salmon.readthedocs.io/en/latest/file_formats.html
        quant_file = os.path.join(sample_dict[sample]['quant'], 'quant.sf')
        if os.path.exists(quant_file):
            quant_df = pd.read_csv(quant_file, delimiter='\t')
            quant_df.index = quant_df['Name']
            quant_df = quant_df['TPM']
            quant_df.name = sample
            sample_data_list.append(quant_df)

    if len(sample_data_list) > 0:
        data_df = pd.concat(sample_data_list, axis=1)
        data_df.to_csv(os.path.join(dataset_dir, 'gods21_data_salmon.csv'))


if __name__ == "__main__":
    main()
