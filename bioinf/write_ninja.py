import glob
import json
import os
import subprocess


def main():
    _DATA_DIR = '/local/scratch-3/pb737/gods21_pilot/all_data/'

    sample_dir_list = glob.glob(os.path.join(_DATA_DIR, 'Sample_*'), recursive=True)
    sample_dict = {sample_dir.split('/')[-1]: {'dir': sample_dir} for sample_dir in sample_dir_list}
    for sample in sample_dict.keys():
        sample_dict[sample]['R1'] = sample_dict[sample]['dir'] + '/' + sample + '_R1.fastq.gz'
        sample_dict[sample]['R2'] = sample_dict[sample]['dir'] + '/' + sample + '_R2.fastq.gz'
        sample_dict[sample]['L001_R1'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L001_R1_001.fastq.gz'))[0]
        sample_dict[sample]['L002_R1'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L002_R1_001.fastq.gz'))[0]
        sample_dict[sample]['L001_R2'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L001_R2_001.fastq.gz'))[0]
        sample_dict[sample]['L002_R2'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L002_R2_001.fastq.gz'))[0]

    build_ninja = [
        f'# THIS FILE IS PROPRIETARY AND NOT INTENDED FOR SHARING',
        f'rule unify-lanes',
        f"    command = echo $in | xargs -l bash -c 'cat $$0 $$1 >$out'",
        f'    description = Unify reads',
        f'',
        # f'',
        # f'rule quality-check',
        # f'    command = fastp -i $datadir/{"${in}"}_R1.fastq.gz -I $datadir/{"${in}"}_R2.fastq.gz -o $datadir/{"${in}"}_R1_fastp.fastq.gz -O $datadir/{"${in}"}_R2_fastp.fastq.gz',
        # f'    description = Unify reads',
        # f'',
        # f'',
    ]

    # analyze some substances only
    for sample in sample_dict.keys():
        build_ninja.append(f'build {sample_dict[sample]["R1"]}: unify-lanes {sample_dict[sample]["L001_R1"]} {sample_dict[sample]["L002_R1"]}')

    # create ninja build
    build_ninja.append("\n")
    build_ninja = "\n".join(build_ninja)
    text_file = open("/home/pb737/Projects/gods21/bioinf/build.ninja", "w")
    text_file.write(build_ninja)
    text_file.close()

    # python3 write_ninja.py
    # ninja -j 2 -k 10
    # ninja -j 3 -k 10
    # ninja -j 5 -k 10
    # ninja -j 30 -k 20


if __name__ == "__main__":
    main()
