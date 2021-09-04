import glob
import json
import os
import subprocess


def main():
    _DATA_DIR = '/local/scratch-3/pb737/gods21_pilot/all_data/'

    sample_dir_list = glob.glob(os.path.join(_DATA_DIR, 'Sample_*'), recursive=True)
    sample_dict = {sample_dir.split('/')[-1]: {'dir': sample_dir} for sample_dir in sample_dir_list}
    for sample in sample_dict.keys():
        fastp_dir = os.path.join(sample_dict[sample]['dir'], 'fastp')
        os.makedirs(fastp_dir, exist_ok=True)
        sample_dict[sample]['fastp_R1'] = os.path.join(fastp_dir, sample + '_R1_fastp.fastq.gz')
        sample_dict[sample]['fastp_R2'] = os.path.join(fastp_dir, sample + '_R2_fastp.fastq.gz')
        sample_dict[sample]['fastp_html'] = os.path.join(fastp_dir, sample + '_fastp.html')
        sample_dict[sample]['R1'] = sample_dict[sample]['dir'] + '/' + sample + '_R1.fastq.gz'
        sample_dict[sample]['R2'] = sample_dict[sample]['dir'] + '/' + sample + '_R2.fastq.gz'
        sample_dict[sample]['L001_R1'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L001_R1_001.fastq.gz'))[0]
        sample_dict[sample]['L002_R1'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L002_R1_001.fastq.gz'))[0]
        sample_dict[sample]['L001_R2'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L001_R2_001.fastq.gz'))[0]
        sample_dict[sample]['L002_R2'] = glob.glob(os.path.join(sample_dict[sample]['dir'], '*L002_R2_001.fastq.gz'))[0]

    build_ninja = [
        f"# Ninja build for bioinformatics pipeline in parallel!",
        f"",
        f"rule unify-lanes",
        f"    command = echo $in | xargs -l bash -c 'cat $$0 $$1 >$out'",
        f"    description = Unify lanes",
        f"",
        f"",
        f"rule quality-check",
        f"    command = echo $in $out | xargs -l bash -c 'fastp --html=$$4 -i $$0 -I $$1 -o $$2 -O $$3'",
        f"    description = Quality check",
        f"",
        f"",
    ]

    # analyze some substances only
    for sample in sample_dict.keys():
        build_ninja.append(f'build {sample_dict[sample]["R1"]}: unify-lanes {sample_dict[sample]["L001_R1"]} {sample_dict[sample]["L002_R1"]}')
        build_ninja.append(f'build {sample_dict[sample]["R2"]}: unify-lanes {sample_dict[sample]["L001_R2"]} {sample_dict[sample]["L002_R2"]}')
        build_ninja.append(f'build {sample_dict[sample]["fastp_R1"]} {sample_dict[sample]["fastp_R2"]} {sample_dict[sample]["fastp_html"]}: quality-check {sample_dict[sample]["R1"]} {sample_dict[sample]["R2"]}')

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
