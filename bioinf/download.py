import glob
import os
import wget
import subprocess


def main():
    # Reference
    _DATA_DIR = '/local/scratch-3/pb737/gods21_pilot/all_data/'
    reference_dir = os.path.join(_DATA_DIR, 'reference')
    reference_gtf_url = 'http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz'
    reference_fasta_url = 'http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.transcripts.fa.gz'
    reference_gtf_dest = os.path.join(reference_dir, 'gencode.v38.annotation.gtf.gz')
    reference_fasta_dest = os.path.join(reference_dir, 'gencode.v38.transcripts.fa.gz')

    os.makedirs(reference_dir, exist_ok=True)
    if not os.path.exists(reference_gtf_dest):
        wget.download(reference_gtf_url, reference_gtf_dest)

    if not os.path.exists(reference_fasta_dest):
        wget.download(reference_fasta_url, reference_fasta_dest)

    # Tools

    # Salmon
    _TOOL_DIR = '/home/pb737/Programs/'
    _DOWNLOAD_DIR = '/home/pb737/Downloads/'
    _SALMON_FILE = 'salmon-1.5.2_linux_x86_64.tar.gz'
    salmon_url = f'https://github.com/COMBINE-lab/salmon/releases/download/v1.5.2/{_SALMON_FILE}'
    salmon_dest = os.path.join(_DOWNLOAD_DIR, _SALMON_FILE)
    salmon_install_dir = os.path.join(_TOOL_DIR, 'salmon-1.5.2_linux_x86_64')
    os.makedirs(salmon_install_dir, exist_ok=True)
    if not os.path.exists(salmon_dest):
        wget.download(salmon_url, salmon_dest)

    subprocess.run(['tar', '-xzvf', salmon_dest, '-C', f'{_TOOL_DIR}'])
    subprocess.run(['rm', salmon_dest])


if __name__ == "__main__":
    main()
