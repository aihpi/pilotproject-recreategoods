#!/bin/bash
#SBATCH --job-name=merge-csv
#SBATCH --partition=aisc
#SBATCH --account=aisc
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

cd data/example_image_dataset
head -n 1 metadata_edit.shard000-of-*.csv > metadata_edit.csv
tail -qn +2 metadata_edit.shard*.csv >> metadata_edit.csv
echo "Merged CSV written to $(pwd)/metadata_edit.csv"