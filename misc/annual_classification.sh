#!/bin/bash

#SBATCH --job-name=oembeddings_annual_classification
#SBATCH -N 1
#SBATCH --ntasks-per-core=2

#SBATCH --qos=zen3_0512
#SBATCH --partition=zen3_0512
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<paul.balluff@univie.ac.at> 

echo "start"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate oembeddings

cd oembeddings

echo "Label Classification"
python3 04_eval/03_label_classification.py --glob "tmp_models/20*/*.json" --threads 128

echo "done."
