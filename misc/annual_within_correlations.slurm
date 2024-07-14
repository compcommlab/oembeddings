#!/bin/bash

#SBATCH --job-name=oembeddings_within_annual
#SBATCH -N 1
#SBATCH --ntasks-per-core=2

#SBATCH --qos=zen3_0512
#SBATCH --partition=zen3_0512
#SBATCH --mail-type=ALL

echo "start"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate oembeddings

cd oembeddings

echo "Cosine Similarity: Within"
python3 04_eval/01_eval_cosine_within.py --threads 128 --glob "tmp_models/20*"

echo "done."
