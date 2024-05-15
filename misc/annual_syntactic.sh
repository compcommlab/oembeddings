#!/bin/bash

#SBATCH --job-name=oembeddings_annual_syntactic
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

echo "Syntactic and Semantic Tests"
python3 04_eval/02_fasttext_semantic_syntactic.py --glob "tmp_models/20*/*.json"

echo "done."
