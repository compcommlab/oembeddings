#!/bin/bash

# Script for calculating across correlations (04_eval/01_eval_cosine_across.py)
# for each model year (e.g., 2012 vs 2015) combination on a separate node

# Iterate over all combinations of parameters.

dir_path=$DATA/tmp_models

# Use ls and grep to list directories contain the year prefix & store them in an array
directories=($(ls -d "$dir_path"/* | grep "20"))

# Loop through the directories and generate unique pairs
for ((i=0; i<${#directories[@]}; i++)); do
    for ((j=i+1; j<${#directories[@]}; j++)); do
        dir1="${directories[i]}"
        dir2="${directories[j]}"
        echo "$dir1 & $dir2"
        # Generate the SLURM script.
        slurm_script_file="slurm_script_acroos_${dir1##*/}_${dir2##*/}.slurm"
        cat << EOF > ${slurm_script_file}
#!/bin/bash

#SBATCH --job-name=oembeddings_across_${dir1##*/}_${dir2##*/}
#SBATCH -N 1
#SBATCH --ntasks-per-core=2

#SBATCH --qos=zen3_0512
#SBATCH --partition=zen3_0512
#SBATCH --mail-type=ALL

module load miniconda3
eval "\$(conda shell.bash hook)"
conda activate oembeddings

cd oembeddings

python3 -u 04_eval/01_eval_cosine_across.py --threads 128 --model_a ${dir1} --model_b ${dir2}

EOF

    # Submit the SLURM script.
    sbatch ${slurm_script_file}
    done
done