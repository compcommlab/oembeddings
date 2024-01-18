#!/bin/bash

# Script for evaluating label classification task (04_eval/03_label_classification.py)
# for each model family on a separate node

# Iterate over all combinations of parameters.
for dir in $DATA/tmp_models/*; do 
    echo $dir; 
    # Generate the SLURM script.
    slurm_script_file="slurm_script_label_classification_${dir##*/}.slurm"
    cat << EOF > ${slurm_script_file}
#!/bin/bash

#SBATCH --job-name=oembeddings_clsf_${dir##*/}
#SBATCH -N 1
#SBATCH --ntasks-per-core=2

#SBATCH --qos=zen3_0512
#SBATCH --partition=zen3_0512
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<paul.balluff@univie.ac.at> 


module load miniconda3
eval "\$(conda shell.bash hook)"
conda activate oembeddings

cd oembeddings

python3 -u 04_eval/03_label_classification.py --threads 128 --modelfamily ${dir}

EOF

    # Submit the SLURM script.
    sbatch ${slurm_script_file}
done