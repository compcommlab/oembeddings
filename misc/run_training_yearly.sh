#!/bin/bash

# Script for Training every year 10 times
# Learning Rate is set to 0.05, because higher learning rates lead to errors

min_count=6
window_size=100

# Do the entire thing 10 times
for iteration in {1..10}; do
# Iterate over all combinations of parameters.
for year in {2012..2022}; do
    training_data_file="data/${year}_data.txt"
    # Remove forward slashes from the training data file name.
    training_data_file_no_slashes=$(echo "${training_data_file}" | sed 's|/||g')
    # Generate the SLURM script.
    slurm_script_file="slurm_script_${training_data_file_no_slashes}_${min_count}_${window_size}.slurm"
    cat << EOF > ${slurm_script_file}
#!/bin/bash

#SBATCH --job-name=oembeddings_${iteration}_${training_data_file_no_slashes}_${min_count}_${window_size}
#SBATCH -N 1
#SBATCH --ntasks-per-core=2

#SBATCH --qos=zen3_0512
#SBATCH --partition=zen3_0512
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<paul.balluff@univie.ac.at> 

echo "start"

module load miniconda3
eval "\$(conda shell.bash hook)"
conda activate oenv

cd oembeddings

# now actually run the program
python3 03_train/01_train.py cbow ${training_data_file} --threads 128 --window_size ${window_size} --min_count ${min_count} --dimensions 300 --learning_rate 0.05 --epochs 1

echo "done."
EOF

    # Submit the SLURM script.
    sbatch ${slurm_script_file}
done
done
