#!/bin/bash

# Get the list of training data files.
training_data_files=(data/training_data.txt data/training_data_lower.txt)

# Get the list of min count values.
min_count_values=(5 10 50 100)

# Get the list of window size values.
window_size_values=(5 6 12 24)

# Iterate over all combinations of parameters.
for training_data_file in "${training_data_files[@]}"; do
  for min_count in "${min_count_values[@]}"; do
    for window_size in "${window_size_values[@]}"; do
      # Remove forward slashes from the training data file name.
      training_data_file_no_slashes=$(echo "${training_data_file}" | sed 's|/||g')
      # Generate the SLURM script.
      slurm_script_file="slurm_script_${training_data_file_no_slashes}_${min_count}_${window_size}.slurm"
      cat << EOF > ${slurm_script_file}
#!/bin/bash

#SBATCH --job-name=oembeddings_test
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
python3 tests/test_train.py cbow ${training_data_file} --threads 128 --window_size ${window_size} --min_count ${min_count} --dimensions 300

echo "done."
EOF

      # Submit the SLURM script.
      sbatch ${slurm_script_file}
    done
  done
done