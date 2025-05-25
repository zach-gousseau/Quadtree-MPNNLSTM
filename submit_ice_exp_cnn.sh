#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --gres=gpu:v100l
#SBATCH --mem=167G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-24:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/experiment_output_cnn_%A_%a.out
#SBATCH --array=1-12

module load StdEnv/2023
module load gcc/12.3
module load eccodes/2.31.0
module load openmpi/4.1.5
module load hdf5/1.14.2
module load netcdf/4.9.2
source /home/zgoussea/geospatial/bin/activate

mpirun -np 1 python cnn_ice_dataset.py --month ${SLURM_ARRAY_TASK_ID}
