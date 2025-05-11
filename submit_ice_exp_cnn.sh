#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-1:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/experiment_output_cnn_nogpu.out
module load StdEnv/2023
module load gcc/12.3
module load eccodes/2.31.0
module load openmpi/4.1.5
module load hdf5/1.14.2
module load netcdf/4.9.2
source /home/zgoussea/geospatial/bin/activate
mpirun -np 1 python ice_exp_cnnlstm.py --month 5 --exp 2