#!/bin/bash
#SBATCH --account=def-dclausi
#SBATCH --gres=gpu:v100l
#SBATCH --array=1,2,3,4,5,6,7,8,9,10,11,12
#SBATCH --mem=187G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-12:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test_era5_lstm_6conv.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_exp_era5.py -m $SLURM_ARRAY_TASK_ID -e 11