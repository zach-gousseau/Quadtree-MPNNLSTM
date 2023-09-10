#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --array=7
#SBATCH --gres=gpu:v100l
#SBATCH --mem=187G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-12:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_exp_glorys.py -m $SLURM_ARRAY_TASK_ID -e 32