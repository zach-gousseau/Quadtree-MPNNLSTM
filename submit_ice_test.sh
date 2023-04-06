#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --gres=gpu:v100l
#SBATCH --array=6
#SBATCH --mem=187G
#SBATCH --cpus-per-task=8
#SBATCH --time=00-12:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test_6.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_test.py -m $SLURM_ARRAY_TASK_ID