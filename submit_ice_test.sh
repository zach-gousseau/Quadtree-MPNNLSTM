#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --array=1-12
#SBATCH --mem=187G
#SBATCH --time=00-12:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_test.py --month $SLURM_ARRAY_TASK_ID