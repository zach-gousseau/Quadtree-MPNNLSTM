#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --array=1,2,3,4,5,6,7,8,9,10,11,12
#SBATCH --mem=187G
#SBATCH --cpus-per-task=1
#SBATCH --time=00-6:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test_34_2.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/scrap2.py -m $SLURM_ARRAY_TASK_ID -e 34