#!/bin/bash
#SBATCH --account=def-ka3scott
#SBATCH --gres=gpu:v100l
#SBATCH --mem=187G
#SBATCH --time=00-12:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test_apr27_6.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_exp.py -m 4 -e 6