#!/bin/bash
#SBATCH --account=def-dclausi
#SBATCH --gres=gpu:v100l
#SBATCH --mem=187G
#SBATCH --time=00-12:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test_20years_small.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_exp.py -m 6 -e 11