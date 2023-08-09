#!/bin/bash
#SBATCH --account=def-dclausi
#SBATCH --gres=gpu:v100l
#SBATCH --mem=187G
#SBATCH --time=00-03:00            # time (DD-HH:MM)
#SBATCH --output=/home/zgoussea/scratch/logs/output_ice_test_20years_small_era5_lstm.out
source /home/zgoussea/geospatial/bin/activate
python /home/zgoussea/projects/def-ka3scott/zgoussea/Quadtree-MPNNLSTM/ice_exp_era5.py -m 6 -e 11