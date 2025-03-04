#!/bin/bash
#SBATCH --account=def-foutsekh
#SBATCH --time=1-00:00
#SBATCH --mail-user=a.majdinasab@hotmail.com
#SBATCH --mail-type=ALL
source /scratch/f/foutsekh/vamaj/sum_tf_env/bin/activate
cd /scratch/f/foutsekh/vamaj/sum_tf
python train.py