#!/bin/bash
#SBATCH --job-name="VCorr"
#SBATCH --time=24:00:00
#SBATCH --account=impose
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --mem=512G
#SBATCH -o %j.out
#SBATCH -e %j.err

cd /home/users/bidyut/UoE/20260218_Monsoons
module load jaspy
python /home/users/bidyut/UoE/20260218_Monsoons/20260414_Verical_Correlation.py
