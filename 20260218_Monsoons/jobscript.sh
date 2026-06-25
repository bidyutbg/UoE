#!/bin/bash
#SBATCH --job-name="LS"
#SBATCH --time=04:00:00
#SBATCH --account=sheat_mip
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --mem=256G
#SBATCH -o %j.out
#SBATCH -e %j.err

cd /home/users/bidyut/UoE/20260218_Monsoons
module load jaspy
python /home/users/bidyut/UoE/20260218_Monsoons/20260528_PR_Length_n_Strength.py
