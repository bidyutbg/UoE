#!/bin/bash
#SBATCH --job-name="GMS"
#SBATCH --time=24:00:00
#SBATCH --account=nerc_srm
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --mem=512G
#SBATCH -o %j.out
#SBATCH -e %j.err

cd /home/users/bidyut/UoE/20260225_AUXILIARY
module load jaspy
python /home/users/bidyut/UoE/20260225_AUXILIARY/Sort_SpHum_and_Omega.py
