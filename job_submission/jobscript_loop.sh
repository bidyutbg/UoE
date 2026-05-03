#!/bin/bash
#SBATCH --job-name="Compute_MSE_T"
#SBATCH --time=12:00:00
#SBATCH --account=sheat_mip
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --mem=512G
#SBATCH -o %j.out
#SBATCH -e %j.err

cd /home/users/bidyut/UoE/20260218_Monsoons
module load jaspy

SCRIPT=/home/users/bidyut/UoE/20260218_Monsoons/20260303_MSEsfc_Change_Q.py
TMPSCRIPT=/home/users/bidyut/UoE/20260218_Monsoons/tmp_run_Q.py

# Define experiment pairs (EXP1 EXP2)
PAIRS=(
    "G6sulfur HIST"
    "SSP245   HIST"
    "SSP585   HIST"
    "G6sulfur SSP245"
    "G6sulfur SSP585"
)

for PAIR in "${PAIRS[@]}"; do
    EXP1=$(echo $PAIR | awk '{print $1}')
    EXP2=$(echo $PAIR | awk '{print $2}')

    echo "Running: EXP1=${EXP1}  EXP2=${EXP2}"

    # Create temp script with substituted experiment names
    sed -e "s/EXP1 = \"minuend\"/EXP1 = \"${EXP1}\"/" \
        -e "s/EXP2 = \"subtrahend\"/EXP2 = \"${EXP2}\"/" \
        ${SCRIPT} > ${TMPSCRIPT}

    python ${TMPSCRIPT}

    echo "Finished: EXP1=${EXP1}  EXP2=${EXP2}"

    rm -f ${TMPSCRIPT}
done
