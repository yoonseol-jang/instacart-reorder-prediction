#!/bin/bash
#SBATCH --job-name=validate_phase4
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/instacart/logs/validate_phase4_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/validate_phase4_%j.err

module purge
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

PROJECT=/scratch/$USER/instacart
mkdir -p $PROJECT/logs

echo "=== validate_phase4  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="

python $PROJECT/src/validate_phase4_artifacts.py \
    --root $PROJECT \
    --tag final

echo "=== Done $(date) ==="
