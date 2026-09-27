#!/bin/bash
#SBATCH --job-name=compare_final_models
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/instacart/logs/compare_final_models_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/compare_final_models_%j.err

module purge
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

PROJECT=/scratch/$USER/instacart
cd $PROJECT
mkdir -p logs results/tables

echo "=== compare_final_models job=$SLURM_JOB_ID node=$(hostname) $(date) ==="

python src/compare_final_models.py \
  --tag final \
  --n-boot 1000

echo "=== Done $(date) ==="