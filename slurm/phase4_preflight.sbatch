#!/bin/bash
#SBATCH --job-name=phase4_preflight
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/instacart/logs/phase4_preflight_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/phase4_preflight_%j.err

PROJECT=/scratch/$USER/instacart
cd $PROJECT
mkdir -p logs

source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

python -u src/phase4_preflight.py --warn-threshold-only