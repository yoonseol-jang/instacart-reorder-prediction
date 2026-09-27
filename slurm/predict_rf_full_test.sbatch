#!/bin/bash
#SBATCH --job-name=predict_rf_full
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/instacart/logs/predict_rf_full_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/predict_rf_full_%j.err

PROJECT=/scratch/$USER/instacart
cd $PROJECT
mkdir -p logs

source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

python -u src/predict_rf_full_test.py