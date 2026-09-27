#!/bin/bash
#SBATCH --job-name=xgb_scale_fulltest
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/instacart/logs/xgb_scale_fulltest_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/xgb_scale_fulltest_%j.err

module purge
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

PROJECT=/scratch/$USER/instacart
cd $PROJECT
mkdir -p logs results/tables

echo "=== XGBoost full-trained prediction on full test $(date) ==="

python src/predict_full_test.py \
  --model xgboost \
  --model-path models/xgboost/full_backup/xgboost_best_full.pkl \
  --threshold 0.5 \
  --source-tag final_full \
  --out-tag final_fulltrain_fulltest \
  --batch-size 100000

echo "=== XGBoost scale comparison $(date) ==="

python src/predict_full_test.py \
  --compare-xgboost-scale \
  --base-tag final_fulltest \
  --scaled-tag final_fulltrain_fulltest \
  --n-boot 1000 \
  --scale-out-prefix xgboost_scale

echo "=== Done $(date) ==="