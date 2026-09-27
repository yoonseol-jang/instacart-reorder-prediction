#!/bin/bash
#SBATCH --job-name=predict_full_test_all
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/instacart/logs/predict_full_test_all_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/predict_full_test_all_%j.err

module purge
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

PROJECT=/scratch/$USER/instacart
cd $PROJECT
mkdir -p logs results/tables

echo "=== predict_full_test_all job=$SLURM_JOB_ID node=$(hostname) $(date) ==="

for MODEL in logreg rf adaboost xgboost; do
  echo "=== Predicting full test for $MODEL $(date) ==="
  python src/predict_full_test.py \
      --model $MODEL \
      --source-tag final \
      --out-tag final_fulltest \
      --batch-size 100000
done

echo "=== Comparing final_fulltest predictions $(date) ==="

python src/compare_final_models.py \
  --tag final_fulltest \
  --n-boot 1000

echo "=== Done $(date) ==="