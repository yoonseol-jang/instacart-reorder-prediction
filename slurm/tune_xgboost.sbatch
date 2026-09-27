#!/bin/bash
#SBATCH --job-name=tune_xgboost
#SBATCH --account=ds_ga_1003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/instacart/logs/tune_xgboost_%j.out
#SBATCH --error=/scratch/%u/instacart/logs/tune_xgboost_%j.err

module purge
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate instacart

PROJECT=/scratch/$USER/instacart
DATA=$PROJECT/data/processed
mkdir -p $PROJECT/logs

echo "=== tune_xgboost  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="

python $PROJECT/src/train_xgboost.py \
    --mode tune \
    --data-dir $DATA \
    --out-dir $PROJECT \
    --max-train-rows 300000 \
    --max-test-rows 75000 \
    --n-iter 15 \
    --n-jobs $SLURM_CPUS_PER_TASK \
    --run-tag tune

echo "=== Done $(date) ==="
