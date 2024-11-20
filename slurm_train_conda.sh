#!/bin/bash
#SBATCH --output=outputs/slurm_outputs/%u-slurm-%j.out
#SBATCH --error=outputs/slurm_outputs/%u-slurm-%j.out

# for example: $ sbatch -p gpu_4_a100 --gres=gpu:3 --time=05:00:00 --mem=80gb slurm_train_conda.sh --tiny=True --batch=64 --epoch=1
eval "$(conda shell.bash hook)"
conda activate t5dec
echo "Training with parameters ${*}"
echo "Job ID: $SLURM_JOB_ID , $SLURM_JOB_NAME"
python train.py ${*} > "debug_log.txt" # hydra.job.id=$SLURM_JOB_ID
conda deactivate