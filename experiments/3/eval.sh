#!/bin/bash

#SBATCH --job-name='3_eval'
#SBATCH --partition=gpushort
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --array=212-214
#SBATCH --output=eval_seed_%a.out

EXP_ID=3
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf.py \
--root_dir $ROOT_DIR \
--arch bert \
--eval \
--load_model $ROOT_DIR/models/bert_outputs_50_epoch_seed_${SLURM_ARRAY_TASK_ID}_lr_1e-6/best_model
