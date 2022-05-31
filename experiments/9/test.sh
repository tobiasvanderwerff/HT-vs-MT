#!/bin/bash

#SBATCH --job-name='9_test'
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

exp_id=9
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

arch="microsoft/deberta-v3-large"
seed=1
checkpoint_dir=/home/s4314719/macocu_experiments/9/models/microsoft-deberta-v3-large_lr=5e-06_bsz=32_seed=1/checkpoint-1400

logfile="${root_dir}/test_google_seed=${seed}.out"

cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--load_model $checkpoint_dir \
--arch $arch \
--test \
--use_google_data \
&> $logfile
