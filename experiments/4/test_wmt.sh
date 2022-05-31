#!/bin/bash

#SBATCH --job-name='4_test'
#SBATCH --partition=gpushort
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1-4


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

exp_id=4
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

arch="microsoft/deberta-v3-large"
mt="deepl"
seed=3
test_set="wmt${SLURM_ARRAY_TASK_ID}"

checkpoint_dir=/home/s4314719/macocu_experiments/4/models/${mt}/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=${seed}/checkpoint-*
logfile="${root_dir}/train_${mt}_test_${test_set}_seed=${seed}.out"

cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--load_model $checkpoint_dir \
--arch $arch \
--test $test_set \
&> $logfile
