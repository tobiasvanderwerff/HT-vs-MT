#!/bin/bash

#SBATCH --job-name='11_tst_mj'
#SBATCH --partition=gpushort
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1-3

# Document-level classification using majority voting with a sentence-level model.


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

exp_id=11
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

# Hyper-parameters
arch="microsoft/deberta-v3-large"
mt="google"
split="dev"
seed=${SLURM_ARRAY_TASK_ID}

checkpoint_dir="/data/pg-macocu/MT_vs_HT/experiments/11/models/${mt}/allenai-longformer-base-4096_lr=1e-05_bsz=4_seed=${seed}/checkpoint-*"

if [ $mt == "google" ]; then
    flags="--use_google_data"
else
    flags=""
fi

if [ $split == "dev" ]; then
    split_flag="--eval"
else
    split_flag="--test $mt"
fi

logfile="${root_dir}/${split}_majority_${mt}_seed=${seed}.out"


cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--load_model $checkpoint_dir \
--arch $arch \
--use_majority_classification \
$split_flag \
$flags \
&> $logfile
