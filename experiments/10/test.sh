#!/bin/bash

#SBATCH --job-name='10_test'
#SBATCH --partition=gpushort
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

exp_id=10
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

# arch="allenai/longformer-base-4096"
# max_length=10000000000000000

arch="microsoft/deberta-v3-large"
max_length=512

mt="google"
seed=${SLURM_ARRAY_TASK_ID}

if [ $arch == "allenai/longformer-base-4096" ]; then
    checkpoint_dir="/data/pg-macocu/MT_vs_HT/experiments/${exp_id}/models/${mt}/allenai-longformer-base-4096_lr=1e-05_bsz=32_epochs=*_seed=${seed}/checkpoint-*"
else  # deberta-v3
    checkpoint_dir="/data/pg-macocu/MT_vs_HT/experiments/${exp_id}/models/${mt}/microsoft-deberta-v3-large_lr=1e-05_bsz=32_epochs=*_seed=${seed}/checkpoint-*"
fi

mt_flag=""
if [ $mt == "google" ]; then
    mt_flag="--use_google_data"
fi

log_model_name=$(echo $arch | sed 's/\//-/g')
logfile="${root_dir}/test_${log_model_name}_${mt}_seed=${seed}.out"

cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--load_model $checkpoint_dir \
--arch $arch \
--max_length $max_length \
--test $mt \
$mt_flag \
&> $logfile
