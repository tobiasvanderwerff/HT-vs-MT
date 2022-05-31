#!/bin/bash

#SBATCH --job-name='4_test'
#SBATCH --partition=gpushort
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

exp_id=4
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

arch="microsoft/deberta-v3-large"
mt="google"
seed=3

# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/deepl/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=1/checkpoint-1200
# checkpoint_dir=/home/s4314719/macocu_experiments/4/models/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=1/checkpoint-1800
# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=2/checkpoint-2800
# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=3/checkpoint-3000

# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=1/checkpoint-2800
# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=2/checkpoint-1400
checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=3/checkpoint-1000

if [ $mt == "google" ]; then
    mt_flag="--use_google_data"
else
    mt_flag=""
fi

logfile="${root_dir}/test_${mt}_seed=${seed}.out"

cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--load_model $checkpoint_dir \
--arch $arch \
--test $mt \
$mt_flag \
&> $logfile
