#!/bin/bash

#SBATCH --job-name='9_train'
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=3


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

exp_id=9
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

arch="microsoft/deberta-v3-large"
mt="google"
learning_rate=5e-06
bsz=32
num_epochs=5
weight_decay=0
max_grad_norm=1
warmup_steps=200
seed=${SLURM_ARRAY_TASK_ID}

if [ $mt == "google" ]; then
    flags="--use_google_data"
else
    flags=""
fi

log_model_name=$(echo $arch | sed 's/\//-/g')
# Make sure the logdir specified below corresponds to the directory defined in the
# main() function of the `classifier_trf_hf.py` script!
logdir="${root_dir}/models/${mt}/${log_model_name}_lr=${learning_rate}_bsz=${bsz}_seed=${seed}/"
logfile="${logdir}/train.out"
mkdir -p $logdir

# DeepL
# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/${log_model_name}_lr=1e-05_bsz=32_seed=1/checkpoint-1800
# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/${log_model_name}_lr=1e-05_bsz=32_seed=2/checkpoint-2800
# checkpoint_dir=/data/pg-macocu/MT_vs_HT/experiments/4/models/${log_model_name}_lr=1e-05_bsz=32_seed=3/checkpoint-3000
# Google
# checkpoint_dir=/home/s4314719/macocu_experiments/4/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=1/checkpoint-2800
# checkpoint_dir=/home/s4314719/macocu_experiments/4/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=2/checkpoint-1400
checkpoint_dir=/home/s4314719/macocu_experiments/4/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_seed=3/checkpoint-1000

# Copy source code
mkdir -p $logdir/src
cp $HOME/MaCoCu/student_project_mt_ht/classifier_trf_hf.py $logdir/src

# Copy this script
cp $(realpath $0) $logdir


cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--load_model $checkpoint_dir \
--arch $arch \
--batch_size $bsz \
--learning_rate $learning_rate \
--num_epochs $num_epochs \
--warmup_steps $warmup_steps \
--max_grad_norm $max_grad_norm \
--weight_decay $weight_decay \
&> $logfile
