#!/bin/bash

#SBATCH --job-name='10_dbr_gt'
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1-5


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

exp_id=10
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

# Hyper-parameters
arch="microsoft/deberta-v3-large"
max_length=512
bsz=8
gradient_accumulation_steps=4

# arch="allenai/longformer-base-4096"
# bsz=4
# gradient_accumulation_steps=8

mt="google"
learning_rate=1e-05
num_epochs=30
weight_decay=0
max_grad_norm=1
warmup_steps=20
label_smoothing=0.0
dropout=0.1
use_fp16="yes"
seed=${SLURM_ARRAY_TASK_ID}

let "bsz_efctv = $bsz * $gradient_accumulation_steps"

flags=""
if [ $mt == "google" ]; then
    flags="--use_google_data"
fi

fp_flag=""
if [ $use_fp16 == "yes" ]; then
    fp_flag="--use_fp16"
fi

log_model_name=$(echo $arch | sed 's/\//-/g')
# Make sure the logdir specified below corresponds to the directory defined in the
# main() function of the `classifier_trf_hf.py` script!
logdir="${root_dir}/models/${mt}/${log_model_name}_lr=${learning_rate}_bsz=${bsz_efctv}_epochs=${num_epochs}_seed=${seed}/"
logfile="${logdir}/train.out"
mkdir -p $logdir

# Copy source code
mkdir -p $logdir/src
cp $HOME/MaCoCu/student_project_mt_ht/classifier_trf_hf.py $logdir/src

# Copy this script
cp $(realpath $0) $logdir


cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf_hf.py \
--root_dir $root_dir \
--arch $arch \
--learning_rate $learning_rate \
--batch_size $bsz \
--num_epochs $num_epochs \
--weight_decay $weight_decay \
--max_grad_norm $max_grad_norm \
--warmup_steps $warmup_steps \
--label_smoothing $label_smoothing \
--dropout $dropout \
--seed $seed \
--gradient_accumulation_steps $gradient_accumulation_steps \
--max_length $max_length \
$flags \
$fp_flag \
&> $logfile
