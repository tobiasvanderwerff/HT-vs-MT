#!/bin/bash

#SBATCH --job-name='train'
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=t.n.van.der.werff@student.rug.nl


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

exp_id=1
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

# Hyper-parameters
hparams_file="${root_dir}/hparams.txt"  # file specifiying all the different hyper-parameter configurations
hparams=$(head -$SLURM_ARRAY_TASK_ID $hparams_file | tail -1)
arch=$(echo $hparams | awk '{print $2}')
learning_rate=$(echo $hparams | awk '{print $4}')
bsz=$(echo $hparams | awk '{print $6}')

# Defeault hyper-parameters
num_epochs=8
weight_decay=0
max_grad_norm=1
warmup_steps=200
label_smoothing=0.0
dropout=0.1
seed=1
early_stopping_patience=10

log_model_name=$(echo $arch | sed 's/\//-/g')
# Make sure the logdir specified below corresponds to the directory defined in the
# main() function of the `classifier_trf_hf.py` script!
logdir="${root_dir}/models/${log_model_name}_lr=${learning_rate}_bsz=${bsz}_seed=${seed}/"
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
--early_stopping_patience $early_stopping_patience \
&> $logfile
