#!/bin/bash

#SBATCH --job-name='train_all'
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#--array=1-24
#SBATCH --array=1


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface

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
seed=1

log_model_name=$(echo $arch | awk -F: '{print $2}')
if [[ $log_model_name == */* ]]; then  # check if model name contains `/` character
    log_model_name=$(echo $log_model_name | awk -F/ '{print $1"-"$2}')
fi

# The logdir below corresponds to the directory defined in the main() function of the
# `classifier_trf.py` script. Therefore, ONLY change this logdir if the directory
# defined in the Python script also changed.
logdir="${root_dir}/models/${log_model_name}_lr=${learning_rate}_bsz=${bsz}_seed=${seed}/"
logfile="${logdir}/train.out"
mkdir -p $logdir

cd $HOME/MaCoCu/student_project_mt_ht/
srun python -u classifier_trf.py \
--root_dir $root_dir \
$hparams \
--seed $seed \
&> $logfile
