#!/bin/bash

#SBATCH --job-name='6_train'
#SBATCH --partition=gpushort
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --array=212-214
#SBATCH --output=train_seed_%a.out

EXP_ID=6
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}
SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source $HOME/activate_py3.8.6

cd $HOME/MaCoCu/student_project_mt_ht/
python classifier_trf.py --root_dir $ROOT_DIR --arch bert --num_epochs 50 --seed ${SEED} \
                         --load_model $ROOT_DIR/../2/models/bert_outputs_50_epoch_seed_${SEED}/best_model/ --learning_rate 1e-6
