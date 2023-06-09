#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[3],sls-1080-3,sls-sm-5
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ssast_pretrain"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc #[Hyosun] commented out

#source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate #[Hyosun] commented out

# [Hyosun] editied the folder_name
#export TORCH_HOME=../../pretrained_models
export TORCH_HOME=../../pretrained_models_by_Hyosun
# [/Hyosun] editied the folder_name

mkdir exp
mkdir slurm_log

task=pretrain_joint
mask_patch=400
epoch_iter=5 #[Hyosun] added

# audioset and librispeech
#dataset=asli #[Hyosun] commendted out
dataset=esc50  #[Hyosun] added

# [Hyosun:folds logic] commented out
# # [Hyosun] editied the data_file_path: 여기 파일 1,2..로 나뉜거 말고 합친거로 돌려야할듯
# #tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/audioset_librispeech.json
# tr_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/data/datafiles/esc_train_data_1.json
# #te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
# te_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/data/datafiles/esc_eval_data_1.json
# # [/Hyosun] editied the data_file_path
# [/Hyosun:folds logic] commented out

# [Hyosun] edit needed later
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128
# [/Hyosun] edit needed later

model_size=base
# no patch split overlap
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}
# no class balancing as it implicitly uses label information
bal=none
batch_size=24
lr=1e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
epoch=10
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0


#[Hyosun:folds logic] commented out by Hyosun to use folds
#exp_dir=./exp/mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
#[Hyosun:folds logic] commented out by Hyosun to use folds

# # [Hyosun] added
# label-csv=./data/esc_class_labels_indices.csv
# save_model=False
# n-print-steps=100
# epoch_iter=4000
# # [/Hyosun] added

#[Hyosun:folds logic] original code commented out by Hyosun to make it in a for-loop for folds
# # [Hyosun] editied the file_path into: ./data/esc_class_labels_indices.csv \
# CUDA_CACHE_DISABLE=1 python -W ignore ../run.py --dataset ${dataset} \
# --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
# --label-csv ./data/esc_class_labels_indices.csv \
# --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
# --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
# --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
# --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
# --model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
# --task ${task} --lr_patience ${lr_patience} --epoch_iter 4000
#[/Hyosun:folds logic] original code commented out by Hyosun to make it in a for-loop for folds

#[Hyosun] added the results & folds logic from [ast] run_esc.sh ====================================
#[Hyosun] edited
#base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}
base_exp_dir=./exp/mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
#[/Hyosun] edited

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

#   tr_data=./data/datafiles/esc_train_data_${fold}.json
#   te_data=./data/datafiles/esc_eval_data_${fold}.json

# [Hyosun] editied the data_file_path: 여기 파일 1,2..로 나뉜거 말고 합친거로 돌려야할듯
#tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/audioset_librispeech.json
  tr_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/data/datafiles/esc_train_data_${fold}.json
#te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
  te_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/data/datafiles/esc_eval_data_${fold}.json
# [/Hyosun] editied the data_file_path

  # [Hyosun] insert the original code from [ssast] run_mask_patch.sh
  # [Hyosun] editied the file_path into: ./data/esc_class_labels_indices.csv \
  CUDA_CACHE_DISABLE=1 python -W ignore ../run.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
  --model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
  --task ${task} --lr_patience ${lr_patience} --epoch_iter 5
  # [Hyosun] commented out and changed --epoch_iter 4000 into 5
  #--task ${task} --lr_patience ${lr_patience} --epoch_iter 4000
  # [/Hyosun] commented out and changed --epoch_iter 4000 into 5
  # [/Hyosun] insert the original code from [ssast] run_mask_patch.sh

  # #[Hyosun] original code for [ast] run_esc.sh commented out by Hyosun
  # CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  # --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  # --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  # --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  # --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  # --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  # --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  # --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}
  # #[/Hyosun] original code for ssast commented out by Hyosun
done

python ./get_esc_result.py --exp_path ${base_exp_dir}
#[/Hyosun] added the results & folds logic from [ast] run_esc.sh ===================================