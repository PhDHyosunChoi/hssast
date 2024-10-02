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

#mkdir exp
#mkdir slurm_log

task=pretrain_joint
mask_patch=400

# audioset and librispeech
dataset=esc50 #asli

# [Hyosun] editied the data_file_path
#tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/audioset_librispeech.json
#tr_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/esc_class_labels_indices.csv
tr_data=/content/drive/MyDrive/hssast/src/pretrain/ESC50/data/datafiles/esc_train_data_1.json
#te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
te_data=/content/drive/MyDrive/hssast/src/pretrain/ESC50/data/datafiles/esc_eval_data_1.json
# [/Hyosun] editied the file_path

# # [Hyosun] edit needed later
# dataset_mean=-4.2677393
# dataset_std=4.5689974
# target_length=1024
# num_mel_bins=128
# # [/Hyosun] edit needed later

# [Hyosun] commented out ==> modified and inserted again 20240813
#[Hyosun] added for dataset esc50
#dataset=esc50
dataset_mean=-6.6268077
dataset_std=5.358466
target_length=1024 #[Hyosun] 2024-08-13 #512 #[Hyosun] back to the original 20240329 #1024 #512 #[Hyosun] The original value 512 is modified 20230704
#noise=True #[Hyosun] might need to be inserted? 20230704 try experiment #commented out
num_mel_bins=128 #[Hyosun] confirmed by the file src/finetune/esc50/run_esc_patch.sh
#[/Hyosun] added for dataset esc50
# [/Hyosun] commented out ==> modified and inserted again 20240813

model_size=tiny
# no patch split overlap
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}
# no class balancing as it implicitly uses label information
bal=none
batch_size=120
lr=5e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
epoch=2000
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0

# [Hyosun] edited the path 2024-08-13
#exp_dir=./exp/mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
exp_dir=./exp/"$(date +'%Y-%m-%d/%H:%M:%S%p')"-mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}-loss-${loss}-${model_size}
# loss-${loss}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}
# base_exp_dir=./exp/"$(date +'%Y-%m-%d/%H:%M:%S%p')"-test01-${dataset}-comp_fusion-${comp_fusion}-comp_fusion_method-${comp_fusion_method}-comp_fusion_multi_layer-${comp_fusion_multi_layer}-pooling-${pooling_ty}-comp_fusion_mlp${mlp_layers}-loss-${loss}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}
# [/Hyosun] edited the path 2024-08-13 

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/esc_class_labels_indices.csv \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 800