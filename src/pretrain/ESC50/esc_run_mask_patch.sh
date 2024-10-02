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
#export TORCH_HOME=../../pretrained_models_by_Hyosun
export TORCH_HOME=../../pretrained_models #[Hyosun] restored the original line here 
# [/Hyosun] editied the folder_name

mkdir exp
mkdir slurm_log

task=pretrain_joint
mask_patch=400
epoch_iter=5 #[Hyosun] added

# audioset and librispeech
#dataset=asli #[Hyosun] commendted out
dataset=esc50  #[Hyosun] added

#[Hyosun] loss specified for the exp result folders
loss='CE'
#[/Hyosun]loss specified for the exp result folders

#[Hyosun] comp_fusion logic added
comp_fusion='True'
comp_fusion_method='use_all_patch'
comp_fusion_multi_layer='[4,11]'
pooling_ty='mean_max' #[Hyosun] #choices=["mean", "min", "max", "mean_min", "mean_max"] #[/Hyosun]
mlp_layers=2 #6 #4 #2 #6 #4
#[/Hyosun] comp_fusion logic added

# [Hyosun:folds logic] commented out
# # [Hyosun] editied the data_file_path: 여기 파일 1,2..로 나뉜거 말고 합친거로 돌려야할듯
# #tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/audioset_librispeech.json
# tr_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/data/datafiles/esc_train_data_1.json
# #te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
# te_data=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/prep_data/esc50/data/datafiles/esc_eval_data_1.json
# # [/Hyosun] editied the data_file_path
# [/Hyosun:folds logic] commented out

# [Hyosun] commented out 
# # [Hyosun] edit needed later  #[Hyosun] The below 4 lines might have affected the accuracy performances
# dataset_mean=-4.2677393
# dataset_std=4.5689974
# target_length=1024
# num_mel_bins=128
# # [/Hyosun] edit needed later #[/Hyosun] The below 4 lines might have affected the accuracy performances
# [/Hyosun] commented out 

# [Hyosun] commented out ==> modified and inserted again 20230704
#[Hyosun] added for dataset esc50
#dataset=esc50
dataset_mean=-6.6268077
dataset_std=5.358466
target_length=512 #[Hyosun] back to the original 20240329 #1024 #512 #[Hyosun] The original value 512 is modified 20230704
noise=True #[Hyosun] might need to be inserted? 20230704 try experiment
num_mel_bins=128 #[Hyosun] confirmed by the file src/finetune/esc50/run_esc_patch.sh
#[/Hyosun] added for dataset esc50
# [/Hyosun] commented out ==> modified and inserted again 20230704

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

#base_exp_dir=./exp/mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
#base_exp_dir=./exp/date"$(date +'%Y-%m-%d/%H:%M:%S%p')"-mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
base_exp_dir=./exp/"$(date +'%Y-%m-%d/%H:%M:%S%p')"-test01-${dataset}-comp_fusion-${comp_fusion}-comp_fusion_method-${comp_fusion_method}-comp_fusion_multi_layer-${comp_fusion_multi_layer}-pooling-${pooling_ty}-comp_fusion_mlp${mlp_layers}-loss-${loss}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}
#[/Hyosun] edited

for((fold=1;fold<=2;fold++));#5;fold++)); #[Hyosun: for testing(temp)] 5fold->2
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

#   tr_data=./data/datafiles/esc_train_data_${fold}.json
#   te_data=./data/datafiles/esc_eval_data_${fold}.json

# [Hyosun] editied the data_file_path: 여기 파일 1,2..로 나뉜거 말고 합친거로 돌려야할듯
  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json
# [/Hyosun] editied the data_file_path

  # [Hyosun] insert the original code from [ssast] run_mask_patch.sh
  # [Hyosun] editied the file_path into: ./data/esc_class_labels_indices.csv \
  CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
  --model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
  --task ${task} --lr_patience ${lr_patience} --epoch_iter ${epoch_iter} \
  --comp_fusion ${comp_fusion} --comp_fusion_method ${comp_fusion_method} --comp_fusion_multi_layer ${comp_fusion_multi_layer} --pooling_ty ${pooling_ty} \
  --mlp_layers ${mlp_layers}
  # [Hyosun] commented out and changed --epoch_iter 4000 into 5 #[Hyosun] into 500 again for easy testing
  #--task ${task} --lr_patience ${lr_patience} --epoch_iter 4000
  #--task ${task} --lr_patience ${lr_patience} --epoch_iter 5
  #--task ${task} --lr_patience ${lr_patience} --epoch_iter 1
  # [/Hyosun] commented out and changed --epoch_iter 4000 into 5 #[/Hyosun] into 500 again for easy testing
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