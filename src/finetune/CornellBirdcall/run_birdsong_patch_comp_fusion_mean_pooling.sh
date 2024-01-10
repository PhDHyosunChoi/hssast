#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
##comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc #[Hyosun] commented out

#source ../../../venvssast/bin/activate #[Hyosun] commented out: virtual env activate하는 line

# [Hyosun] editied
#export TORCH_HOME=../../pretrained_models
export TORCH_HOME=../../../pretrained_models
#[/Hyosun]
mkdir exp

# prep esc50 dataset and download the pretrained model
if [ -e data/datafiles ]
then
    echo "birdsong already downloaded and processed."
else
    python prep_birdsong.py
fi
if [ -e SSAST-Base-Patch-400.pth ] # [Hyosun] -e: whether exist  
then
    echo "pretrained model already downloaded."
else
    wget https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1 -O SSAST-Base-Patch-400.pth
fi

#[Hyosun] save my own model.pth here, then use it for fine-tuning
pretrain_exp=
#/[Hyosun] save my own model.pth here, then use it for fine-tuning

pretrain_model=SSAST-Base-Patch-400
#[Hyosun] edited
#pretrain_path=./${pretrain_exp}/${pretrain_model}.pth 
pretrain_path=./${pretrain_model}.pth 
#[/Hyosun] 

#[Hyosun] birdsong setting: edited for birdsong mean=-6.519173 std=4.813366
## -- target_length 바꾸면 acc 올라가나?? length 도 바꿀까, 1024정도로?? 2023-10-04
dataset=birdsong
dataset_mean=-6.519173
dataset_std=4.813366
target_length=512
noise=True
#[/Hyosun] birdsong setting: edited for birdsong mean=-6.519173 std=4.813366

bal=none
#[Hyosun] Experiments of lr #[/Hyosun]
lr=1e-4
freqm=24
timem=96
mixup=0
epoch=22 #45 #50 #10 #50 #50 #5  #50 [Hyosun] The original epoch 50: reduced into 5 for experiments
batch_size=48
fshape=16
tshape=16
fstride=10
tstride=10

task=ft_avgtok
model_size=base
#[Hyosun] Experiments of lr #[/Hyosun]
head_lr=1

#[Hyosun] loss specified for the exp result folders
loss='CE'
#[/Hyosun]loss specified for the exp result folders

#[Hyosun] comp_fusion logic added
comp_fusion='True'
comp_fusion_method='use_all_patch'
comp_fusion_multi_layer='[4,11]'
pooling_ty='mean' #[Hyosun] #choices=["mean", "min", "max", "mean_min", "mean_max"] #[/Hyosun]
mlp_layers=6 #4 #2
#[/Hyosun] comp_fusion logic added

#[Hyosun]
#base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}
#base_exp_dir=/content/drive/MyDrive/ColabNotebooks/Github/ssast/src/finetune/esc50/exp/test 
#01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise} #[Hyosun]curtailed
#base_exp_dir=/content/drive/MyDrive/ColabNotebooks/Github/hssast/src/finetune/esc50/exp/test #[Hyosun] ssast->hssast
#base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise} #[Hyosun] ssast->hssast
#base_exp_dir=./exp/"$(date +'%Y-%m-%d/%H:%M:%S%p')"-test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise} #[Hyosun] ssast->hssast
base_exp_dir=./exp/"$(date +'%Y-%m-%d/%H:%M:%S%p')"-test01-${dataset}-comp_fusion-${comp_fusion}-comp_fusion_method-${comp_fusion_method}-comp_fusion_multi_layer-${comp_fusion_multi_layer}-pooling-${pooling_ty}-comp_fusion_mlp${mlp_layers}-loss-${loss}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}
#                                                            [Hyosun]-comp_fusion-${comp_fusion}-comp_fusion_method-${comp_fusion_method}-pooling-${pooling_ty}-added
#[/Hyosun]

for((fold=1;fold<=1;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/birdsong_train_data_${fold}.json
  te_data=./data/datafiles/birdsong_test_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/birdsong_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
  --model_size ${model_size} --adaptschedule False \
  --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss ${loss} --metrics acc \
  --comp_fusion ${comp_fusion} --comp_fusion_method ${comp_fusion_method} --comp_fusion_multi_layer ${comp_fusion_multi_layer} --pooling_ty ${pooling_ty} \
  --mlp_layers ${mlp_layers}
done
#[Hyosun] "--comp_fusion $comp_fusion --comp_fusion_method $comp_fusion_method  --pooling_ty $pooling_ty" added [/Hyosun]
#[Hyosun] editied: --loss CE -> --loss ${loss} [/Hyosun]
#[Hyosun] added:   --mlp_layers ${mlp_layers}  [/Hyosun]
#[Hyosun] editied: --n_class 10 -> 50          [/Hyosun]

#[Hyosun] edited 2023-10-17
#python ./get_esc_result.py --exp_path ${base_exp_dir}
python get_birdsong_result.py --exp_path ${base_exp_dir} #요거이용하기(2023-10-02)
#[/Hyosun] edited needed 2023-10-17