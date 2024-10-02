# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
from array import ArrayType
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import ASTModel
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
#print("basepath: ", basepath) #[Hyosun] added for checkout
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep the same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')

parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])

# pretraining augments
#parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models or [Hyosun]the fine-tuned models path")
    #[Hyosun:continue_train==True]the best optimizer state path to continue(resume) training as fine-tuning 2024-05-02
parser.add_argument("--optimizer_path", type=str, default=None, help="[Hyosun:continue_train==True]the best optimizer state path to continue(resume) training as fine-tuning")
    #[/Hyosun:continue_train==True]the best optimizer state path to continue(resume) training as fine-tuning 2024-05-02
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
  #[Hyosun] comp_fusion logic added
parser.add_argument("--comp_fusion", type=str, default='True', help="whether composing fusion or not for finetuning", choices=['True', 'False'])
parser.add_argument("--comp_fusion_method", type=str, default="use_all_patch", help="whether to use_all_patch or only cls_token for finetuning", choices=['use_all_patch', 'cls_token'])
parser.add_argument("--comp_fusion_multi_layer", type=str, default='[4,11]', help="which multi-layers you want to apply fusion for finetuning", choices=['[4,11]','[5,12]', '[]'])
parser.add_argument("--pooling_ty", type=str, default="mean_max", help="which pooling type you want to use", choices=["mean", "min", "max", "mean_min", "mean_max", "max_max"])
parser.add_argument("--mlp_layers", type=int, default=4, help="how many mlp_head layers you want to use for comp_fusion", choices=[0,2,4,6,8])
  #[/Hyosun] comp_fusion logic  added
  #[Hyosun] classification_only added 2024-01-28
parser.add_argument("--classification_only", type=str, default="False", help="whether to apply classification only", choices=["True", "False"])
parser.add_argument("--best_model_dir", type=str, default="", help="The best model path")
  #[/Hyosun] classification_only added 2024-01-28
  #[Hyosun] continue_train added 2024-04-30 
parser.add_argument("--continue_train", type=str, default="False", help="whether to continue training from where you left off", choices=["True", "False"])
  #[/Hyosun] continue_train added 2024-04-30 
args = parser.parse_args()

# # dataset spectrogram mean and std, used to normalize the input
# norm_stats = {'librispeech':[-4.2677393, 4.5689974], 'howto100m':[-4.2677393, 4.5689974], 'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
# target_length = {'librispeech': 1024, 'howto100m':1024, 'audioset':1024, 'esc50':512, 'speechcommands':128}
# # if add noise for data augmentation, only use for speech commands
# noise = {'librispeech': False, 'howto100m': False, 'audioset': False, 'esc50': False, 'speechcommands':True}

audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# if use balanced sampling, note - self-supervised pretraining should not use balance sampling as it implicitly leverages the label information.
if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

# in the pretraining stage
if 'pretrain' in args.task:
    cluster = (args.num_mel_bins != args.fshape)
    if cluster == True:
        print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
    else:
        print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
    # no label dimension needed as it is self-supervised, fshape=fstride and tshape=tstride
    audio_model = ASTModel(
                    #[Hyosun] comp_logic added
                    comp_fusion=args.comp_fusion,                          #[Hyosun] comp_fusion added
                    comp_fusion_method=args.comp_fusion_method,            #[Hyosun] comp_fusion_method added
                    comp_fusion_multi_layer=args.comp_fusion_multi_layer,  #[Hyosun] comp_fusion_multi_layer added
                    pooling_ty=args.pooling_ty,                            #[Hyosun] pooling_ty added
                    mlp_layers=args.mlp_layers,                            #[Hyosun] mlp_layers added
                    #[/Hyosun] comp_logic added        
                    fshape=args.fshape, tshape=args.tshape, fstride=args.fshape, tstride=args.tshape,
                    input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True)
# in the fine-tuning stage [Hyosun] focus and use!!!
else: #[Hyosun] set-up model for fine-tuning: 1.normal fine-tuning, 2.classficaiton_only, 3.continue_train
    # if args.classification_only=="False": #[Hyosun]fine-tuning
    print("[Hyosun:run.py] args.comp_fusion: ", args.comp_fusion)
    print("[Hyosun:run.py] args.comp_fusion_method: ", args.comp_fusion_method)
    print("[Hyosun:run.py] args.comp_fusion_multi_layer: ", args.comp_fusion_multi_layer)
    print("[Hyosun:run.py] args.pooling_ty: ", args.pooling_ty)
    audio_model = ASTModel(
                    #[Hyosun] comp_logic added
                    comp_fusion=args.comp_fusion,                          #[Hyosun] comp_fusion added
                    comp_fusion_method=args.comp_fusion_method,            #[Hyosun] comp_fusion_method added
                    comp_fusion_multi_layer=args.comp_fusion_multi_layer,  #[Hyosun] comp_fusion_multi_layer added
                    pooling_ty=args.pooling_ty,                            #[Hyosun] pooling_ty added
                    mlp_layers=args.mlp_layers,                            #[Hyosun] mlp_layers added
                    #[/Hyosun] comp_logic added
                    label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                    input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                    #[Hyosun] classification_only(no fine-tuning) or continue_train as fine-tuning
                    load_pretrained_mdl_path=args.pretrained_mdl_path,     #[Hyosun] call the model that are already saved, using args.pretrained_mdl_path 
                    load_optimizer_path=args.optimizer_path,               #[Hyosun] optimizer_path added to continue(resume) training as fine-tuning 2024-05-02
                    classification_only=args.classification_only,          #[Hyosun] classification_only added
                    continue_train=args.continue_train                     #[Hyosun] continue_train added
                    #[/Hyosun] classification_only(no fine-tuning) or continue_train as fine-tuning
    )
                    #  #[Hyosun] comp_logic added
                    #  comp_fusion=args.comp_fusion,                      #[Hyosun] comp_fusion added
                    #  comp_fusion_method=args.comp_fusion_method,        #[Hyosun] comp_fusion_method added
                    #  comp_fusion_multi_layer=args.comp_fusion_multi_layer,  #[Hyosun] comp_fusion_multi_layer added
                    #  pooling_ty=args.pooling_ty)                        #[Hyosun] pooling_ty added
                    #  #[/Hyosun] comp_logic added
    # else: #[Hyosun] classification_only

if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

#[Hyosun commented] Pretrain, Fine-tuning(inc.continue_train), or classification_only here
if 'pretrain' not in args.task: #[Hyosun] fine-tuning: connects to traintest.py's train(), or classification only[/Hyosun]
    #[Hyosun] fine-tuning: if-structure added 2024-01-28[/Hyosun]
    if args.classification_only=="False": #[Hyosun] continue_train == 'True': is the case here too. 2024-05-05
        print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
        train(audio_model, train_loader, val_loader, args) #[Hyosun] fine-tuning: 여기서 fusion? Yes
    #[Hyosun] classification_only added 2024-01-28
    else:#[Hyosun] args.classification_only=="True":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.pretrained_mdl_path, map_location=device)
        if not isinstance(audio_model, torch.nn.DataParallel):
            audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=False)

        #[Hyosun] added 2024-02-01
        import torch
        from torch import nn
        # best models on the validation set => [Hyosun]connects to traintest.py's validate()
        # args.loss_fn = torch.nn.BCEWithLogitsLoss() #[Hyosun] already asigned as nn.CrossEntropyLoss() at traintest.py    
        if args.loss == 'BCE':  #[Hyosun:comment] seems mostly for speech or audio datasets
            loss_fn = nn.BCEWithLogitsLoss()
        elif args.loss == 'CE': #[Hyosun:comment] Data ESC50, TAU <= controlled by bash files
            loss_fn = nn.CrossEntropyLoss()
        args.loss_fn = loss_fn
        
        # audio_output = audio_model(audio_input, args.task) #[Hyosun] call forward(): fusion here? Yes inside forward()
        # if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        #     loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
        # else:
        #     loss = loss_fn(audio_output, labels)
        #[/Hyosun] added 2024-02-01

        print("[Hyosun:run.py] args.classification_only==True and args.loss: ", args.loss, " for val idate")
        stats, _ = validate(audio_model, val_loader, args, 'valid_set')
        # note it is NOT mean of class-wise accuracy
        val_acc = stats[0]['acc']
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the validation set---------------')
        print("Accuracy: {:.6f}".format(val_acc))
        print("AUC: {:.6f}".format(val_mAUC))
        #[Hyosun] save the results [/Hyosun]
        np.savetxt(args.exp_dir + '/val_result_classification_only.csv', [val_acc, val_mAUC])
    #[/Hyosun] classification_only added 2024-01-28
else: #[Hyosun] pretrain => [Hyosun] connects to traintest_mask.py's trainmask()
    print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
    trainmask(audio_model, train_loader, val_loader, args)#[Hyosun] real train masks

# if the dataset has a seperate evaluation set (e.g., speechcommands), then select the model using the validation set and eval on the evaluation set.
# this is only for fine-tuning
#[Hyosun:comment] Except for ESC50, TAU and my datasets in my exeriments: --data-eval == None
                # skip this if-statement logic because it doesn't have --data-eval (the same as data_eval)[/Hyosun]
if args.data_eval != None: #[Hyosun:comment] For TAU data: --data-train ${tr_data} --data-val ${te_data} but no --data-eval[/Hyosun]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # best models on the validation set => [Hyosun]traintest.py의 validate()함수로 연결
    args.loss_fn = torch.nn.BCEWithLogitsLoss()
    print("[Hyosun:run.py] args.data_eval != None and args.loss_fn: ", args.loss_fn, " for validate")
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the models on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

