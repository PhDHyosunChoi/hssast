# -*- coding: utf-8 -*-
# @Time    : 19/12/23 15:41 PM
# @Author  : Hyosun Choi
# @Affiliation  : RHUL
# @Email   : 
# @File    : event_get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

import dataloader #[Hyosun comment] 현재파일은 자체제작한 dataloader.py랑 연동되는 파일임(같은경로에 dataloader.py두기)

# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 24, 'timem': 192, 'mixup': 0.5, 'skip_norm': True, 'mode': 'train', 'dataset': 'event'}

train_loader = torch.utils.data.DataLoader(
    # [Hyosun] edited as the function name changed by SSAST team since AST
    # dataloader.AudiosetDataset('./data/datafiles/tau_train_data_1.json', label_csv='./data/tau_class_labels_indices.csv',
    dataloader.AudioDataset('./data/datafiles/event_04s_train_data_1.json', label_csv='./data/event_04s_class_labels_indices.csv',
                                audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
                                                        #[Hyosun] batch size(This is not a real batch_size, we assign the real one at run.py: train_loader), 
                                                        ##        doesn't matter with the mean,std,length
    # [/Hyosun] edited as the function name changed by SSAST team since AST
print("[Hyosun] The shape of train_loader: ", np.shape(train_loader)) #[Hyosun] we can see the length from here

mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))