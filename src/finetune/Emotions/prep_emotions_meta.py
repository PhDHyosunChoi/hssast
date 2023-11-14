# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_tau.py

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]

# dataset downloaded from https://zenodo.org/record/3819968
# please change it to your TAU dataset path
# [Hyosun] current path: ./src/prep_data/unzipped_tau [/Hyosun]
emotions_path = './data/Emotions/'

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
labels_arr = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised']
if os.path.exists('./data/emotions_class_labels_indices.csv') == False:
    # if os.path.exists('./data') == False:
    #     os.mkdir('./data/')

    # with open('./data/tau_class_labels_indices.csv', 'w') as out_file:
    #     writer = csv.writer(out_file)
    #     writer.writerow(['index','mid','display_name']) # title row
    index = []
    mid = []

    for i in range(0,len(labels_arr)): #7 labels for emotions
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
        # writer.writerow([i,'/m/07rwj'+str(i).zfill(2), labels_arr[i]])
    print("mid: ", mid)
    print("labels_arr: ", labels_arr)
    dict = {'index': index, 'mid':mid, 'display_name': labels_arr}
    df = pd.DataFrame(dict)
    print(df)
  
    # saving the dataframe
    df.to_csv('./data/emotions_class_labels_indices.csv', index=False)  
    # df.to_csv(r'C:\Users\Admin\Desktop\file3.csv')

    # out_file.close()
    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================


# [Hyosun] create fold1_train.csv, fold1_test.csv, fold1_evaluate.csv
if os.path.exists('./data/evaluation_setup/fold1_train.csv') == False or \
   os.path.exists('./data/evaluation_setup/fold1_test.csv') == False:
    if os.path.exists('./data/evaluation_setup') == False:
        os.mkdir('./data/evaluation_setup/')
    # get audio file names in each emotion folder: 'Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised'
    base_dir = './data/Emotions/'
    import random
    train_dict_list = []
    test_dict_list = []
    labels = []
    train_rate = 0.8
    for i in range(0,len(labels_arr)):
        random.seed(1981)
        audio_list_each_label = []
        audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])
        no_train = 0
        no_test = 0
        #[Hyosun] shuffle or not, put the code here (later) [/Hyosun]
        for j in range(0, int(len(audio_list_each_label)*train_rate)):
            # print("[Hyosun] ", base_dir + labels_arr[i] + '/' + audio_list_each_label[j], " will be added to train_dict_list")
            train_dict = {"wav": base_dir + labels_arr[i] + '/' + audio_list_each_label[j], "labels": labels_arr[i]}
            train_dict_list.append(train_dict)
            no_train+=1 
       
        for k in range(int(len(audio_list_each_label)*train_rate),len(audio_list_each_label)): #[Hyosun] from 81% to the end
            # print("[Hyosun] ", base_dir + labels_arr[i] + '/' + audio_list_each_label[k], " will be added to test_dict_list")
            test_dict = {"wav": base_dir + labels_arr[i] + '/' + audio_list_each_label[k], "labels": labels_arr[i]}
            test_dict_list.append(test_dict)
            no_test+=1

        print('[Hyosun:{:s}] {:d} train samples and {:d} test samples resulting {:d} total samples in label {:s}'.format(labels_arr[i],no_train,no_test,no_train + no_test,labels_arr[i]))                    
        print('[Hyosun:{:s}] {:d} train samples in label {:s}'.format(labels_arr[i],int(len(audio_list_each_label)*train_rate),labels_arr[i]))          
        print('[Hyosun:{:s}] {:d} test samples in label {:s}'.format(labels_arr[i],len(audio_list_each_label)-int(len(audio_list_each_label)*train_rate),labels_arr[i]))
        print('[Hyosun:{:s}] total {:d} samples in label {:s}'.format(labels_arr[i],len(audio_list_each_label),labels_arr[i]))  

    print('[Hyosun] {:d} training samples, {:d} testing samples of total {:d} samples'.format(len(train_dict_list), len(test_dict_list),len(train_dict_list) + len(test_dict_list)))

    # with open('./data/evaluation_setup/fold1_train' +'.csv', 'w') as f:
    #     json.dump({'data': train_dict_list}, f, indent=1)

    # with open('./data/evaluation_setup/fold1_test' +'.csv', 'w') as f:
    #     json.dump({'data': test_dict_list}, f, indent=1)

    train_df = pd.DataFrame(train_dict_list)
    test_df = pd.DataFrame(test_dict_list)
    # print(train_df)
    # print(test_df)
  
    # saving the dataframe
    train_df.to_csv('./data/evaluation_setup/fold1_train.csv', sep='\t', index=False)  
    test_df.to_csv('./data/evaluation_setup/fold1_test.csv', sep='\t', index=False)      
    print("[Hyosun] create train and test files: All Finished!!")
# [/Hyosun] create fold1_train.csv, fold1_test.csv, fold1_evaluate.csv

    # print('[Hyosun] Finished Emotions Meta Data Preparation')
