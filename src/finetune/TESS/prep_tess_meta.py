# -*- coding: utf-8 -*-
# @Time    : 05/02/24 11:00 AM
# @Author  : Hyosun Choi
# @Affiliation  : RHUL
# @Email   : bbeyes81@gmail.com
# @File    : prep_tess.py

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random       #[Hyosun]

# dataset downloaded from https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download
# please change it to your TESS dataset path
# [Hyosun] current path: ./src/prep_data/tess [/Hyosun]
tess_path = './TESS/'

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
Emotions_labels_arr = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised'] #[Hyosun] Emotions model has 7 labels
labels_arr = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'ps'] #[Hyosun] TESS has 7 labels: 'ps' means 'pleasant surprise'
short_labels_arr = ['ANG', 'HAP', 'SAD', 'NEU', 'FEA', 'DIS', 'PSU']
if os.path.exists('./data/tess_class_labels_indices.csv') == False:
    os.mkdir('./data/')
    index = []
    mid = []
    for i in range(0,len(labels_arr)): #7 labels for tess
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
        # writer.writerow([i,'/m/07rwj'+str(i).zfill(2), labels_arr[i]])
    print("mid: ", mid)
    print("labels_arr: ", labels_arr)
    print("short_labels_arr: ", short_labels_arr)
    dict = {'index': index, 'mid':mid, 'display_name': labels_arr, 'short_display_name': short_labels_arr}
    df = pd.DataFrame(dict)
    print(df)
  
    # saving the dataframe
    df.to_csv('./data/tess_class_labels_indices.csv', index=False)  

    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================
# [Hyosun: create the label description dictionary] 
if os.path.exists('./data/tess_class_labels_indices.csv') == True:    
    label_set = np.loadtxt('./data/tess_class_labels_indices.csv', delimiter=',', dtype='str')
    long_map_label = {}
    short_map_label = {}
    label_map_short = {}
    for i in range(1, len(label_set)):
        print("[Hyosun] label_set[i][0]: ", label_set[i][0])
        print("[Hyosun] label_set[i][2]: ", label_set[i][2])
        print("[Hyosun] label_set[i][3]: ", label_set[i][3])
        long_map_label[(label_set[i][2])] = label_set[i][0]  
        short_map_label[(label_set[i][3])] = label_set[i][0]   
        label_map_short[(label_set[i][0])] = label_set[i][3]
    print(long_map_label)
    print(short_map_label)
    print(label_map_short)    
    print("[Hyosun] create the label description dictionary: using tess_class_labels_indices.csv finished!!")
#[/Hyosun: create the label description dictionary] 

if os.path.exists('./TESS/') == False:
    print("Please check the directory './TESS/'")

# [Hyosun] create fold1_train.csv, fold1_test.csv, fold1_evaluate.csv
if os.path.exists('./data/evaluation_setup/fold1_train.csv') == False or \
   os.path.exists('./data/evaluation_setup/fold1_test.csv') == False:
    if os.path.exists('./data/evaluation_setup') == False:
        os.mkdir('./data/evaluation_setup/')
    base_dir = './TESS/'  

    # create train set and test set: 
    # create train and test files.csv
    no_train=0
    no_test=0
    
    train_dict_list = []
    test_dict_list = []
    # labels = []
    # train_rate = 0.8

    # [Hyosun] speaker(actor) independent testing 
    # random.seed(1981)
    ac_lb_list = []
    for ac in ['YAF', 'OAF']:
        for lb in labels_arr:
            ac_lb_list.append(ac+'_'+lb)
    # ac_lb_list = get_immediate_files(base_dir)    
    for ac_lb in ac_lb_list:
        print("ac_lb: " , ac_lb)
        random.seed(1981)
        audio_list_each_ac_lb = []
        audio_list_each_ac_lb = get_immediate_files(base_dir+ac_lb)

        #[Hyosun] shuffle or not, put the code here (later) [/Hyosun]
        for audio in audio_list_each_ac_lb:
            # print("[Hyosun] ", base_dir + tr_actor + '/' + tr_audio, " will be added to train_dict_list")
            file_name = ac_lb.split('_')
            label = file_name[1] 
            
            # train set
            if file_name[0] == 'YAF':           
                train_dict = {"wav": base_dir + ac_lb + '/' + audio, "labels": label}
                train_dict_list.append(train_dict)
                no_train+=1
                
            # test set
            else: #if file_name[0] == 'OAF':    
                test_dict = {"wav": base_dir + ac_lb + '/' + audio, "labels": label} 
                test_dict_list.append(test_dict)
                no_test+=1
    # [/Hyosun] speaker(actor) independent testing 

    print('[Hyosun] {:d} train samples and {:d} test samples resulting {:d} total samples'.format(no_train,no_test,no_train + no_test))                  
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

    # print('[Hyosun] Finished Meta Data Preparation')
