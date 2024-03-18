# -*- coding: utf-8 -*-
# @Time    : 06/02/24 11:00 AM
# @Author  : Hyosun Choi
# @Affiliation  : RHUL
# @Email   : bbeyes81@gmail.com
# @File    : prep_savee_meta.py

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random       #[Hyosun]

# dataset downloaded from https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
# please change it to your SAVEE dataset path
# [Hyosun] current path: ./src/prep_data/savee [/Hyosun]
savee_path = './SAVEE/'

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
Emotions_labels_arr = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised'] #[Hyosun] Emotions model has 7 labels
#SAVEE labels: anger, disgust, fear, happiness, sadness and surprise. A neutral category is also added to provide recordings of 7 emotion categories.(from SAVEE homepage)
labels_arr = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise'] #[Hyosun] SAVEE has 7 labels
short_labels_arr = ['ANG', 'HAP', 'SAD', 'NEU', 'FEA', 'DIS', 'SUR']
ini_labels_arr = ['a', 'h', 'sa', 'n', 'f', 'd', 'su'] #[Hyosun] added for SAVEE data file names
if os.path.exists('./data/savee_class_labels_indices.csv') == False:
    os.mkdir('./data/')
    index = []
    mid = []
    for i in range(0,len(labels_arr)): #7 labels for savee
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
        # writer.writerow([i,'/m/07rwj'+str(i).zfill(2), labels_arr[i]])
    print("mid: ", mid)
    print("labels_arr: ", labels_arr)
    print("short_labels_arr: ", short_labels_arr)
    print("ini_labels_arr: ", ini_labels_arr)
    dict = {'index': index, 'mid':mid, 'display_name': labels_arr, 'short_display_name': short_labels_arr, 'ini_labels_arr': ini_labels_arr}
    df = pd.DataFrame(dict)
    print(df)
  
    # saving the dataframe
    df.to_csv('./data/savee_class_labels_indices.csv', index=False)  

    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================
# [Hyosun: create the label description dictionary] 
if os.path.exists('./data/savee_class_labels_indices.csv') == True:    
    label_set = np.loadtxt('./data/savee_class_labels_indices.csv', delimiter=',', dtype='str')
    long_map_label = {}
    short_map_label = {}
    label_map_short = {}
    ini_map_label = {}
    ini_map_long = {}
    for i in range(1, len(label_set)):
        print("[Hyosun] label_set[i][0]: ", label_set[i][0])
        print("[Hyosun] label_set[i][2]: ", label_set[i][2])
        print("[Hyosun] label_set[i][3]: ", label_set[i][3])
        long_map_label[(label_set[i][2])]  = label_set[i][0]  
        short_map_label[(label_set[i][3])] = label_set[i][0]   
        label_map_short[(label_set[i][0])] = label_set[i][3]
        ini_map_label[(label_set[i][4])]   = label_set[i][0]
        ini_map_long[(label_set[i][4])]   = label_set[i][2]
    print(long_map_label)
    print(short_map_label)
    print(label_map_short)    
    print(ini_map_label)   
    print(ini_map_long) 
    print("[Hyosun] create the label description dictionary: using savee_class_labels_indices.csv finished!!")
#[/Hyosun: create the label description dictionary] 

if os.path.exists('./SAVEE/') == False:
    print("Please check the directory './SAVEE/'")

# [Hyosun] create fold1_train.csv, fold1_test.csv, fold1_evaluate.csv
if os.path.exists('./data/evaluation_setup/fold1_train.csv') == False or \
   os.path.exists('./data/evaluation_setup/fold1_test.csv') == False:
    if os.path.exists('./data/evaluation_setup') == False:
        os.mkdir('./data/evaluation_setup/')
    base_dir = './SAVEE/'  

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
    actor_arr = ['DC','JE','JK','KL']
    for ac in actor_arr: # 4 speakers, for independent testing, randomly pick 2 speakers' audio
        for lb in ini_labels_arr:
            ac_lb_list.append(ac+'_'+lb)
    # ac_lb_list = get_immediate_files(base_dir)    
    random.seed(1981)
    testset_actors_arr = random.sample(actor_arr,2) # [Hyosun] 80:20 = train: test
    print("testset_actors_arr: ", testset_actors_arr)
    trainset_actors_arr = list(set(actor_arr) - set(testset_actors_arr))
    print("trainset_actors_arr: ", trainset_actors_arr)

    # get audio files
    random.seed(1981)
    audio_list = []
    audio_list = get_immediate_files(base_dir)

    #[Hyosun] shuffle or not, put the code here (later) [/Hyosun]
    for audio in audio_list:
        # print("[Hyosun] ", base_dir + tr_actor + '/' + tr_audio, " will be added to train_dict_list")
        file_name = audio.split('_') #file_name[0]: actorID among ['DC','JE','JK','KL']
        lb_path = file_name[1].split('.') #lb_path : lb + file_number
        print("lb_path[0]: ", lb_path[0])
        if len(lb_path[0]) == 4:
            lb = (lb_path[0])[0:2]
        else: #len(lb_path) == 3:
            lb = (lb_path[0])[0]
        print("lb: ", lb)
        # print('actor_label_path: {:s} and cur_path: {:s}'.format(ac_lb_path, cur_path))
        label = ini_map_long[lb] #[Hyosun] convert into a corresponding long label description
        print("[Hyosun] label: ", label)

        # train set
        if file_name[0] in trainset_actors_arr:           
            train_dict = {"wav": base_dir   + audio, "labels": label}
            train_dict_list.append(train_dict)
            no_train+=1
            
        # test set
        else: #if file_name[0] in testset_actors_arr:         
            test_dict = {"wav": base_dir   + audio, "labels": label} 
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
