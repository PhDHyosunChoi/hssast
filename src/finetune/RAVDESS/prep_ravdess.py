# -*- coding: utf-8 -*-
# @Time    : 08/02/24 19:00 PM
# @Author  : Hyosun Choi
# @Affiliation  : RHUL
# @Email   : bbeyes81@gmail.com
# @File    : prep_ravdess.py

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random       #[Hyosun]

# dataset downloaded from https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
# please change it to your RAVDESS dataset path 
# [Hyosun] current path: ./src/prep_data/ravdess [/Hyosun]
ravdess_path = './RAVDESS/'
Emotions_labels_arr = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised'] #[Hyosun] Emotions model has 7 labels
#RAVDESS labels: This portion of the RAVDESS contains 1440 files: 
        # 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), 
        # vocalizing two lexically-matched statements in a neutral North American accent. 
        # Speech emotions includes [Hyosun] neutral, calm, happy, sad, angry, fearful, surprise, and disgust expressions. => 8 labels, but our model is for 7 labels
        # Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression (from RAVDESS homepage)
labels_arr = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise', 'calm'] #[Hyosun] RAVDESS has 8 labels
short_labels_arr = ['ANG', 'HAP', 'SAD', 'NEU', 'FEA', 'DIS', 'SUR', 'CAL']
ini_labels_arr = ['a', 'h', 'sa', 'n', 'f', 'd', 'su', 'c'] #[Hyosun] added for RAVDESS data file names
def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


# download RAVDESS 
# dataset provided in https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
if os.path.exists('./RAVDESS/') == False:
    print("Please download RAVDESS first :)")
# convert the audio to 16kHz
else: # os.path.exists('./RAVDESS/') == True:
    if os.path.exists('./RAVDESS_audio_16k/') == False:
        # convert all samples to 16kHZ : #[Hyosun] repositioned 
        print('Now converting all audio to 16kHz')
        #[/Hyosun] repositioned 

        # convert the audio to 16kHz
        base_dir = './RAVDESS/'
        out_dir  = './RAVDESS_audio_16k/'
        os.mkdir('./RAVDESS_audio_16k/')

        ac_lb_list = []
        actor_arr = [('Actor_'+str(i).zfill(2)) for i in range(1, 25)]
        # for ac in actor_arr: # Total 24 speakers, for independent testing, randomly pick 5 speakers' audio
        #     for lb in ini_labels_arr:
        #         ac_lb_list.append(ac+'_'+lb)

        for actor in actor_arr:
            os.mkdir('./RAVDESS_audio_16k/'+ actor)
            random.seed(1981)
            audio_list_each_actor= []
            audio_list_each_actor = get_immediate_files(base_dir + actor)

            for audio in audio_list_each_actor:
                # print('sox ' + base_dir   + audio + ' -r 16000 ' + out_dir + ac_lb + '/' + audio)
                # os.system('sox ' + base_dir  + audio + ' -r 16000 ' + out_dir + ac_lb + '/' + audio)
                print('sox ' + base_dir + actor + '/' + audio + ' -r 16000 ' + out_dir + actor + '/' + audio)
                os.system('sox ' + base_dir + actor + '/' + audio + ' -r 16000 ' + out_dir  + actor + '/'+ audio)
        print("[Hyosun] convert the audio to 16kHz: All Finished!!")

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
if os.path.exists('./data/ravdess_class_labels_indices.csv') == False:
    index = []
    mid = []

    for i in range(0,len(labels_arr)): #7 labels for ravdess
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
    print("mid: ", mid)
    print("labels_arr: ", labels_arr)
    print("short_labels_arr: ", short_labels_arr)
    print("ini_labels_arr: ", ini_labels_arr)
    dict = {'index': index, 'mid':mid, 'display_name': labels_arr, 'short_display_name': short_labels_arr, 'ini_labels_arr': ini_labels_arr}
    df = pd.DataFrame(dict)
    print(df)
  
    # saving the dataframe
    df.to_csv('./data/ravdess_class_labels_indices.csv', index=False)  

    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================

# [Hyosun: create the label description dictionary] 
if os.path.exists('./data/ravdess_class_labels_indices.csv') == True:    
    label_set = np.loadtxt('./data/ravdess_class_labels_indices.csv', delimiter=',', dtype='str')
    long_map_label = {}
    short_map_label = {}
    label_map_short = {}
    ini_map_label = {}  
    ini_map_long = {}
    label_map_long = {} # [Hyosun] added : RAVDESS
    for i in range(1, len(label_set)):
        print("[Hyosun] label_set[i][0]: ", label_set[i][0])
        print("[Hyosun] label_set[i][2]: ", label_set[i][2])
        print("[Hyosun] label_set[i][3]: ", label_set[i][3])
        long_map_label[(label_set[i][2])] = label_set[i][0]     
        short_map_label[(label_set[i][3])] = label_set[i][0]   
        label_map_short[(label_set[i][0])] = label_set[i][3]       
        ini_map_label[(label_set[i][4])]   = label_set[i][0]
        ini_map_long[(label_set[i][4])]   = label_set[i][2]
        label_map_long[(label_set[i][0])] = label_set[i][2] #[Hyosun] added : RAVDESS
    print(long_map_label)
    print(short_map_label)
    print(label_map_short)
    print(ini_map_label)   
    print(ini_map_long)
    print(label_map_long) #[Hyosun] added : RAVDESS
    print("[Hyosun] create the label description dictionary: using ravdess_class_labels_indices.csv finished!!")
#[/Hyosun: create the label description dictionary] 

#[Hyosun] create json files for train_set, (validation_set,) evaluation_set
base_path = os.path.abspath(os.getcwd()) + "/RAVDESS_audio_16k/"

if os.path.exists('./data/ravdess_class_labels_indices.csv') == True:
    # fix bug: generate an empty directory to save json files
    if os.path.exists('./data/datafiles') == False:
        os.mkdir('./data/datafiles')
        
    # [Hyosun] train_meta: create train_data json file
    train_meta = np.loadtxt('./data/evaluation_setup/fold1_train.csv', delimiter='\t', dtype='str', skiprows=1)
    train_wav_list = []
    for i in range(0, len(train_meta)):
        # print("train_meta[i][1]: ", train_meta[i][1])
        cur_label = long_map_label[train_meta[i][1]] #[Hyosun] long label -> label
        cur_path = train_meta[i][0]
        file_name, actor_path, cur_path = cur_path.rsplit('/', 2)

        # /m/07rwj is just a dummy prefix
        # cur_dict = {"wav": base_path + ac_lb_path + "/"  + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        cur_dict = {"wav": base_path  + actor_path + '/' + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        train_wav_list.append(cur_dict)

    # [Hyosun] test_meta: create test_data json file
    test_meta = np.loadtxt('./data/evaluation_setup/fold1_test.csv', delimiter='\t', dtype='str', skiprows=1)
    test_wav_list = []
    for i in range(0, len(test_meta)):
        # print("test_meta[i][1]: ", test_meta[i][1])
        cur_label = long_map_label[test_meta[i][1]]        
        cur_path = test_meta[i][0]
        file_path, actor_path, cur_path = cur_path.rsplit('/', 2)

       # /m/07rwj is just a dummy prefix
        # cur_dict = {"wav": base_path + ac_lb_path + "/" + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        cur_dict = {"wav": base_path  + actor_path + '/' + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        test_wav_list.append(cur_dict)
    #[/Hyosun] train_meta, test_meta, eval_meta load and list append Finished.
    # [Hyosun] Later, two sets will be used as below: 
    #   tr_data=./data/datafiles/emotions_train_data_1.json
    #   te_data=./data/datafiles/emotions_test_data_1.json 
    # [/Hyosun]

    # print('{:d} training samples, {:d} test samples, {:d} evaluation samples'.format(len(train_wav_list), len(test_wav_list), len(eval_wav_list)))
    print('[Hyosun] total {:d} training samples, {:d} testing samples'.format(len(train_wav_list), len(test_wav_list)))

    with open('./data/datafiles/ravdess_train_data_1' +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/ravdess_test_data_1' +'.json', 'w') as f:
        json.dump({'data': test_wav_list}, f, indent=1)

    # with open('./data/datafiles/emotions_eval_data_1' +'.json', 'w') as f:
    #     json.dump({'data': eval_wav_list}, f, indent=1)

    print("[Hyosun] create json files: All Finished!!")
    #[/Hyosun] create json files for train_set, (validation_set,) evaluation_set

    print('[Hyosun] Finished RAVDESS Data Preparation')