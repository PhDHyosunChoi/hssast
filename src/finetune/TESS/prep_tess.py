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
Emotions_labels_arr = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised'] #[Hyosun] Emotions model has 7 labels
labels_arr = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'ps'] #[Hyosun] TESS has 7 labels: 'ps' means 'pleasant surprise'
short_labels_arr = ['ANG', 'HAP', 'SAD', 'NEU', 'FEA', 'DIS', 'PSU']
def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


# download TESS 
# dataset provided in https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download
if os.path.exists('./TESS/') == False:
    print("Please download TESS first :)")
# convert the audio to 16kHz
else: # os.path.exists('./TESS/') == True:
    if os.path.exists('./TESS_audio_16k/') == False:
        # convert all samples to 16kHZ : #[Hyosun] repositioned 
        print('Now converting all audio to 16kHz')
        #[/Hyosun] repositioned 

        # convert the audio to 16kHz
        base_dir = './TESS/'
        out_dir  = './TESS_audio_16k/'
        os.mkdir('./TESS_audio_16k/')
        actor_label_list = []
        for ac in ['YAF', 'OAF']:
            for lb in labels_arr:
                actor_label_list.append(ac+'_'+lb)

        # actor_arr = list([str(i) for i in range(1001,1092)])
        for ac_lb in actor_label_list:
            os.mkdir('./TESS_audio_16k/'+ ac_lb)
            random.seed(1981)
            audio_list_each_ac_lb= []
            audio_list_each_ac_lb = get_immediate_files(base_dir + ac_lb)

            for audio in audio_list_each_ac_lb:
                print('sox ' + base_dir + ac_lb + '/' + audio + ' -r 16000 ' + out_dir + ac_lb + '/' + audio)
                os.system('sox ' + base_dir + ac_lb + '/' + audio + ' -r 16000 ' + out_dir + ac_lb + '/' + audio)
        print("[Hyosun] convert the audio to 16kHz: All Finished!!")

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
if os.path.exists('./data/tess_class_labels_indices.csv') == False:
    index = []
    mid = []

    for i in range(0,len(labels_arr)): #7 labels for tess
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
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

#[Hyosun] create json files for train_set, (validation_set,) evaluation_set
base_path = os.path.abspath(os.getcwd()) + "/TESS_audio_16k/"

if os.path.exists('./data/tess_class_labels_indices.csv') == True:
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
        file_path, ac_lb_path, cur_path = cur_path.rsplit('/', 2) 
        print('actor_label_path: {:s} and cur_path: {:s}'.format(ac_lb_path, cur_path))
        # cur_fold = int(train_meta[i][1])

        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + ac_lb_path + "/"  + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        train_wav_list.append(cur_dict)

    # [Hyosun] test_meta: create test_data json file
    test_meta = np.loadtxt('./data/evaluation_setup/fold1_test.csv', delimiter='\t', dtype='str', skiprows=1)
    test_wav_list = []
    for i in range(0, len(test_meta)):
        # print("test_meta[i][1]: ", test_meta[i][1])
        cur_label = long_map_label[test_meta[i][1]]        
        cur_path = test_meta[i][0]
        file_path, ac_lb_path, cur_path = cur_path.rsplit('/', 2)
        print('actor_label_path: {:s} and cur_path: {:s}'.format(ac_lb_path, cur_path))

        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + ac_lb_path + "/" + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        test_wav_list.append(cur_dict)
    #[/Hyosun] train_meta, test_meta, eval_meta load and list append Finished.
    # [Hyosun] Later, two sets will be used as below: 
    #   tr_data=./data/datafiles/emotions_train_data_1.json
    #   te_data=./data/datafiles/emotions_test_data_1.json 
    # [/Hyosun]

    # print('{:d} training samples, {:d} test samples, {:d} evaluation samples'.format(len(train_wav_list), len(test_wav_list), len(eval_wav_list)))
    print('[Hyosun] total {:d} training samples, {:d} testing samples'.format(len(train_wav_list), len(test_wav_list)))

    with open('./data/datafiles/tess_train_data_1' +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/tess_test_data_1' +'.json', 'w') as f:
        json.dump({'data': test_wav_list}, f, indent=1)

    # with open('./data/datafiles/emotions_eval_data_1' +'.json', 'w') as f:
    #     json.dump({'data': eval_wav_list}, f, indent=1)

    print("[Hyosun] create json files: All Finished!!")
    #[/Hyosun] create json files for train_set, (validation_set,) evaluation_set

    print('[Hyosun] Finished TESS Data Preparation')