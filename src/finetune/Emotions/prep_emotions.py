# -*- coding: utf-8 -*-
# @Time    : 09/11/2023
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_emotions.py

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random       #[Hyosun]

# dataset downloaded from https://zenodo.org/record/3819968
# please change it to your Emotions dataset path
# [Hyosun] current path: ./src/prep_data/Emotions [/Hyosun]
emotions_path = './data/Emotions/'
labels_arr = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised']

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


# downlooad Emotions
# dataset provided in https:
if os.path.exists('./data/Emotions') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/Emotions/')
else: #os.path.exists('./data/Emotions') == True:
    if os.path.exists('./data/Emotions_audio_16k/') == False:
        # convert all samples to 16kHZ : #[Hyosun] repositioned 
        print('Now converting all emotions audio to 16kHz')
        #[/Hyosun] repositioned 

        # convert the audio to 16kHz
        base_dir = './data/Emotions/'
        out_dir  = './data/Emotions_audio_16k/'
        os.mkdir('./data/Emotions_audio_16k/')
        audio_list = get_immediate_files('./data/Emotions')
        for i in range(0,len(labels_arr)):
            os.mkdir('./data/Emotions_audio_16k/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            for audio in audio_list_each_label:
                print('sox ' + base_dir + labels_arr[i] + '/' + audio + ' -r 16000 ' + out_dir + labels_arr[i] + '/' + audio)
                os.system('sox ' + base_dir + labels_arr[i] + '/' + audio + ' -r 16000 ' + out_dir + labels_arr[i] + '/' + audio)
        print("[Hyosun] convert the audio to 16kHz: All Finished!!")

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
if os.path.exists('./data/emotions_class_labels_indices.csv') == False:
    index = []
    mid = []

    for i in range(0,len(labels_arr)): #7 labels for emotions
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
    print("mid: ", mid)
    print("labels_arr: ", labels_arr)
    dict = {'index': index, 'mid':mid, 'display_name': labels_arr}
    df = pd.DataFrame(dict)
    print(df)
  
    # saving the dataframe
    df.to_csv('./data/emotions_class_labels_indices.csv', index=False)  

    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================

# [Hyosun: create the label description dictionary] 
if os.path.exists('./data/emotions_class_labels_indices.csv') == True:    
    label_set = np.loadtxt('./data/emotions_class_labels_indices.csv', delimiter=',', dtype='str')
    label_map = {}
    for i in range(1, len(label_set)):
        print("[Hyosun] label_set[i][0]: ", label_set[i][0])
        print("[Hyosun] label_set[i][2]: ", label_set[i][2])
        label_map[(label_set[i][2])] = label_set[i][0]     
    print(label_map)
    print("[Hyosun] create the label description dictionary: using emotions_class_labels_indices.csv finished!!")
#[/Hyosun: create the label description dictionary] 



#[Hyosun] create json files for train_set, (validation_set,) evaluation_set
# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

base_path = os.path.abspath(os.getcwd()) + "/data/Emotions_audio_16k/"

if os.path.exists('./data/emotions_class_labels_indices.csv') == True:

    # [Hyosun] train_meta: create train_data json file
    train_meta = np.loadtxt('./data/evaluation_setup/fold1_train.csv', delimiter='\t', dtype='str', skiprows=1)
    train_wav_list = []
    for i in range(0, len(train_meta)):
        # print("train_meta[i][1]: ", train_meta[i][1])
        cur_label = label_map[train_meta[i][1]]
        cur_path = train_meta[i][0]
        file_path, label_path, cur_path = cur_path.rsplit('/', 2) 
        print('label_path: {:s} and cur_path: {:s}'.format(label_path, cur_path))
        # cur_fold = int(train_meta[i][1])


        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + label_path + "/"  + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        train_wav_list.append(cur_dict)

    # [Hyosun] test_meta: create test_data json file
    test_meta = np.loadtxt('./data/evaluation_setup/fold1_test.csv', delimiter='\t', dtype='str', skiprows=1)
    test_wav_list = []
    for i in range(0, len(test_meta)):
        # print("test_meta[i][1]: ", test_meta[i][1])
        cur_label = label_map[test_meta[i][1]]        
        cur_path = test_meta[i][0]
        file_path, label_path, cur_path = cur_path.rsplit('/', 2)
        print('label_path: {:s} and cur_path: {:s}'.format(label_path, cur_path))

        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + label_path + "/" + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        test_wav_list.append(cur_dict)
    #[/Hyosun] train_meta, test_meta, eval_meta load and list append Finished.
    # [Hyosun] Later, two sets will be used as below: 
    #   tr_data=./data/datafiles/emotions_train_data_1.json
    #   te_data=./data/datafiles/emotions_test_data_1.json 
    # [/Hyosun]

    # print('{:d} training samples, {:d} test samples, {:d} evaluation samples'.format(len(train_wav_list), len(test_wav_list), len(eval_wav_list)))
    print('[Hyosun] total {:d} training samples, {:d} testing samples'.format(len(train_wav_list), len(test_wav_list)))

    with open('./data/datafiles/emotions_train_data_1' +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/emotions_test_data_1' +'.json', 'w') as f:
        json.dump({'data': test_wav_list}, f, indent=1)

    # with open('./data/datafiles/emotions_eval_data_1' +'.json', 'w') as f:
    #     json.dump({'data': eval_wav_list}, f, indent=1)

    print("[Hyosun] create json files: All Finished!!")
    #[/Hyosun] create json files for train_set, (validation_set,) evaluation_set

    print('[Hyosun] Finished Emotions Data Preparation')