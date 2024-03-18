# -*- coding: utf-8 -*-
# @Time    : 14/02/2024
# @Author  : Hyosun Choi
# @Affiliation  : RHUL
# @Email   : bbeyes81@gmail.com
# @File    : prep_orievent.py

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random       #[Hyosun]

# dataset downloaded from https://zenodo.org/records/1213793
# please change it to your IurbanEvent dataset path
# [Hyosun] current path: ./src/prep_data/IurbanEvent [/Hyosun]
event_path = './data/event/'
#[Hyosun] use 11 labels  
labels_arr = [#'bell', 
              'bird', #'broom', #'carhorn', 
              'citycar', #'citystep', 
              'construction', #'coughing', 
              'dog', #'doorbell', 'doorcar', 'doorhouse', 
              'plane', 'roadcar', 'siren', 
              'step', #[Hyosun] combined into one folder "step" from 'stepcity' and 'steppark'
              'stopcar', #'storm', #'streetnoise', 'suitcase', 
              'train', #'tram', 
              'truck'] #, 'voice']


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


# downlooad Isolated Urban Event 
# dataset provided in https://zenodo.org/records/1213793
if os.path.exists('./data/event') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/event/')
else: #os.path.exists('./data/event') == True:
    if os.path.exists('./data/event_audio_16k/') == False:
        # convert all samples to 16kHZ : #[Hyosun] repositioned 
        print('Now converting all event audio to 16kHz')
        #[/Hyosun] repositioned 

        # convert the audio to 16kHz
        base_dir = './data/event/'
        out_dir  = './data/event_audio_16k/'
        os.mkdir('./data/event_audio_16k/')
        audio_list = get_immediate_files('./data/event')
        for i in range(0,len(labels_arr)):
            os.mkdir('./data/event_audio_16k/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            for audio in audio_list_each_label:
                print('sox ' + base_dir + labels_arr[i] + '/' + audio + ' -r 16000 ' + out_dir + labels_arr[i] + '/' + audio)
                os.system('sox ' + base_dir + labels_arr[i] + '/' + audio + ' -r 16000 ' + out_dir + labels_arr[i] + '/' + audio)
        print("[Hyosun] convert the audio to 16kHz: All Finished!!")

# [Hyosun] create labels_info.csv file including only labels information ========================== 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
if os.path.exists('./data/event_class_labels_indices.csv') == False:
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
    df.to_csv('./data/event_class_labels_indices.csv', index=False)  

    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================

# [Hyosun: create the label description dictionary] 
if os.path.exists('./data/event_class_labels_indices.csv') == True:    
    label_set = np.loadtxt('./data/event_class_labels_indices.csv', delimiter=',', dtype='str')
    label_map = {}
    for i in range(1, len(label_set)):
        print("[Hyosun] label_set[i][0]: ", label_set[i][0])
        print("[Hyosun] label_set[i][2]: ", label_set[i][2])
        label_map[(label_set[i][2])] = label_set[i][0]     
    print(label_map)
    print("[Hyosun] create the label description dictionary: using event_class_labels_indices.csv finished!!")
#[/Hyosun: create the label description dictionary] 

# [Hyosun] create evaluation_setup/fold1_train.csv, fold1_test.csv, fold1_evaluate.csv
if os.path.exists('./data/evaluation_setup/fold1_train.csv') == False or \
os.path.exists('./data/evaluation_setup/fold1_test.csv') == False:
    if os.path.exists('./data/evaluation_setup') == False:
        os.mkdir('./data/evaluation_setup/')
    # get audio file names in each 50 labels folder
    base_dir = './data/event_audio_16k/'
    import random
    train_dict_list = []
    test_dict_list = []
    labels = []
    train_rate = 0.8 #[Hyosun] trainset vs testset = 8:2
    for i in range(0,len(labels_arr)):
        random.seed(1981)
        audio_list_each_label = []
        audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

        # #[Hyosun] for max samples 550 per label
        # if len(audio_list_each_label) > 550:
        #     # audio_list_each_label = []
        #     audio_list_each_label = audio_list_each_label[0:550]
        #     print("[Hyosun] max 550 samples per label- ", labels_arr[i], ": len(audio_list_each_label) = ",len(audio_list_each_label))
        # #[/Hyosun] for max samples 550 per label  
        no_train = 0
        no_test = 0
        #[Hyosun] shuffle or not, put the code here (later) => no need, as we do it in the main codes [/Hyosun]
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
    print("[Hyosun] create evaluation_setup/train and test files: All Finished!!")
# [/Hyosun] create fold1_train.csv, fold1_test.csv, fold1_evaluate.csv


#[Hyosun] create datafiles/json files for train_set, (validation_set,) evaluation_set
# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

    base_path = os.path.abspath(os.getcwd()) + "/data/event_audio_16k/"

    if os.path.exists('./data/event_class_labels_indices.csv') == True and \
    os.path.exists('./data/evaluation_setup') == True :

        # [Hyosun] train_meta: create train_data json file
        train_meta = np.loadtxt('./data/evaluation_setup/fold1_train.csv', delimiter='\t', dtype='str', skiprows=1)
        train_wav_list = []
        for i in range(0, len(train_meta)):
            # print("train_meta[i][1]: ", train_meta[i][1])
            cur_label = label_map[train_meta[i][1]]
            cur_path = train_meta[i][0]
            file_path, label_path, cur_path = cur_path.rsplit('/', 2) 
            # print('label_path: {:s} and cur_path: {:s}'.format(label_path, cur_path))
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
            # print('label_path: {:s} and cur_path: {:s}'.format(label_path, cur_path))

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

        with open('./data/datafiles/event_train_data_1' +'.json', 'w') as f:
            json.dump({'data': train_wav_list}, f, indent=1)

        with open('./data/datafiles/event_test_data_1' +'.json', 'w') as f:
            json.dump({'data': test_wav_list}, f, indent=1)

        # with open('./data/datafiles/emotions_eval_data_1' +'.json', 'w') as f:
        #     json.dump({'data': eval_wav_list}, f, indent=1)

        print("[Hyosun] create datafiles/json files: All Finished!!")
#[/Hyosun] create json files for train_set, (validation_set,) evaluation_set

print('[Hyosun] Finished event Data Preparation')