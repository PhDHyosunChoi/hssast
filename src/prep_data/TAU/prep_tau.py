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
tau_path = './TAU-urban-acoustic-scenes-2020-mobile-development/audio/'

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# [Hyosun]
# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
if os.path.exists('./data/TAU') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/TAU/audio/')
else: #os.path.exists('./data/TAU') == True:
    if os.path.exists('./data/TAU/audio_16k/') == False:
        # convert all samples to 16kHZ : #[Hyosun] repositioned 
        print('Now converting all TAU audio to 16kHz')
        #[/Hyosun] repositioned 

        # convert the audio to 16kHz
        base_dir = './data/TAU/'
        os.mkdir('./data/TAU/audio_16k/')
        audio_list = get_immediate_files('./data/TAU/audio')
        for audio in audio_list:
            print('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
            os.system('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        print("[Hyosun] convert the audio to 16kHz: All Finished!!")

# [Hyosun] [done] 이부분 pandas로 다시 만듬, 그래야 바로 아래꺼 돌아가서
# [Hyosun] create labels_info.csv file including only labels information ========================== 
# 위치를 여기로 옮김: 바로 아래 label_map해야해서 
# i.e. if there are 10 labels in the data, then there will be title row + 10 rows = 11 row in the file (very simple file)
if os.path.exists('./data/tau_class_labels_indices.csv') == False:
    labels_arr = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']
    # with open('./data/tau_class_labels_indices.csv', 'w') as out_file:
    #     writer = csv.writer(out_file)
    #     writer.writerow(['index','mid','display_name']) # title row
    index = []
    mid = []

    for i in range(0,len(labels_arr)): #10 labels for tau
        index.append(i)
        mid.append('/m/07rwj'+str(i).zfill(2))
        # writer.writerow([i,'/m/07rwj'+str(i).zfill(2), labels_arr[i]])
    print("mid: ", mid)
    print("labels_arr: ", labels_arr)
    dict = {'index': index, 'mid':mid, 'display_name': labels_arr}
    df = pd.DataFrame(dict)
    print(df)
  
    # saving the dataframe
    df.to_csv('./data/tau_class_labels_indices.csv', index=False)  
    # df.to_csv(r'C:\Users\Admin\Desktop\file3.csv')

    # out_file.close()
    print("[Hyosun] create labels_info.csv file: All Finished!!")
# [/Hyosun] create labels_info.csv file including only labels information =========================

#[Hyosun: create the label description dictionary] 
if os.path.exists('./data/tau_class_labels_indices.csv') == True:
    # label_map = {}
    # with open('./data/tau_class_labels_indices.csv', 'r') as labels_file:
    #     label_set = csv.reader(labels_file)
    #     for i in range(1, len(label_set)):
    #         print("[Hyosun] label_set[i][0]: ", label_set[i][0])
    #         print("[Hyosun] label_set[i][1]: ", label_set[i][1])
    #         label_map[eval(label_set[i][0])] = label_set[i][1] #[Hyosun] I don't know what they are doing here in this line ==> find it!!
    # print(label_map)
    # labels_file.close()
    # print("[Hyosun] label_map using tau_class_labels_indices.csv finished!!")

    label_set = np.loadtxt('./data/tau_class_labels_indices.csv', delimiter=',', dtype='str')
    label_map = {}
    for i in range(1, len(label_set)):
        print("[Hyosun] label_set[i][0]: ", label_set[i][0])
        print("[Hyosun] label_set[i][2]: ", label_set[i][2])
        # label_map[eval(label_set[i][2])] = label_set[i][0] #eval makes an error
        label_map[(label_set[i][2])] = label_set[i][0]     
    print(label_map)
    print("[Hyosun] create the label description dictionary: using tau_class_labels_indices.csv finished!!")
#[/Hyosun: create the label description dictionary] 
# [/Hyosun] 


#[Hyosun] create json files for train_set, (validation_set,) evaluation_set
# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

base_path = os.path.abspath(os.getcwd()) + "/data/TAU/audio_16k/"

if os.path.exists('./data/tau_class_labels_indices.csv') == True:

    # [Hyosun] train_meta: create train_data json file
    train_meta = np.loadtxt('./data/evaluation_setup/fold1_train.csv', delimiter='\t', dtype='str', skiprows=1)
    train_wav_list = []
    for i in range(0, len(train_meta)):
        # print("train_meta[i][1]: ", train_meta[i][1])
        cur_label = label_map[train_meta[i][1]]
        cur_path = train_meta[i][0]
        file_path, cur_path = cur_path.rsplit('/', 1) #[Hyosun]added[/Hyosun]
        # cur_fold = int(train_meta[i][1])

        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        train_wav_list.append(cur_dict)

    # # [Hyosun] test_meta: create test_data json file ==> No labels in the file 
    # # [2023-10-17] implemented to extract labels from the file_names
    # test_meta = np.loadtxt('./data/evaluation_setup/fold1_test.csv', delimiter=' ', dtype='str', skiprows=1)
    # test_wav_list = []
    # for i in range(0, len(test_meta)):
    #     cur_path = test_meta[i]
    #     file_path, cur_path = cur_path.rsplit('/', 1) #[Hyosun]added[/Hyosun]
    #     cur_label, file_name_rest = cur_path.rsplit('-', 1) #[Hyosun]added[/Hyosun]

    #     # /m/07rwj is just a dummy prefix
    #     cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
    #     test_wav_list.append(cur_dict)


    # [Hyosun] eval_meta: create eval_data json file
    eval_meta = np.loadtxt('./data/evaluation_setup/fold1_evaluate.csv', delimiter='\t', dtype='str', skiprows=1)
    eval_wav_list = [] #        eval_wav_list.append(cur_dict)
    for i in range(0, len(eval_meta)):
        cur_label = label_map[eval_meta[i][1]]
        cur_path = eval_meta[i][0]
        file_path, cur_path = cur_path.rsplit('/', 1) #[Hyosun]added[/Hyosun]
        # cur_fold = int(train_meta[i][1])

        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        eval_wav_list.append(cur_dict)
    #[/Hyosun] train_meta, test_meta, eval_meta load and list append Finished.
    # [Hyosun] Later, two sets will be used as below: 
    #   tr_data=./data/datafiles/tau_train_data.json
    #   te_data=./data/datafiles/tau_eval_data.json 
    # [/Hyosun]

    # print('{:d} training samples, {:d} test samples, {:d} evaluation samples'.format(len(train_wav_list), len(test_wav_list), len(eval_wav_list)))
    print('[Hyosun] {:d} training samples, {:d} evaluation samples'.format(len(train_wav_list), len(eval_wav_list)))

    with open('./data/datafiles/tau_train_data_1' +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    # with open('./data/datafiles/tau_test_data_1' +'.json', 'w') as f:
    #     json.dump({'data': test_wav_list}, f, indent=1)

    with open('./data/datafiles/tau_eval_data_1' +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)

    print("[Hyosun] create json files: All Finished!!")
    #[/Hyosun] create json files for train_set, (validation_set,) evaluation_set

    print('[Hyosun] Finished TAU-urban-acoustic-scenes-2020-mobile-development Preparation')
    #[/Hyosun]