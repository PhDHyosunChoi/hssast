# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_iurbansound_len.py : This script cut wav audio files into 2s length wav audio files

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random

# dataset downloaded from https://zenodo.org/records/1213793
# please change it to your Isolate Urban Sound dataset path
# [Hyosun] current path: ./src/prep_data/background [/Hyosun]
asemotions_path = './data/iurban/'
#[Hyosun] use 10 labels : each label has  samples => x10 = total  samples
# labels_arr = ['bell', 'bird', 'bird_event', #'broom', 
#               'carhorn', 'citycar', 'citystep', 'construction', 'constructionsite', 'coughing', 'crowd', 'dog', 
#               'doorbell', 'doorcar', 'doorhouse', 'fountain', 'park', 'plane', 'rain', 'roadcar', 'schoolyard', 'siren', 'stepcity', 'steppark', 'stopcar', 
#               'storm', 'streetnoise', 'suitcase', 'traffic', 'train', 'tram', 'truck', 'ventilation', 'voice', 'windtree']
labels_arr = ['bird', 'construction', 'crowd', 'fountain', 'park', 
              'rain', 'schoolyard', 'traffic', 'ventilation', 'windtree']
# labels_arr = ['01', '02', '03', '04', '05', '06'] #, '07', '08', '09', '10', '11', '12'] #, '13']

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

#[Hyosun] for data BirdSong
import wave
def get_duration_wave(file_path):
   with wave.open(file_path, 'r') as audio_file:
      frame_rate = audio_file.getframerate()
      n_frames = audio_file.getnframes()
      duration = n_frames / float(frame_rate)
      return duration
#[/Hyosun] for data BirdSong

# download Isolated Urban Sound 
# dataset provided in 
if os.path.exists('./data/iurban') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/iurban/')
else: #os.path.exists('./data/iurban') == True:
    if os.path.exists('./data/iurban_2s/') == False:
        # cut all samples to the same length: 
        print('Now converting all wav files to 2s wav files')
        #[/Hyosun] repositioned 

        # convert the various length of wav audio to 2s wav
        base_dir = './data/iurban/'
        out_dir  = './data/iurban_2s/'
        os.mkdir('./data/iurban_2s/')

        for i in range(0,len(labels_arr)):
            os.mkdir('./data/iurban_2s/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            #[Hyosun] cutting into 2s clips, and drop the last clip less than 2s
            # #ffmpeg -i "original.wav" -ss 0 -to 2 "original_0-2s.wav"
            for audio in audio_list_each_label: 
                wav_file = audio.split('.')
                #[Hyosun] for data Isolated Urban Sound
                audio_len = get_duration_wave(base_dir + labels_arr[i] + '/' + audio)

                sub_num = 0
                s_sec = 0 
                e_sec = 2
                while 2<= audio_len:   #[Hyosun] drop the last clip less than 2s                                 
                    print('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) +  ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_'+ str(sub_num) + '.wav')
                    os.system('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) + ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_' + str(sub_num) +'.wav')
                    sub_num +=1
                    s_sec +=2 
                    e_sec +=2
                    audio_len-=2        
                #[/Hyosun] for data Isolated Urban Sound      
        
        print("[Hyosun] cut the wav files to the same length of wav files: All Finished!!")