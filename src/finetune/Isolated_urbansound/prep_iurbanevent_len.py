# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_iurbanevent_len.py : This script cut wav audio files into 0.4s length wav audio files

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random

# dataset downloaded from https://zenodo.org/records/1213793
# please change it to your Isolate Urban Sound Event dataset path
# [Hyosun] current path: ./src/prep_data/event [/Hyosun]
asemotions_path = './data/event/'
#[Hyosun] use 11 labels : 
labels_arr = [#'bell', 
              'bird', #'broom', #'carhorn', 
              'citycar', #'citystep', 
              'construction', #'coughing', 
              'dog', #'doorbell', 'doorcar', 'doorhouse', 
              'plane', 'roadcar', 'siren', 
              'stepcity', 'steppark', #[Hyosun] combined into one folder "step" later
              'stopcar', #'storm', #'streetnoise', 'suitcase', 
              'train', #'tram', 
              'truck'] #, 'voice']
# labels_arr = ['bird', 'construction', 'crowd', 'fountain', 'park', 
#               'rain', 'schoolyard', 'traffic', 'ventilation', 'windtree']

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

# download Isolated Urban Event 
# dataset provided in https://zenodo.org/records/1213793
if os.path.exists('./data/event') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/event/')
else: #os.path.exists('./data/event') == True:
    if os.path.exists('./data/event_04s/') == False:
        # cut all samples to the same length: 
        print('Now converting all wav files to 0.4s wav files')
        #[/Hyosun] repositioned 

        # convert the various length of wav audio to 0.4s wav
        base_dir = './data/event/'
        out_dir  = './data/event_04s/'
        os.mkdir('./data/event_04s/')

        for i in range(0,len(labels_arr)):
            os.mkdir('./data/event_04s/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            #[Hyosun] cutting into 0.4s clips, and drop the last clip less than 0.4s
            # #ffmpeg -i "original.wav" -ss 0 -to 2 "original_0-2s.wav"
            for audio in audio_list_each_label: 
                wav_file = audio.split('.')
                #[Hyosun] for data Isolated Urban Event Sound
                audio_len = get_duration_wave(base_dir + labels_arr[i] + '/' + audio)

                sub_num = 0
                s_sec = 0 
                e_sec = 0.4
                while 0.4<= audio_len:   #[Hyosun] drop the last clip less than 0.4s                                 
                    print('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) +  ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_'+ str(sub_num) + '.wav')
                    os.system('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) + ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_' + str(sub_num) +'.wav')
                    sub_num +=1
                    s_sec +=0.4 
                    e_sec +=0.4
                    audio_len-=0.4        
                #[/Hyosun] for data Isolated Urban Event Sound      
        
        print("[Hyosun] cut the wav files to the same length of wav files: All Finished!!")