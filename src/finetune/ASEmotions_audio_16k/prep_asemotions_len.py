# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_asemotions_len.py : This script cut wav 16k audio files into 2s length 16k audio files

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random

# dataset downloaded from 
# please change it to your ASEmotions dataset path
# [Hyosun] current path: ./src/prep_data/ASEmotions [/Hyosun]
asemotions_path = './data/ASEmotions_audio_16k_intolabels/'
#[Hyosun] use 13 labels : each label has  samples => x13 = total  samples
labels_info_arr = ['01_Boredom_Sigh', '02_Neutral_Calm', '03_Happy_Laugh_Gaggle', '04_Sad_Cry', '05_Angry_Grunt_Frustration', 
                  '06_Fearful_Scream_Panic', '07_Disgust_Dislike_Contempt','08_Surprised_Gasp_Amazed','09_Excited',
                  '10_Pleasure','11_Pain_Groan','12_Disappointment_Disapproval','13_Breath']
labels_arr = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

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

# download ASEmotions 
# dataset provided in 
if os.path.exists('./data/ASEmotions_audio_16k_intolabels') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/ASEmotions_audio_16k_intolabels/')
else: #os.path.exists('./data/ASEmotions') == True:
    if os.path.exists('./data/ASEmotions_audio_16k_intolabels_2s/') == False:
        # cut all samples to the same length in 16kHZ : 
        print('Now converting all ASEmotions wav 16k files to 2s wav 16k files')
        #[/Hyosun] repositioned 

        # convert the various length of 16kHz wav audio to 2s 16kHz wav
        base_dir = './data/ASEmotions_audio_16k_intolabels/'
        out_dir  = './data/ASEmotions_audio_16k_intolabels_2s/'
        os.mkdir('./data/ASEmotions_audio_16k_intolabels_2s/')

        for i in range(0,len(labels_arr)):
            os.mkdir('./data/ASEmotions_audio_16k_intolabels_2s/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            #[Hyosun] cutting into 2s clips, and drop the last clip less than 2s
            # #ffmpeg -i "original.wav" -ss 0 -to 2 "original_0-2s.wav"
            for audio in audio_list_each_label: 
                wav_file = audio.split('.')
                #[Hyosun] for data ASEmotions
                audio_len = get_duration_wave(base_dir + labels_arr[i] + '/' + audio)

                sub_num = 0
                s_sec = 0 
                e_sec = 2
                while 2<= audio_len <=20:   #max 10 => 20 #[Hyosun] drop the last clip less than 2s                                 
                    print('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) +  ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_'+ str(sub_num) + '.wav')
                    os.system('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) + ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_' + str(sub_num) +'.wav')
                    sub_num +=1
                    s_sec +=2 
                    e_sec +=2
                    audio_len-=2        
                #[/Hyosun] for data ASEmotions        
        
        print("[Hyosun] cut the wav 16k files to the same length of wav 16k files: All Finished!!")
      
       
      
    
      
     
   
  
