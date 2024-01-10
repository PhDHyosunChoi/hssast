# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_birdsong_len.py : This script cut wav 16k audio files into 10s length 16k audio files

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random

# dataset downloaded from https://www.kaggle.com/c/birdsong-recognition/data
# please change it to your BirdSong dataset path
# [Hyosun] current path: ./src/prep_data/BirdSong [/Hyosun]
birdsong_path = './data/BirdSong_audio_16k/'
#[Hyosun] use 50 labels birds : each label has 100 samples => 100x50 = total 5,000 samples
labels_arr = ['aldfly', 'amecro', 'amegfi', 'amepip', 'amered', 'amerob', 'annhum', 'astfly', 'balori', 'banswa', 
              'barswa', 'bewwre', 'bkbwar', 'bkcchi', 'bkhgro', 'bkpwar', 'blkpho', 'blugrb1', 'blujay', 'bnhcow', 
              'boboli', 'brdowl', 'brespa', 'brncre', 'brnthr', 'btnwar', 'buggna', 'buhvir', 'bulori', 'bushti', 
              'cacwre', 'cangoo', 'canwar', 'canwre', 'carwre', 'caster1', 'chispa', 'chswar', 'comgra', 'comrav', 
              'comred', 'comter', 'comyel', 'daejun', 'dowwoo', 'easmea', 'eastow', 'eawpew', 'eucdov', 'evegro' ] 


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

# download BirdSong 
# dataset provided in https://www.kaggle.com/c/birdsong-recognition/data
if os.path.exists('./data/BirdSong_audio_16k') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/BirdSong_audio_16k/')
else: #os.path.exists('./data/BirdSong') == True:
    if os.path.exists('./data/BirdSong_audio_16k_10s/') == False:
        # cut all samples to the same length in 16kHZ : 
        print('Now converting all BirdSong wav 16k files to 10s wav 16k files')
        #[/Hyosun] repositioned 

        # convert the various length of 16kHz wav audio to 10s 16kHz wav
        base_dir = './data/BirdSong_audio_16k/'
        out_dir  = './data/BirdSong_audio_16k_10s/'
        os.mkdir('./data/BirdSong_audio_16k_10s/')

        for i in range(0,len(labels_arr)):
            os.mkdir('./data/BirdSong_audio_16k_10s/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            #[Hyosun] cutting into 10s clips, and drop the last clip less than 10s
            # #ffmpeg -i "original.wav" -ss 0 -to 10 "original_0-10s.wav"
            for audio in audio_list_each_label: 
                wav_file = audio.split('.')
                #[Hyosun] for data BirdSong
                audio_len = get_duration_wave(base_dir + labels_arr[i] + '/' + audio)
                # print(f"Duration: {audio_len:.2f} seconds")
                # if audio_len < 10: #[Hyosun] we ignore the audio less than 10s
                #     continue #[Hyosun] to the next audio in for-loop
                sub_num = 0
                s_sec = 0 
                e_sec = 10
                while audio_len >= 10:  #[Hyosun] drop the last clip less than 10s                                 
                    print('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) +  ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_'+ str(sub_num) + '.wav')
                    os.system('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' -ss ' + str(s_sec) + ' -to ' + str(e_sec) + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '_' + str(sub_num) +'.wav')
                    sub_num +=1
                    s_sec +=10 
                    e_sec +=10
                    audio_len-=10        
                #[/Hyosun] for data BirdSong        
        
        print("[Hyosun] cut the wav 16k files to the same length of wav 16k files: All Finished!!")
      
       
      
    
      
     
   
  
