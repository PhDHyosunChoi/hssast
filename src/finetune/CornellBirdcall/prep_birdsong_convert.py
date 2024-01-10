# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : Hyosun Choi
# @Affiliation  : 
# @Email   : bbeyes81@gmail.com
# @File    : prep_birdsong_convert.py : This script converts mp3 audio files into wav audio files

import numpy as np
import json
import os
import csv #[Hyosun]
import pandas as pd #[Hyosun]
import random

# dataset downloaded from https://www.kaggle.com/c/birdsong-recognition/data
# please change it to your BirdSong dataset path
# [Hyosun] current path: ./src/prep_data/BirdSong [/Hyosun]
birdsong_path = './data/BirdSong_mp3/'
#[Hyosun] use 50 labels birds : each label has 100 samples => 100x50 = total 5,000 samples
labels_arr = ['aldfly', 'amecro', 'amegfi', 'amepip', 'amered', 'amerob', 'annhum', 'astfly', 'balori', 'banswa', 
              'barswa', 'bewwre', 'bkbwar', 'bkcchi', 'bkhgro', 'bkpwar', 'blkpho', 'blugrb1', 'blujay', 'bnhcow', 
              'boboli', 'brdowl', 'brespa', 'brncre', 'brnthr', 'btnwar', 'buggna', 'buhvir', 'bulori', 'bushti', 
              'cacwre', 'cangoo', 'canwar', 'canwre', 'carwre', 'caster1', 'chispa', 'chswar', 'comgra', 'comrav', 
              'comred', 'comter', 'comyel', 'daejun', 'dowwoo', 'easmea', 'eastow', 'eawpew', 'eucdov', 'evegro' ] 


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# downlooad BirdSong 
# dataset provided in https://www.kaggle.com/c/birdsong-recognition/data
if os.path.exists('./data/BirdSong_mp3') == False:
    # esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # wget.download(esc50_url, out='./data/')
    # with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./data/')
    # os.remove('./data/ESC-50-master.zip')

    os.mkdir('./data/BirdSong_mp3/')
else: #os.path.exists('./data/BirdSong') == True:
    if os.path.exists('./data/BirdSong/') == False:
        # convert all samples to 16kHZ : #[Hyosun] repositioned 
        print('Now converting all BirdSong mp3 files to wav files')
        #[/Hyosun] repositioned 

        # convert the audio to 16kHz
        base_dir = './data/BirdSong_mp3/'
        out_dir  = './data/BirdSong/'
        os.mkdir('./data/BirdSong/')
        # audio_list = get_immediate_files('./data/BirdSong_mp3')
        for i in range(0,len(labels_arr)):
            os.mkdir('./data/BirdSong/'+labels_arr[i]) #[Hyosun] no directory error fix
            random.seed(1981)
            audio_list_each_label = []
            audio_list_each_label = get_immediate_files(base_dir + labels_arr[i])

            for audio in audio_list_each_label:
                wav_file = audio.split('.')
                print('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '.wav')
                os.system('ffmpeg -i ' + base_dir + labels_arr[i] + '/' + audio + ' ' + out_dir + labels_arr[i] + '/' + wav_file[0] + '.wav')
        print("[Hyosun] convert the mp3 files to wav files: All Finished!!")
