# Set Home Directory (modify as needed)

home_directory = ''

# Import Libraries


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

import os
import time
import random

from PIL import Image


## File Processing

### Create Directories

numpy_save_directory = home_directory + 'UrbanSound8K/processed_np/' 
sub_dirs = ['fold' + str(x) for x in np.arange(1,11)]


parent_directory = home_directory + 'UrbanSound8K/audio/'
numpy_save_directory = home_directory + 'UrbanSound8K/processed_np/'
model_save_directory = home_directory + 'models/'

if not os.path.exists(numpy_save_directory):
    os.makedirs(numpy_save_directory)
    
sub_dirs = ['fold' + str(x) for x in np.arange(1,11)]

for sub_dir in sub_dirs:
    if not os.path.exists(numpy_save_directory + sub_dir):
        os.makedirs(numpy_save_directory + sub_dir)
        
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)


### Define Mel Spectrogram Parameters


melspec_params = {
    'n_mels': 128,
    'duration': 4*22050,
    'hop_length': 347*4,
    'n_fft': 128*20,
    'fmin': 20
}


### Define Helper Functions


def load_audio(params, file_path):
    y, sr = librosa.load(parent_directory + sub_dir + '/' + file_list[i])

    # clip silence
    yt, index = librosa.effects.trim(y, top_db=60)       

    # pad to a length of 4s
    if len(yt) > params['duration']:
        yt = yt[:params['duration']]
    else:
        padding = params['duration'] - len(yt)
        offset = padding // 2
        yt = np.pad(yt, (offset, params['duration'] - len(yt) - offset), 'constant')
    
    return yt, sr

def create_melspec(params, audio_data, sampling_rate):
    S = librosa.feature.melspectrogram(audio_data, 
                                       sr=sampling_rate, 
                                       n_mels=params['n_mels'],
                                       hop_length=params['hop_length'],
                                       n_fft=params['n_fft'],
                                       fmin=params['fmin'],
                                       fmax=(sampling_rate // 2))
    Sb = librosa.power_to_db(S, ref=np.max)
    Sb = Sb.astype(np.float32)
    
    return Sb
        
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V  

def display_melspec(params, mels, sampling_rate): 
    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=sampling_rate, hop_length=params['hop_length'],
                             fmin=params['fmin'], fmax=(sampling_rate // 2))
    plt.colorbar()
    plt.show()


### Process WAV Files

for sub_dir in sub_dirs:
    file_list = os.listdir(parent_directory + sub_dir)
    file_list = [x for x in file_list if 'wav' in x]
    file_labels = [int(x.split('-')[1]) for x in file_list]
    print('processing {}'.format(sub_dir))
    
    for i in range(0, len(file_list)):
        y, sr = load_audio(melspec_params, parent_directory + sub_dir + '/' + file_list[i])
        melspec = create_melspec(melspec_params, y, sr)
        melspec_color = mono_to_color(melspec)
        np.savez(numpy_save_directory + sub_dir + '/' + file_list[i][:-4], melspec_color)

