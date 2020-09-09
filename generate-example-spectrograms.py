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


metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv') 

metadata.head()


# Add Duration Feature

metadata['duration'] = metadata['end'] - metadata['start']
metadata['duration'].describe()


## Listen to Audio From Test File


test_file = 'UrbanSound8K/audio/fold5/100263-2-0-117.wav'

y, sr = librosa.load(test_file)

print('Sample rate: ' + str(sr))

import IPython.display as ipd
ipd.Audio(test_file)


### Sample Waveform

time = np.arange(0, len(y)) / sr

plt.figure(figsize=(18, 6))
plt.plot(time[2000:2110], y[2000:2110])
plt.ylabel('amplitude')
plt.xlabel('time (s)')
plt.savefig('images/sample_wave.png')


## Spectrograms

### Linear Spectrogram

Y = librosa.stft(y)
Ydb = librosa.amplitude_to_db(np.abs(Y), ref=np.max)
plt.figure(figsize=(18, 6))
librosa.display.specshow(Ydb, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar()
plt.savefig('images/example_spectrogram.png')


### Log Spectrogram

plt.figure(figsize=(18, 6))
librosa.display.specshow(Ydb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.savefig(home_directory + 'images/example_log_spectrogram.png')


### Mel Spectrogram

S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=8000)
Sdb = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(18, 6))
librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()
plt.savefig('images/example_mel_spectrogram.png')