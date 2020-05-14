#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io.wavfile
import librosa
import glob
import numpy as np 
from IPython.display import Audio
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


root_dir = './Dataset/'
base_train_path = root_dir + 'training/'
base_val_path = root_dir + 'validation/'
base_noise_path = root_dir + '_background_noise_/'
train_val_classes = os.listdir(base_train_path)

train_paths = []
val_paths = []

for x in train_val_classes:
    train_paths.extend(glob.glob(base_train_path+x+'/*'))
    val_paths.extend(glob.glob(base_val_path+x+'/*'))


# In[3]:


def plot_spectrogram(spec,sample_rate,num_samples):
    plt.figure(figsize=(15,8))
    spec_plotted = plt.imshow(spec,origin='lower',cmap='plasma')
    num_samples_per_frame = spec.shape[0]*2
    yticks = np.linspace(0,spec.shape[0],11)

    freq_resolution = sample_rate/num_samples_per_frame
    yticklabels = np.uint16(yticks*freq_resolution)
    plt.yticks(yticks,yticklabels)
    plt.ylabel("Frequency( in Hz )")
    
    xticks = np.linspace(0,spec.shape[1],10)
    total_time = num_samples/sample_rate
    total_frames = spec.shape[1]
    time_per_frame = total_time/total_frames
    
    xticklabels = (xticks*time_per_frame).astype(np.float64)
    xticklabels = [round(x,2) for x in xticklabels]
    plt.xticks(xticks,xticklabels,rotation=45)
#     plt.set_xticklabels(xticklabels, rotation = 45, ha="right")
    plt.xlabel("Time (in s)")
    plt.colorbar()
    
    plt.show()


# In[6]:


def get_spectrogram(audio_file,sample_rate=16000,noise_sample=None,frame_size=0.025,frame_stride=0.01):
    signal,sample_rate = librosa.load(audio_file,sr=sample_rate)
    max_len_sample = sample_rate                         # since all samples are max 1 sec
    padded_suffix = np.zeros((max_len_sample-len(signal)))
    # padded_suffix += 1e-8
    signal = np.concatenate((signal,padded_suffix),axis=0)
    signal +=1e-10
    num_samples = len(signal)
    if noise_sample is not None:
        signal+=noise_sample
    plot_signal_in_time_freq(signal,sample_rate,num_samples)
    signal_frames = quantize_intervals(signal,num_samples,sample_rate,frame_size=frame_size,frame_stride=frame_stride)
#     signal_frames *= np.hamming(signal_frames.shape[1])
    spec = fft(signal_frames,sample_rate)
#     print(spec.shape)
    return spec,num_samples,sample_rate


# In[7]:


my_spec,num_samples,sample_rate = get_spectrogram(train_paths[101],sample_rate=16000,frame_size=0.025)
plot_spectrogram(my_spec,sample_rate,num_samples)


# In[ ]:




