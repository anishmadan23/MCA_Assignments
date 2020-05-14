#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io.wavfile
import glob
import numpy as np 
import librosa
from IPython.display import Audio
import matplotlib.pyplot as plt
from utils import *
from q1 import plot_spectrogram
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.fftpack import dct


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


# ### Formulae used
# - m=2595*log10(1 + f/700) <br>
# - f=700(10^(m/2595) − 1)

# In[3]:


def get_freq_from_mel(mel_arr):
    return 700*(10**(mel_arr/2595) - 1)


# In[4]:


def get_mel_from_freq(freq_arr):
    freq_arr = np.array(freq_arr)
#     print(freq_arr,type(freq_arr))
    return 2595*np.log10(1+freq_arr/700)


# In[5]:


def make_filter_bank(periogram_frames,sample_rate,num_triang_filts=30):
    start_freq = 0
    end_freq = get_mel_from_freq([sample_rate/2])[0]
    mel_axis_pts = np.linspace(start_freq,end_freq,num_triang_filts+2)
#     freq_axis_pts = 
#     print(mel_axis_pts)
    N = periogram_frames.shape[1]*2
#     print(N)
#     print(get_freq_from_mel(mel_axis_pts))
#     print((N+1)*get_freq_from_mel(mel_axis_pts)/sample_rate)
    freq_axis_pts = np.uint8(np.floor((N+1)*get_freq_from_mel(mel_axis_pts)/sample_rate))
#     print(freq_axis_pts,len(freq_axis_pts),len(np.unique(freq_axis_pts)))
    
    mel_banks = np.zeros((num_triang_filts,int(N/2)))
    
    for m in range(1,num_triang_filts+1):
        
        for k in range(freq_axis_pts[m-1],freq_axis_pts[m+1]):
            if freq_axis_pts[m-1]<=k and k<freq_axis_pts[m]:
                mel_banks[m-1,k] = (k-freq_axis_pts[m-1])/(freq_axis_pts[m]-freq_axis_pts[m-1])
            elif freq_axis_pts[m]<k and k<= freq_axis_pts[m+1]:
                mel_banks[m-1,k] = (freq_axis_pts[m+1]-k)/(freq_axis_pts[m+1]-freq_axis_pts[m])
            elif k==freq_axis_pts[m]:
                mel_banks[m-1,k] = 1
            
    mel_banks = np.dot(periogram_frames, mel_banks.T)
    mel_banks_zero = list(zip(np.where(mel_banks==0)[0],np.where(mel_banks==0)[1]))
    for x in mel_banks_zero:
        mel_banks[x] = 1e-20
    
    mel_banks = 10 * np.log10(mel_banks) 
        
    return mel_banks
    


# In[6]:


def plot_mfcc_spectrogram(mfcc,sample_rate,num_samples):
    plt.figure(figsize=(20,6))
    mfcc_plotted = plt.imshow(mfcc,origin='lower',cmap='jet')
    # num_samples_per_frame = spec.shape[0]*2
    yticks = np.uint8(np.linspace(0,mfcc.shape[0],5))
    yticklabels = np.uint8(np.linspace(0,mfcc.shape[0],5))
    # freq_resolution = sample_rate/num_samples_per_frame
    # yticklabels = np.uint16(yticks*freq_resolution)
    plt.yticks(yticks,yticklabels)
    plt.ylabel("MFCC Coeffs")
    
    xticks = np.linspace(0,mfcc.shape[1],10)
    total_time = num_samples/sample_rate
    total_frames = mfcc.shape[1]
    time_per_frame = total_time/total_frames
    
    xticklabels = (xticks*time_per_frame).astype(np.float64)
    xticklabels = [round(x,2) for x in xticklabels]
    plt.xticks(xticks,xticklabels,rotation=45)
#     plt.set_xticklabels(xticklabels, rotation = 45, ha="right")
    plt.xlabel("Time (in s)")
    plt.colorbar()
    
    plt.show()


# In[9]:


def get_mfcc_feats(audio_file,sample_rate=16000,noise_sample=None,frame_size=0.025,frame_stride=0.01,num_triang_filts = 30):
    signal,sample_rate = librosa.load(audio_file,sr=sample_rate)
    max_len_sample = sample_rate                         # since all samples are max 1 sec
    padded_suffix = np.zeros((max_len_sample-len(signal)))
    signal = np.concatenate((signal,padded_suffix),axis=0)
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    num_samples = len(signal)
    if noise_sample is not None:
        signal+=noise_sample
    # plot_signal_in_time_freq(signal,sample_rate,num_samples)
    signal_frames = quantize_intervals(signal,num_samples,sample_rate,frame_size=frame_size,frame_stride=frame_stride)

    signal_frames = signal_frames* np.hamming(signal_frames.shape[1])
    spec_frames = np.absolute(np.fft.rfft(signal_frames, signal_frames.shape[1]))
    log_spec_frames = np.log10(spec_frames).T
    perio_gram_frames = ((spec_frames) ** 2)/(sample_rate*frame_size)

    mel_power_banks = make_filter_bank(spec_frames,sample_rate,num_triang_filts=num_triang_filts)
    num_coeffs_mfcc = 12
    mfcc = dct(mel_power_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_coeffs_mfcc + 1)]
    
    #optional liftering( helps in ASR)
    n = np.arange(mfcc.shape[0])
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    lift = np.expand_dims(lift,axis=1)
    mfcc = mfcc * lift

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)         # normalize

    
    return mel_power_banks,mfcc.T,num_samples,sample_rate


# In[13]:


mel_power_banks,mfcc_feat,num_samples,sample_rate = get_mfcc_feats(train_paths[101],sample_rate=16000,frame_size=0.025)
# plot_spectrogram(mel_power_banks,sample_rate,num_samples)
plot_mfcc_spectrogram(mfcc_feat,sample_rate,num_samples)


# In[ ]:




