import os
import scipy.io.wavfile
import glob
import numpy as np 
from IPython.display import Audio
import matplotlib.pyplot as plt

def plot_signal_in_time_freq(signal,sample_rate,num_samples):
    x_axis = np.arange(num_samples)/num_samples
    plt.figure(dpi=120)
    plt.plot(x_axis,signal)
    plt.xlabel('Time (in s)')
    plt.ylabel('Amplitude')
    plt.show()

def quantize_intervals(signal,num_samples,sample_rate,frame_size=0.025, frame_stride=0.01):
    # convert time variables into samples format
    frame_size_samples = int(round(frame_size*sample_rate))
    frame_stride_samples = int(round(frame_stride*sample_rate))
    
    my_frames = []
    for i in range(0,num_samples,frame_stride_samples):
        if (num_samples-i)<frame_size_samples:
            padded_suffix = np.zeros((frame_size_samples -(num_samples-i)))
            padded_frame = np.concatenate((signal[i:],padded_suffix),axis=0)
            assert (len(padded_frame)==frame_size_samples)
            my_frames.append(padded_frame)
            break
        else:
            my_frames.append(signal[i:i+frame_size_samples])
            
    return np.array(my_frames)

def fft(signal_frames,sample_rate,frame_size=0.025,frame_stride=0.01):
    num_frames = signal_frames.shape[0]
    N = signal_frames.shape[1]              # frame window size
    n = np.arange(N)
    
    my_spec = np.zeros((num_frames,int(round(N/2))))
    
    for i in range(num_frames):
        for k in range(0,int(round(N/2))):            # nyquist limit
            frame = signal_frames[i,:]
            exp_term = np.exp((-1j*2*np.pi*k*n)/N)
            Xk = np.dot(frame,exp_term)/N              # check divide by N
            my_spec[i,k] = np.abs(Xk)*2
            
    return np.log10(my_spec).T




