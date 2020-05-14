#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import recall_score,precision_score
import numpy as np 
from q1 import get_spectrogram
from q2 import get_mfcc_feats
import glob
import librosa
import math
import random
random.seed(0)
import joblib


# In[2]:


root_dir = './Dataset/'
base_train_path = root_dir + 'training/'
base_val_path = root_dir + 'validation/'
base_noise_path = root_dir + '_background_noise_/'
train_val_classes = os.listdir(base_train_path)
noise_files = glob.glob(base_noise_path+'*')
train_paths = []
val_paths = []

for x in train_val_classes:
    train_paths.extend(glob.glob(base_train_path+x+'/*'))
    val_paths.extend(glob.glob(base_val_path+x+'/*'))


# In[3]:


def get_noise_feats(noise_files,sample_rate=16000):
#     assert(num_noises==len(list_noise_coeffs))
    max_subseq_noise = 20         # number of 1 second sections to take from noise sample
    noise_mat = []
    for idx,noise_file in enumerate(noise_files):
        noise_signal,noise_sr = librosa.load(noise_file,sr=sample_rate)
        num_start_points = math.floor(len(noise_signal)/sample_rate)-1
        start_points = np.arange(num_start_points)
        start_points_samples = start_points*sample_rate
#         print(start_points_samples)
        sel_start_pts = random.sample(list(start_points_samples),max_subseq_noise)
        for start_pt in sel_start_pts:
            noise_subseq = noise_signal[start_pt:start_pt+sample_rate]
            noise_mat.append(noise_subseq)
    return np.array(noise_mat)


# In[4]:


def get_random_noise_sample(noise_mat,num_noises=1,list_noise_coeffs=[0.005],sample_rate=16000):
    assert(num_noises==len(list_noise_coeffs))
    total_noise_samples = np.arange(noise_mat.shape[0])
    noise_inds = random.sample(list(total_noise_samples),num_noises)
    
    noise_vec = np.zeros_like(noise_mat[0,:])
    for idx,noise_ind in enumerate(noise_inds):
        noise_vec += list_noise_coeffs[idx]*noise_mat[noise_ind,:]
        
    return noise_vec


# In[5]:


noise_mat = get_noise_feats(noise_files)
# noise_vec = get_random_noise_sample(noise_mat,num_noises=1,list_noise_coeffs=[0.005])
# print(noise_vec.shape)


# In[6]:


class_label_map = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}


# In[14]:


def load_feats(list_of_paths,noise_mat,sample_rate=16000,frame_size=0.025,frame_stride=0.01,feat_type='spec',num_triang_filts=30):
    num_samples = len(list_of_paths)
    ## load dummy feat
    dummy_path = list_of_paths[0]
    
    if feat_type=='spec':
        dummy_feat,_,_ = get_spectrogram(dummy_path,sample_rate=sample_rate,frame_size=frame_size,frame_stride=frame_stride)
    elif feat_type=='mfcc':
        dummy_mel_bank,dummy_feat,_,_ = get_mfcc_feats(dummy_path,sample_rate=sample_rate,noise_sample=None,frame_size=frame_size,frame_stride=frame_stride,num_triang_filts = num_triang_filts)
        
    X = np.zeros((num_samples,dummy_feat.shape[0]*dummy_feat.shape[1]))
    y = np.zeros((num_samples))
    for idx, path in enumerate(list_of_paths):
        print('Progress:',feat_type,idx+1,len(list_of_paths))
        label = path.split('/')[-2]
        my_noise_sample = get_random_noise_sample(noise_mat,num_noises=3,list_noise_coeffs=[0.005,0.001,0.002])
        if feat_type=='spec': 
            my_feat,_,_ = get_spectrogram(path,noise_sample=my_noise_sample,sample_rate=sample_rate,frame_size=frame_size,frame_stride=frame_stride)
        elif feat_type=='mfcc':
            my_mel_bank,my_feat,_,_ = get_mfcc_feats(path,noise_sample=my_noise_sample,sample_rate=sample_rate,frame_size=frame_size,frame_stride=frame_stride,num_triang_filts = num_triang_filts)
        
        X[idx,:] = my_feat.flatten()
        y[idx] = class_label_map[label.strip()]
    print(X.shape,y.shape)
    return X,y
        


# In[ ]:


def load_feats_no_noise(list_of_paths,noise_mat,sample_rate=16000,frame_size=0.025,frame_stride=0.01,feat_type='spec',num_triang_filts=30):
    num_samples = len(list_of_paths)
    ## load dummy feat
    dummy_path = list_of_paths[0]
    
    if feat_type=='spec':
        dummy_feat,_,_ = get_spectrogram(dummy_path,sample_rate=sample_rate,frame_size=frame_size,frame_stride=frame_stride)
    elif feat_type=='mfcc':
        dummy_mel_bank,dummy_feat,_,_ = get_mfcc_feats(dummy_path,sample_rate=sample_rate,noise_sample=None,frame_size=frame_size,frame_stride=frame_stride,num_triang_filts = num_triang_filts)
        
    X = np.zeros((num_samples,dummy_feat.shape[0]*dummy_feat.shape[1]))
    y = np.zeros((num_samples))
    for idx, path in enumerate(list_of_paths):
        print('Progress:',feat_type,idx+1,len(list_of_paths))
        label = path.split('/')[-2]
#         my_noise_sample = get_random_noise_sample(noise_mat,num_noises=3,list_noise_coeffs=[0.005,0.001,0.002])
        if feat_type=='spec': 
            my_feat,_,_ = get_spectrogram(path,noise_sample=None,sample_rate=sample_rate,frame_size=frame_size,frame_stride=frame_stride)
        elif feat_type=='mfcc':
            my_mel_bank,my_feat,_,_ = get_mfcc_feats(path,noise_sample=None,sample_rate=sample_rate,frame_size=frame_size,frame_stride=frame_stride,num_triang_filts = num_triang_filts)
        
        X[idx,:] = my_feat.flatten()
        y[idx] = class_label_map[label.strip()]
    print(X.shape,y.shape)
    return X,y
        


# In[20]:


X_train_spec,y_train_spec = np.load('X_train_spec_noise.npy'),np.load('y_train_spec_noise.npy')

X_train_mfcc,y_train_mfcc = np.load('X_train_mfcc_noise.npy'),np.load('y_train_spec_noise.npy')


# In[23]:


X_val_spec_nn,y_val_spec_nn = np.load('X_val_spec_nn.npy'),np.load('y_val_spec_nn.npy')
X_val_mfcc_nn, y_val_mfcc_nn = np.load('X_val_mfcc_nn.npy'),np.load('y_val_mfcc_nn.npy')


# In[22]:


clf_spec = SVC(gamma='auto')
clf_spec.fit(X_train_spec, y_train_spec)


# In[37]:


def calc_preds(y_true,y_pred):
    prec_score1 = precision_score(y_true, y_pred, average='macro')
    prec_score2 = precision_score(y_true, y_pred, average='micro')
    prec_score3 = precision_score(y_true, y_pred, average='weighted')
    recall_score1 = recall_score(y_true, y_pred, average='macro')
    recall_score2 = recall_score(y_true, y_pred, average='micro')
    recall_score3 = recall_score(y_true, y_pred, average='weighted')
    print("Precision",prec_score1,prec_score2,prec_score3)
    print("Recall",recall_score1,recall_score2,recall_score3)
    print('Accuracy:{}'.format(sum(y_true==y_pred)/len(y_true)))


# In[26]:


y_pred_spec = clf_spec.predict(X_val_spec_nn)
calc_preds(y_val_spec_nn,y_pred_spec)


# In[30]:


joblib.dump(clf_spec,'clf_spec_noise_model.joblib')


# In[ ]:




