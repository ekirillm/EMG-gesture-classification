import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from nitime.algorithms.autoregressive import AR_est_LD
from sklearn.preprocessing import StandardScaler
import glob
import os
import random

random.seed(101)

def read_emg(path):
    sessions_csv = []
    for path, _, files in os.walk(path):
        for name in files:
            sessions_csv.append(os.path.join(path, name))

    data = pd.concat([pd.read_csv(file, header = None) for file in sessions_csv]).values
    print('input shape', data.shape)
    
    # reshape data
    # one column - one channel
    data_x = data[:,:-1]
    data_y = data[:,-1]
    data_x = data_x.reshape((-1, 8))
    data_y = data_y.repeat(8)
    data_y = data_y.reshape((-1,1))
    data = np.concatenate((data_x, data_y), axis=1)
    print('All sessions shape: ', data.shape)

    return data

def emg_windowing(data, window_size):
    data_x = data[:,:-1]
    data_y = data[:,-1]
    n, m = data_x.shape
#     print('shape: ', n, m)
    size = n * m
#     print('size: ', size)
    residual_rows_num =  n % window_size
#     print('delete ', residual_rows_num, 'rows')
    if residual_rows_num != 0:
        data_x = data_x[:-residual_rows_num,:]
        data_y = data_y[:-residual_rows_num]
#     print('data_x: ', data_x.shape)
#     print('data_y: ', data_y.shape)
    data_x = data_x.reshape((-1, m * window_size))
    
    data_y = data_y.reshape((-1, window_size))
    data_y = np.array(list(map(np.mean, data_y)))
#     print('data_x: ', data_x.shape)
#     print('data_y: ', data_y.shape)
    
    mixed_classes_idxs = np.where(data_y % 1 != 0)
#     print('mixed_classes_idxs: ', mixed_classes_idxs)
#     print(mixed_classes_idxs[0])
    
    data = np.c_[data_x, data_y]
    data = np.delete(data, mixed_classes_idxs, 0)
    
    return data


import math

def integrated_absolute_value(segment):
    return sum([abs(s) for s in segment])

def mean_absolute_value(segment):
    return sum([abs(s) for s in segment])/len(segment)

def waveform_length(segment):
    n = len(segment)
    wl = 0
    for i in range(1, n):
        wl += abs(segment[i] - segment[i-1])
    return wl

def zero_crossing(segment):
    n = len(segment)
    zc = 0
    for i in range(n - 1):
        if segment[i] * segment[i+1] < 0:
            zc += 1
    return zc

def slope_sign_changes(segment):
    n = len(segment)
    ssc = 0
    for i in range(1, n-1):
        if segment[i-1] < segment[i] and segment[i] > segment[i+1] or segment[i-1] > segment[i] and segment[i] < segment[i+1]:
            ssc += 1
    return ssc

def root_mean_square(segment):
    return math.sqrt(sum([s*s for s in segment])/len(segment))


def autoregression_coefficients(emg, order):
    coef = AR_est_LD(emg, order=order)[0]
    return coef


def calculate_features(data_x, channels_num):
    n, m = data_x.shape
    features = []
    
    for channel in range(channels_num):
        channel_features = []
        
        # Calculate MAV, ZC, SSC, WL features
        channel_features.append(list(map(mean_absolute_value, data_x[:,channel::channels_num])))
        channel_features.append(list(map(waveform_length, data_x[:,channel::channels_num])))
        channel_features.append(list(map(zero_crossing, data_x[:,channel::channels_num])))
        channel_features.append(list(map(slope_sign_changes, data_x[:,channel::channels_num])))
        
        # calculate AR6 coefficients
        ar_order = 6
        ar_coef = np.array(list(map(lambda x: autoregression_coefficients(x, ar_order), data_x[:,channel::channels_num])))
        channel_features += ar_coef.transpose().tolist()
        features += channel_features
    
    return np.array(features).transpose()


data = read_emg('5sessions')
emg_windows = emg_windowing(data, 40)

data_x = emg_windows[:,:-1]
data_y = emg_windows[:,-1]
features = calculate_features(data_x, 8)
print('Features shape:', features.shape)

train_x, test_x, train_y, test_y = train_test_split(features, data_y, test_size=0.3)

# Create svm Classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)

print("Accuracy:", metrics.accuracy_score(test_y, pred_y))

