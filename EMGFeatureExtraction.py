import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from nitime.algorithms.autoregressive import AR_est_LD
from sklearn.preprocessing import StandardScaler
import pywt
from scipy.ndimage import zoom
from scipy import signal
import glob
import os
import random
import math

random.seed(101)

def dict_product(dicts):
    result = []
    for d in dicts:
        result += list((dict(zip(d, x))) for x in product(*d.values()))
    return result

def samples_num_in_window(frequency, window_size_ms):
    return int(window_size_ms * frequency / 1000)

def emg_data_windowing(data, window_size):
    data_win = np.copy(data)
    data_x = data_win[:,:-1]
    data_y = data_win[:,-1]
    n, m = data_x.shape
    size = n * m
    residual_rows_num =  n % window_size
    if residual_rows_num != 0:
        data_x = data_x[:-residual_rows_num,:]
        data_y = data_y[:-residual_rows_num]
    data_x = data_x.reshape((-1, m * window_size))
    
    data_y = data_y.reshape((-1, window_size))
    data_y = np.array(list(map(np.mean, data_y)))
    
    mixed_classes_idxs = np.where(data_y % 1 != 0)
    
    data_win = np.c_[data_x, data_y]
    data_win = np.delete(data_win, mixed_classes_idxs, 0)
    
    return data_win

def read_emg(data_path):
    sessions_csv = []
    for path, _, files in os.walk(data_path):
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
    print('result shape: ', data.shape)

    return data


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


def calculate_features(data_x, channels_num, ar_features=True):
    n, m = data_x.shape
    features = []
    
    for channel in range(channels_num):
        channel_features = []
        
        # Calculate MAV, ZC, SSC, WL, RMS features
        channel_features.append(list(map(mean_absolute_value, data_x[:,channel::channels_num])))
        channel_features.append(list(map(waveform_length, data_x[:,channel::channels_num])))
        channel_features.append(list(map(zero_crossing, data_x[:,channel::channels_num])))
        channel_features.append(list(map(slope_sign_changes, data_x[:,channel::channels_num])))
        channel_features.append(list(map(root_mean_square, data_x[:,channel::channels_num])))
        
        if ar_features:
            # calculate AR6 coefficients
            ar_order = 6
            ar_coef = np.array(list(map(lambda x: autoregression_coefficients(x, ar_order), data_x[:,channel::channels_num])))
            channel_features += ar_coef.transpose().tolist()
        
        features += channel_features
    
    return np.array(features).transpose()


def calculate_CWT_vector(row, zoom_factor):
    coef, freqs = pywt.cwt(row, scales=np.arange(1, 33), wavelet='mexh')
    coef = zoom(coef, zoom_factor, order=0)
    return coef.transpose()

def calculate_CWT(data_x, channels_num, zoom_factor):
    features = []
    for channel in range(channels_num):        
        coef = list(map(lambda x: calculate_CWT_vector(x, zoom_factor), data_x[:,channel::channels_num]))
        features.append(np.array(coef).transpose())
    return np.array(features).transpose()

from scipy import signal

def spectrogram(data_x, channels_num, fs, npserseg, noverlap):
    features = []
    for channel in range(channels_num):        
        coef = list(map(lambda x: spectrogram_vector(x, fs, npserseg, noverlap), data_x[:,channel::channels_num]))
        features.append(np.array(coef).transpose())
    return np.array(features).transpose()


def spectrogram_vector(vector, fs, npserseg, noverlap):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window="hann",
                                                                                         scaling="spectrum")
    return spectrogram_of_vector[1:]

