#TODO
import numpy as np
import os
import math
import torch
import scipy.io as scio
import random
from random import choice
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
def noised(signal):
    # signal (19, 500)
    SNR = 5             #
    noise = np.random.randn(signal.shape[0], signal.shape[1])
    noise = noise - np.mean(noise)#
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise
def negated(signal):
    return signal * -1
def opposite_time(signal):
    return signal[:,::-1]
def permuted(signal, segment_length = 25):

    num_channels, num_timepoints = signal.shape
    num_segments = math.ceil(num_timepoints / segment_length)
    listA = [i for i in range(num_segments)]
    random.shuffle(listA)
    sig = signal[:, listA[0] * segment_length:listA[0] * segment_length + segment_length]
    for i in range(1, len(listA)):
        sig = np.concatenate((sig, signal[:,listA[i] * segment_length:listA[i] * segment_length + segment_length]), axis=-1)
    return sig
def scale(signal, sc = [0.5, 2, 1.5, 0.8]):
    s = choice(sc)
    return signal * s
def inter_data(hr, window=11):
    time3 = savgol_filter(hr, window_length=window, polyorder=2)
    return time3
def time_warp(signal):
    for i in range(signal.shape[0]):
        signal[i,:] = inter_data(signal[i,:],11)
    return signal
def regular_mm(data):
    batch, ch, n_times = data.shape
    dim = ch * n_times
    data = data.reshape(data.shape[0], dim)

    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = data.reshape(data.shape[0], ch, n_times)
    return data
def freq_mask(data, mask_ratio=0.1):
    fft = np.fft.fft(data, axis=1)
    freq_len = fft.shape[1]
    mask = np.random.choice([0, 1], size=freq_len, p=[mask_ratio, 1-mask_ratio])
    return np.fft.ifft(fft * mask, axis=1).real
def bandpass_emphasis(data, low=0.1, high=30.0, fs=256, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)
def channel_dropout(data, drop_prob=0.2):

    num_channels = data.shape[0]
    mask = np.random.binomial(1, 1-drop_prob, size=num_channels)
    return data * mask[:, np.newaxis]
def channel_shuffle(data):
    indices = np.random.permutation(data.shape[0])
    return data[indices]
def dynamic_scaling(data, scale_range=(0.8, 1.2)):
    scales = np.random.uniform(scale_range[0], scale_range[1], size=data.shape[0])
    return data * scales[:, np.newaxis]

def transformation(dataX, low=0.1, high=30.0, fs=256, order=4, drop_prob = 0.2, scale_range=(0.8, 1.2)):
    data_i = {
        'raw' : dataX,
        'noised' : np.zeros((dataX.shape[0],dataX.shape[1],dataX.shape[2])),
        'negated' : np.zeros((dataX.shape[0],dataX.shape[1],dataX.shape[2])),
        'opposite_time': np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'permuted': np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'scale': np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'time_warp' : np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'freq_mask': np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'bandpass_emphasis' : np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'channel_dropout' : np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'channel_shuffle' : np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2])),
        'dynamic_scaling' : np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2]))
    }


    for i in range(dataX.shape[0]):
        data_i['noised'][i] = noised(dataX[i].copy())
        data_i['negated'][i] = negated(dataX[i].copy())
        data_i['opposite_time'][i] = opposite_time(dataX[i].copy())
        data_i['permuted'][i] = permuted(dataX[i].copy())
        data_i['scale'][i] = scale(dataX[i].copy())
        data_i['time_warp'][i] = time_warp(dataX[i].copy())
        data_i['freq_mask'][i] = freq_mask(dataX[i].copy())
        data_i['bandpass_emphasis'][i] = bandpass_emphasis(
            dataX[i].copy(), low=low, high=high, fs=fs, order=order)
        data_i['channel_dropout'][i] = channel_dropout(dataX[i].copy(), drop_prob=drop_prob)
        data_i['channel_shuffle'][i] = channel_shuffle(dataX[i].copy())
        data_i['dynamic_scaling'][i] = dynamic_scaling(dataX[i].copy(), scale_range=scale_range)

    for key in data_i.keys():
        data_i[key] = regular_mm(data_i[key])
        data_i[key] = np.reshape(data_i[key], data_i[key].shape + (1,))
    return data_i

def shuffle(normal_i, path, ratio = 0.7,
            transformation_selected:list = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale', 'time_warp']):
    listA = [l for l in range(normal_i['raw'].shape[0])]
    random.shuffle(listA)
    listB = [p for p in range(normal_i['raw'].shape[0])]
    dataset_i = {}
    for key in normal_i.keys():
        dataset_i[key] = np.zeros([normal_i['raw'].shape[0], normal_i['raw'].shape[1], normal_i['raw'].shape[2], normal_i['raw'].shape[3]])
    for w, r in zip(listA, listB):
        for key in normal_i.keys():
            dataset_i[key][r, :, :, :] = normal_i[key][w, :, :, :]
    print('shuffle done')
    x_train_i, x_test_i = {}, {}
    for key in normal_i.keys():
        x_train_i[key] = dataset_i[key][:int(dataset_i[key].shape[0] * ratio), :, :, :]
        x_test_i[key] = dataset_i[key][int(dataset_i[key].shape[0] * ratio):, :, :, :]
    print('save preparing')
    for key in x_train_i.keys():
        torch.save(torch.tensor(x_train_i[key]), os.path.join(path, 'x_train_'+ key +'.pt'))
    for key in x_test_i.keys():
        torch.save(torch.tensor(x_test_i[key]), os.path.join(path, 'x_test_'+ key +'.pt'))

    X_train = np.concatenate([x_train_i[tf] for tf in transformation_selected], axis=-1) # (batch, input_dim_x, input_dim_y, 1) ->(batch, input_dim_x, input_dim_y, trans)
    X_train = X_train.transpose(0,-1, 1, 2)   # (batch, input_dim_x, input_dim_y, trans) -> (batch, trans, input_dim_x, input_dim_y)
    X_train = np.reshape(X_train, X_train.shape + (1,)) # (batch, trans, input_dim_x, input_dim_y) -> (batch, trans, input_dim_x, input_dim_y, 1)
    X_train = X_train.transpose(0, 1, 4, 3, 2)
    print(X_train.shape)
    torch.save(torch.tensor(X_train), os.path.join(path, 'x_train.pt'))

    X_test = np.concatenate([x_test_i[tf] for tf in transformation_selected], axis=-1)
    X_test = X_test.transpose(0,-1, 1, 2)
    X_test = np.reshape(X_test, X_test.shape + (1,))
    X_test = X_test.transpose(0, 1, 4, 3, 2)
    print(X_test.shape)
    torch.save(torch.tensor(X_test), os.path.join(path, 'x_test.pt'))
def DSADS(folder_path = None, save_path = None, mixedbatchdata_standardization = True):
    x_train = torch.load(os.path.join(folder_path, 'x_train.pt')).numpy()
    x_val = torch.load(os.path.join(folder_path, 'x_val.pt')).numpy()
    x_test = torch.load(os.path.join(folder_path, 'x_test.pt')).numpy()

    len_train = len(x_train)
    len_val = len(x_val)
    len_test = len(x_test)

    if mixedbatchdata_standardization:
        dataX = np.vstack([x_train,x_val,x_test])
        data_raw, data_no, data_ne, data_op, data_pe, data_sc, data_ti = transformation(dataX)

        data_raw_train = data_raw[:len_train]
        data_no_train = data_no[:len_train]
        data_ne_train = data_ne[:len_train]
        data_op_train = data_op[:len_train]
        data_pe_train = data_pe[:len_train]
        data_sc_train = data_sc[:len_train]
        data_ti_train = data_ti[:len_train]

        data_raw_val = data_raw[len_train:len_val+len_train]
        data_no_val = data_no[len_train:len_val+len_train]
        data_ne_val = data_ne[len_train:len_val+len_train]
        data_op_val = data_op[len_train:len_val+len_train]
        data_pe_val = data_pe[len_train:len_val+len_train]
        data_sc_val = data_sc[len_train:len_val+len_train]
        data_ti_val = data_ti[len_train:len_val+len_train]

        data_raw_test = data_raw[len_val+len_train:]
        data_no_test = data_no[len_val+len_train:]
        data_ne_test = data_ne[len_val+len_train:]
        data_op_test = data_op[len_val+len_train:]
        data_pe_test = data_pe[len_val+len_train:]
        data_sc_test = data_sc[len_val+len_train:]
        data_ti_test = data_ti[len_val+len_train:]




    print('save preparing')
    torch.save(torch.tensor(data_raw_train), os.path.join(save_path, 'data_raw_train.pt'))
    torch.save(torch.tensor(data_no_train), os.path.join(save_path, 'data_no_train.pt'))
    torch.save(torch.tensor(data_ne_train), os.path.join(save_path, 'data_ne_train.pt'))
    torch.save(torch.tensor(data_op_train), os.path.join(save_path, 'data_op_train.pt'))
    torch.save(torch.tensor(data_pe_train), os.path.join(save_path, 'data_pe_train.pt'))
    torch.save(torch.tensor(data_sc_train), os.path.join(save_path, 'data_sc_train.pt'))
    torch.save(torch.tensor(data_ti_train), os.path.join(save_path, 'data_ti_train.pt'))

    x_train = np.concatenate((data_raw_train,data_no_train,data_ne_train,data_op_train,data_pe_train,data_sc_train,data_ti_train),axis=-1) # (batch, input_dim_x, input_dim_y, 1) ->(batch, input_dim_x, input_dim_y, trans)
    x_train = x_train.transpose(0,-1,1,2)
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_train = x_train.transpose(0, 1, 4, 2, 3)
    print( x_train.shape)
    torch.save(torch.tensor(x_train), os.path.join(save_path, 'x_train.pt'))



    torch.save(torch.tensor(data_raw_val), os.path.join(save_path, 'data_raw_val.pt'))
    torch.save(torch.tensor(data_no_val), os.path.join(save_path, 'data_no_val.pt'))
    torch.save(torch.tensor(data_ne_val), os.path.join(save_path, 'data_ne_val.pt'))
    torch.save(torch.tensor(data_op_val), os.path.join(save_path, 'data_op_val.pt'))
    torch.save(torch.tensor(data_pe_val), os.path.join(save_path, 'data_pe_val.pt'))
    torch.save(torch.tensor(data_sc_val), os.path.join(save_path, 'data_sc_val.pt'))
    torch.save(torch.tensor(data_ti_val), os.path.join(save_path, 'data_ti_val.pt'))

    x_val = np.concatenate((data_raw_val, data_no_val,data_ne_val,data_op_val,data_pe_val,data_sc_val,data_ti_val), axis=-1)
    x_val = x_val.transpose(0,-1,1,2)
    x_val = np.reshape(x_val, x_val.shape + (1,))
    x_val = x_val.transpose(0, 1, 4, 2, 3)
    print( x_val.shape)
    torch.save(torch.tensor(x_val), os.path.join(save_path, 'x_val.pt'))


    torch.save(data_raw_test, os.path.join(save_path, 'data_raw_test.pt'))
    torch.save(data_no_test, os.path.join(save_path, 'data_no_test.pt'))
    torch.save(data_ne_test, os.path.join(save_path, 'data_ne_test.pt'))
    torch.save(data_op_test, os.path.join(save_path, 'data_op_test.pt'))
    torch.save(data_pe_test, os.path.join(save_path, 'data_pe_test.pt'))
    torch.save(data_sc_test, os.path.join(save_path, 'data_sc_test.pt'))
    torch.save(data_ti_test, os.path.join(save_path, 'data_ti_test.pt'))

    x_test = np.concatenate((data_raw_test, data_no_test,data_ne_test,data_op_test,data_pe_test,data_sc_test,data_ti_test), axis=-1)
    x_test = x_test.transpose(0, -1, 1, 2)
    x_test = np.reshape(x_test, x_test.shape + (1,))
    x_test = x_test.transpose(0, 1, 4, 2, 3)
    print(x_test.shape, end=', ')
    torch.save(torch.tensor(x_test), os.path.join(save_path, 'abnormal.pt'))
def TUSZ(folder_path = None, save_path = None,
         ratio=0.7, low=0.1, high=30.0, fs=256, order=4, drop_prob=0.2, scale_range=(0.8, 1.2),
         transformation_selected: list = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale','time_warp']):
    normal = np.load(os.path.join(folder_path, 'normal.npy'))
    abnormal = np.load(os.path.join(folder_path, 'abnormal.npy'))
    number = normal.shape[0]
    print(normal.shape, abnormal.shape)
    dataX = np.vstack((normal,abnormal))
    print(dataX.shape)

    data_i = transformation(dataX, low=low, high=high, fs=fs, order=order, drop_prob = drop_prob, scale_range=scale_range) # {'raw' : (batch, 19, 256), ...}

    #####正常数据#####
    normal_i = {}
    #####异常数据#####
    abnormal_i = {}
    for key in data_i.keys():
        normal_i[key] = data_i[key][:number]
        abnormal_i[key] = data_i[key][number:]
    ####################################save normal data######################
    shuffle(normal_i, path = save_path, ratio=ratio, transformation_selected=transformation_selected)
    ####################################save abnormal data######################
    for key in abnormal_i.keys():
        torch.save(torch.tensor(abnormal_i[key]), os.path.join(save_path, 'abnormal_' + key + '.pt'))
    abnormal = np.concatenate([abnormal_i[tf] for tf in transformation_selected], axis=-1)
    abnormal = abnormal.transpose(0, -1, 1, 2)
    abnormal = np.reshape(abnormal, abnormal.shape + (1,))
    abnormal = abnormal.transpose(0, 1, 4, 3, 2)
    print(abnormal.shape)
    torch.save(torch.tensor(abnormal), os.path.join(save_path, 'abnormal.pt'))

def AUBMC(folder_path = None, save_path =None,
          ratio = 0.7, low=0.1, high=30.0, fs=500, order=4, drop_prob = 0.2, scale_range=(0.8, 1.2),
          transformation_selected:list = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale', 'time_warp']):
    #################################################################################################
    normal = np.load(r"normal-10-11.npy")
    abnormal = np.load(r"seizure-12-13-14-15.npy")
    save_path = r"Test"
    #################################################################################################
    number = normal.shape[0]
    dataX = np.vstack((normal,abnormal))
    data_i = transformation(dataX, low=low, high=high, fs=fs, order=order, drop_prob = drop_prob, scale_range=scale_range) # {'raw' : (batch, 19, 500), ...}
    normal_i = {}
    abnormal_i = {}
    for key in data_i.keys():
        normal_i[key] = data_i[key][:number]
        abnormal_i[key] = data_i[key][number:]

    shuffle(normal_i, path = save_path, ratio=ratio, transformation_selected=transformation_selected)
    for key in abnormal_i.keys():
        torch.save(torch.tensor(abnormal_i[key]), os.path.join(save_path, 'abnormal_' + key + '.pt'))
    abnormal = np.concatenate([abnormal_i[tf] for tf in transformation_selected], axis=-1)
    abnormal = abnormal.transpose(0, -1, 1, 2)
    abnormal = np.reshape(abnormal, abnormal.shape + (1,))
    abnormal = abnormal.transpose(0, 1, 4, 3, 2)
    print(abnormal.shape)
    torch.save(torch.tensor(abnormal), os.path.join(save_path, 'abnormal.pt'))

def CHB_MIT(folder_path = None, save_path =None,
            ratio=0.7, low=0.1, high=30.0, fs=256, order=4, drop_prob=0.2, scale_range=(0.8, 1.2),
            transformation_selected: list = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale','time_warp']):
    normal = np.load(os.path.join(folder_path, 'normal.npy'))
    abnormal = np.load(os.path.join(folder_path, 'abnormal.npy'))

    number = normal.shape[0]
    print(normal.shape, abnormal.shape)
    dataX = np.vstack((normal,abnormal))
    print(dataX.shape)

    dataX = np.vstack((normal,abnormal))
    data_i = transformation(dataX, low=low, high=high, fs=fs, order=order, drop_prob = drop_prob, scale_range=scale_range) # {'raw' : (batch, 19, 256), ...}
    #####正常数据#####
    normal_i = {}
    #####异常数据#####
    abnormal_i = {}
    for key in data_i.keys():
        normal_i[key] = data_i[key][:number]
        abnormal_i[key] = data_i[key][number:]
    shuffle(normal_i, path = save_path, ratio=ratio, transformation_selected=transformation_selected)
    for key in abnormal_i.keys():
        torch.save(torch.tensor(abnormal_i[key]), os.path.join(save_path, 'abnormal_' + key + '.pt'))
    abnormal = np.concatenate([abnormal_i[tf] for tf in transformation_selected], axis=-1)
    abnormal = abnormal.transpose(0, -1, 1, 2)
    abnormal = np.reshape(abnormal, abnormal.shape + (1,))
    abnormal = abnormal.transpose(0, 1, 4, 3, 2)
    print(abnormal.shape)
    torch.save(torch.tensor(abnormal), os.path.join(save_path, 'abnormal.pt'))

def INCART(folder_path = None, save_path =None,
        ratio=0.7, low=0.1, high=30.0, fs=257, order=4, drop_prob=0.2, scale_range=(0.8, 1.2),
        transformation_selected: list = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale','time_warp']):
    normal = np.load(os.path.join(folder_path, 'normal.npy'))[:7000] # (batch, 62, 200)
    abnormal = np.load(os.path.join(folder_path, 'abnormal.npy'))[:2000]
    number = normal.shape[0]
    print(normal.shape, abnormal.shape)
    dataX = np.vstack((normal,abnormal))
    print(dataX.shape)

    dataX = np.vstack((normal,abnormal))
    data_i = transformation(dataX, low=low, high=high, fs=fs, order=order, drop_prob = drop_prob, scale_range=scale_range) # {'raw' : (batch, 19, 256), ...}
    normal_i = {}
    abnormal_i = {}
    for key in data_i.keys():
        normal_i[key] = data_i[key][:number]
        abnormal_i[key] = data_i[key][number:]
    shuffle(normal_i, path = save_path, ratio=ratio, transformation_selected=transformation_selected)
    for key in abnormal_i.keys():
        torch.save(torch.tensor(abnormal_i[key]), os.path.join(save_path, 'abnormal_' + key + '.pt'))
    abnormal = np.concatenate([abnormal_i[tf] for tf in transformation_selected], axis=-1)
    abnormal = abnormal.transpose(0, -1, 1, 2)
    abnormal = np.reshape(abnormal, abnormal.shape + (1,))
    abnormal = abnormal.transpose(0, 1, 4, 3, 2)
    print(abnormal.shape)
    torch.save(torch.tensor(abnormal), os.path.join(save_path, 'abnormal.pt'))

def PTB(folder_path = None, save_path =None,
        ratio=0.7, low=0.1, high=30.0, fs=1000, order=4, drop_prob=0.2, scale_range=(0.8, 1.2),
        transformation_selected: list = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale','time_warp']):
    normal = np.load(os.path.join(folder_path, 'normal.npy'))[:7000] # (batch, 62, 200)
    abnormal = np.load(os.path.join(folder_path, 'abnormal.npy'))[:2000]
    number = normal.shape[0]
    print(normal.shape, abnormal.shape)
    dataX = np.vstack((normal,abnormal))

    dataX = np.vstack((normal,abnormal))
    data_i = transformation(dataX, low=low, high=high, fs=fs, order=order, drop_prob = drop_prob, scale_range=scale_range) # {'raw' : (batch, 19, 256), ...}
    normal_i = {}
    abnormal_i = {}
    for key in data_i.keys():
        normal_i[key] = data_i[key][:number]
        abnormal_i[key] = data_i[key][number:]
    ####################################save normal data######################
    shuffle(normal_i, path = save_path, ratio=ratio, transformation_selected=transformation_selected)
    ####################################save abnormal data######################
    for key in abnormal_i.keys():
        torch.save(torch.tensor(abnormal_i[key]), os.path.join(save_path, 'abnormal_' + key + '.pt'))
    abnormal = np.concatenate([abnormal_i[tf] for tf in transformation_selected], axis=-1)
    abnormal = abnormal.transpose(0, -1, 1, 2)
    abnormal = np.reshape(abnormal, abnormal.shape + (1,))
    abnormal = abnormal.transpose(0, 1, 4, 3, 2)
    print(abnormal.shape)
    torch.save(torch.tensor(abnormal), os.path.join(save_path, 'abnormal.pt'))

if __name__ == '__main__':
    transformation_selected = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale', 'time_warp',
                               'freq_mask', 'bandpass_emphasis', 'channel_dropout', 'channel_shuffle', 'dynamic_scaling']
    AUBMC(ratio = 0.7, low=0.1, high=30.0, fs=500, order=4, drop_prob = 0.2, scale_range=(0.8, 1.2),
          transformation_selected = ['raw', 'noised', 'negated', 'opposite_time', 'permuted', 'scale','time_warp'])

