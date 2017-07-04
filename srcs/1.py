# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train_path = '..\\data\\train'
data_15 = pd.read_csv(train_path + '\\15\\15_data.csv', parse_dates=["time"], index_col="time")
data_fail = pd.read_csv(train_path + '\\15\\15_failureInfo.csv', parse_dates=["startTime", "endTime"])
data_norm = pd.read_csv(train_path + '\\15\\15_normalInfo.csv', parse_dates=["startTime", "endTime"])


def get_label(x):
    for i in range(data_fail.shape[0]):
        if data_fail['startTime'][i] <= x and x <= data_fail['endTime'][i]:
            return 1
        elif data_norm['startTime'][i] <= x and x <= data_norm['endTime'][i]:
            return 0
        else:
            return np.NaN

data_15["label"] = data_15["time"].map(lambda x: get_label(x))
