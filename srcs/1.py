# coding=utf-8
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train_path = '..\\data\\train'


def run_time(func):
    def wrapper(num):
        start = time.time()
        data_feat, data_norm, data_fail = func(num)
        end = time.time()
        print("->读取%d号风机数据耗时: %.2f s" % (num, (end - start)))
        return data_feat, data_norm, data_fail
    return wrapper


@run_time
def get_data(num):
    num = str(num)
    data_feat = pd.read_csv(train_path + '\\' + num + '\\' + num + '_data.csv', parse_dates=["time"])
    data_norm = pd.read_csv(train_path + '\\' + num + '\\' + num + '_normalInfo.csv', parse_dates=["startTime", "endTime"])
    data_fail = pd.read_csv(train_path + '\\' + num + '\\' + num + '_failureInfo.csv', parse_dates=["startTime", "endTime"])
    return data_feat, data_norm, data_fail


def add_label(x, data_norm, data_fail):
    for i in range(data_fail.shape[0]):
        if data_fail['startTime'][i] <= x <= data_fail['endTime'][i]:
            return 1
        elif data_norm['startTime'][i] <= x <= data_norm['endTime'][i]:
            return 0
        else:
            return np.NaN


if __name__ == '__main__':
    data_15, norm_15, fail_15 = get_data(15)
    data_21, norm_21, fail_21 = get_data(21)
    # data_15["label"] = data_15["time"].map(lambda x: add_label(x, norm_15, fail_15))

