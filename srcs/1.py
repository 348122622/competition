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
    data_feat = pd.read_csv('%s\\%s\\%s_data.csv' % (train_path, num, num), parse_dates=["time"])
    data_norm = pd.read_csv('%s\\%s\\%s_normalInfo.csv' % (train_path, num, num), parse_dates=["startTime", "endTime"])
    data_fail = pd.read_csv('%s\\%s\\%s_failureInfo.csv' % (train_path, num, num), parse_dates=["startTime", "endTime"])
    return data_feat, data_norm, data_fail


def add_label(data_feat, data_norm, data_fail):
    data_feat["label"] = np.NaN
    norm_len = data_norm.shape[0]
    fail_len = data_fail.shape[0]
    for i in range(norm_len):
        data_feat.loc[(data_norm['startTime'][i] <= data_feat["time"]) &
                      (data_feat["time"] <= data_norm['endTime'][i]), "label"] = 0
    for j in range(fail_len):
        data_feat.loc[(data_fail['startTime'][j] <= data_feat["time"]) &
                      (data_feat["time"] <= data_fail['endTime'][j]), "label"] = 1

if __name__ == '__main__':
    data_15, norm_15, fail_15 = get_data(15)
    data_21, norm_21, fail_21 = get_data(21)
    add_label(data_15, norm_15, fail_15)
    add_label(data_21, norm_21, fail_21)
    data_15['label'].value_counts(dropna=False)
    data_21['label'].value_counts(dropna=False)

