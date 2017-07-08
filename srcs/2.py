# coding=utf-8
import srcs.tools as tools
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train_path = '..\\data2\\train'
test_path = '..\\data2\\test'


def run_time(func):
    def wrapper(num):
        start = time.time()
        data_feat, data_fail = func(num)
        end = time.time()
        print("->读取%d号风机数据耗时: %.2f s" % (num, (end - start)))
        return data_feat, data_fail
    return wrapper


@run_time
def get_data(num):
    num = str(num)
    data_feat = pd.read_csv('%s\\%s\\%s_data.csv' % (train_path, num, num), parse_dates=["time"])
    data_fail = pd.read_csv('%s\\%s\\%s_failureInfo.csv' % (train_path, num, num), parse_dates=["t0", "t1", "t2", "t3"])
    return data_feat, data_fail


def get_test(num):
    num = str(num)
    test = pd.read_csv('%s\\%s\\%s_data.csv' % (test_path, num, num), index_col="time")
    return test


def add_label(data_feat, data_fail):
    data_feat["label"] = np.NaN
    fail_len = data_fail.shape[0]
    for j in range(fail_len):
        data_feat.loc[(data_fail['t2'][j] <= data_feat["time"]) &
                      (data_feat["time"] <= data_fail['t1'][j]), "label"] = 1
    return data_feat.fillna(0)


def output(y_p, num):
    # 测试集中的time = index + 1, 重置结果索引
    y_p = pd.Series(y_p, index=[x for x in range(1, len(y_p)+1)])
    fail_index = np.array(y_p[y_p == 1].index)
    n = len(fail_index)
    ans = []
    cur = 0
    for i in range(1, n):
        if fail_index[i] - fail_index[i-1] > 10 or i == n-1:
            if i - cur > 1:
                ans.append([fail_index[i-1], fail_index[cur]])
            cur = i
    if not ans:
        ans = [['NA', 'NA']]
    result = pd.DataFrame(ans, columns=["t1", "t2"])
    if num == 26:
        result.to_csv('..\\upload\\belt_test1\\test1_26_results.csv', index=False)
    elif num == 33:
        result.to_csv('..\\upload\\belt_test1\\test1_33_results.csv', index=False)
    return result


def model(clf, train_X, train_y, test26, test33):
    clf.fit(train_X, train_y)
    y_p = clf.predict(train_X)
    tools.plot_cm(train_y, y_p)
    return clf.predict(test26), clf.predict(test33)


if __name__ == '__main__':
    data_12, fail_12 = get_data(12)
    data_23, fail_23 = get_data(23)
    data_29, fail_29 = get_data(29)

    data12 = add_label(data_12, fail_12)
    data23 = add_label(data_23, fail_23)
    data29 = add_label(data_29, fail_29)

    X12 = data12.iloc[:, 1: -1]
    y12 = data12["label"]
    X23 = data23.iloc[:, 1: -1]
    y23 = data23["label"]
    X29 = data29.iloc[:, 1: -1]
    y29 = data29["label"]

    test26 = get_test(26)
    test33 = get_test(33)

    # gbm0 = GradientBoostingClassifier(random_state=10)
    # gbm0.fit(X23, y23)
    # y23_p = gbm0.predict(X23)
    # tools.plot_cm(y23, y23_p)
    # result0_26 = gbm0.predict(test26)
    # result0_33 = gbm0.predict(test33)
    #
    # gbm1 = GradientBoostingClassifier(random_state=10)
    # gbm1.fit(X29, y29)
    # y29_p = gbm1.predict(X29)
    # tools.plot_cm(y29, y29_p)
    # result26_1 = gbm1.predict(test26)
    # result33_1 = gbm1.predict(test33)
    #
    data = pd.concat([data12, data23, data29])
    X = data.iloc[:, 1: -1]
    y = data["label"]
    #
    # gbm = GradientBoostingClassifier(random_state=10)
    # gbm.fit(X, y)
    # y_p = gbm.predict(X)
    # tools.plot_cm(y, y_p)
    # result26 = gbm.predict(test26)
    # result33 = gbm.predict(test33)

    clf0 = RandomForestClassifier(random_state=1)
    clf1 = GradientBoostingClassifier(random_state=1)
    clf2 = xgb.XGBClassifier()
