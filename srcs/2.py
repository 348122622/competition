# coding=utf-8
import srcs.tools as tools
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


# train, test分割
def data_prep(data, size=0.3):
    if len(data.columns) == 29:  # 过采样数据集已经消去了时间列
        X = data.iloc[:, 1: -1]
    else:
        X = data.iloc[:, : -1]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=10)
    print("训练集大小：%d" % len(X_train))
    print("测试集大小：%d" % len(X_test))
    return X_train, X_test, y_train, y_test


# 构造过采样训练集
def oversample1(data):
    data_fail = data[data["label"] == 1]
    times = int(len(data) / len(data_fail)) - 1
    for i in range(times):
        data = data.append(data_fail)
    print("过采样正常样本大小：%d" % len(data[data["label"] == 0]))
    print("过采样结冰样本大小：%d" % len(data[data["label"] == 1]))
    return data


# 过采样，多种可选方式
def oversample(data, model):
    # model = SMOTE(random_state=0, n_jobs=-1)
    # model = ADASYN(random_state=0, n_jobs=-1)
    # model = RandomOverSampler(random_state=0)
    columns = data.columns[1: -1]
    X, y = model.fit_sample(data.iloc[:, 1: -1], data["label"])
    X = pd.DataFrame(X, columns=columns)
    y = pd.DataFrame(y, columns=["label"])
    print("过采样总样本大小：%d" % len(y))
    print("过采样正常样本大小：%d" % len(y[y["label"] == 0]))
    print("过采样故障样本大小：%d" % len(y[y["label"] == 1]))
    # 注意，返回的数据集已经去掉了时间列
    return pd.concat([X, y], axis=1)


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


def model(clf, X_train, y_train, test26, test33):
    clf.fit(X_train, y_train)
    y_p = clf.predict(X_train)
    tools.plot_cm(y_train, y_p)
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

    # 过采样训练集训练
    # X_train, X_test, y_train, y_test = data_prep(over)
    # clf1.fit(X_train, y_train)
    # y_p = clf1.predict(X_train)
    # tools.plot_cm(y_train, y_p)
    # y_p = clf1.predict(X_test)
    # tools.plot_cm(y_test, y_p)
    # y_p1 = clf1.predict(test26)
    # y_p2 = clf1.predict(test33)
    # output(y_p1, 26)
    # output(y_p2, 33)

    # 过采样全集训练
    # model = SMOTE(random_state=0, n_jobs=-1)
    # model = ADASYN(random_state=0, n_jobs=-1)
    model = RandomOverSampler(random_state=0)
    over = oversample(data, model)
    # over = oversample1(data)  # 没有消去时间列
    X = over.iloc[:, : -2]
    y = over["label"]

    # clf1.fit(X, y)
    # y_p = clf1.predict(X)
    # tools.plot_cm(y, y_p)
    test26 = test26.iloc[:, : -1]
    test33 = test33.iloc[:, : -1]
    y_p1 = clf2.predict(test26)
    y_p2 = clf2.predict(test33)
    output(y_p1, 26)
    output(y_p1, 26)

    predictors = [ 'yaw_position', 'pitch1_angle',
       'pitch2_angle', 'pitch3_angle', 'pitch1_moto_tmp', 'pitch2_moto_tmp',  'int_tmp', 'pitch1_ng5_tmp',
       'pitch2_ng5_tmp', 'pitch3_ng5_tmp']

    score = clf2.feature_importances_
    plt.figure()
    plt.bar(range(len(predictors)), score)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

    y_p1 = clf2.predict(test26[predictors])
    y_p2 = clf2.predict(test33[predictors])