# coding=utf-8
import srcs.tools as tools
import time
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics, cross_validation
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train_path = '..\\data1\\train'
test_path = '..\\data1\\test\\08\\08_data.csv'


# 计时器
def run_time(func):
    def wrapper(num):
        start = time.time()
        data_feat, data_norm, data_fail = func(num)
        end = time.time()
        print("->读取%d号风机数据耗时: %.2f s" % (num, (end - start)))
        return data_feat, data_norm, data_fail
    return wrapper


# 导入数据
@run_time
def get_data(num):
    num = str(num)
    data_feat = pd.read_csv('%s\\%s\\%s_data.csv' % (train_path, num, num), parse_dates=["time"])
    data_norm = pd.read_csv('%s\\%s\\%s_normalInfo.csv' % (train_path, num, num), parse_dates=["startTime", "endTime"])
    data_fail = pd.read_csv('%s\\%s\\%s_failureInfo.csv' % (train_path, num, num), parse_dates=["startTime", "endTime"])
    return data_feat, data_norm, data_fail


# 合并三个文件，加标签
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
    # 舍去标签缺失列, 重置索引
    return data_feat.dropna().reset_index(drop=True)


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


# 构造欠采样训练集
def undersample(data, times=1):
    # 获取正负样本索引
    fail_index = np.array(data[data["label"] == 1].index)
    fail_num = len(fail_index)
    norm_index = np.array(data[data["label"] == 0].index)
    # 默认1：1欠采样， rate设定欠采样比率
    np.random.seed(1)
    undersample_norm_index = np.random.choice(norm_index, times * fail_num, replace=False)
    # undersample_norm_index = np.array(undersample_norm_index)
    undersample_index = np.hstack((undersample_norm_index, fail_index))
    undersample_index.sort()
    undersample_data = data.iloc[undersample_index, :]
    print("欠采样正常样本大小：%d" % len(undersample_norm_index))
    print("欠采样结冰样本大小：%d" % len(fail_index))
    return undersample_data


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
    print("过采样结冰样本大小：%d" % len(y[y["label"] == 1]))
    # 注意，返回的数据集已经去掉了时间列
    return pd.concat([X, y], axis=1)


# 获取8号风机测试集
def get_test():
    test = pd.read_csv(test_path, index_col="time")
    return test


# 生成结果
def output(y_p):
    # 测试集中的time = index + 1, 重置结果索引
    y_p = pd.Series(y_p, index=[x for x in range(1, len(y_p)+1)])
    fail_index = np.array(y_p[y_p == 1].index)
    n = len(fail_index)
    ans = []
    cur = 0
    for i in range(1, n):
        if fail_index[i] - fail_index[i-1] > 10 or i == n-1:
            if i - cur > 1:
                ans.append([fail_index[cur], fail_index[i-1]])
            cur = i
    result = pd.DataFrame(ans, columns=["startTime", "endTime"])
    result.to_csv('..\\upload\\test1_08_results.csv', index=False)
    return result


# 获取评分
def get_score(y, y_p):
    cnf = metrics.confusion_matrix(y, y_p)
    p = len(y[y == 0])
    n = len(y[y == 1])
    fn = cnf[0][1]
    fp = cnf[1][0]
    score = 1 - fn * 0.5 / float(p) - fp * 0.5 / float(n)
    return score


if __name__ == '__main__':
    data_15, norm_15, fail_15 = get_data(15)
    data_21, norm_21, fail_21 = get_data(21)
    data15 = add_label(data_15, norm_15, fail_15)
    data21 = add_label(data_21, norm_21, fail_21)
    test = get_test()
    test = test.iloc[:, :-1]  # del group
    # 查看正负样本数目-->非平衡数据集
    # 15号风机label：
    # 0.0: 350255
    # 1.0: 23892
    # data_15["label"].value_counts(dropna=False)
    # data_21["label"].value_counts(dropna=False)

    
    # 1:1欠采样
    # X15, y15 = get_train(data15)
    # X21, y21 = get_train(data21)
    # X_train, X_test, y_train, y_test = train_test_split(X15, y15, test_size=0.3, random_state=1)
    # y_train:    0.0    16747      y_test:     0.0    7145
    #             1.0    16701                  1.0    7191

    # 随机森林
    # forest = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    # forest.fit(X_train, y_train)
    # y_pred = forest.predict(X_train)
    # tools.plot_cm(y_train, y_pred)
    
    X15 = data15.iloc[:, 1: -1]
    y15 = data15["label"]
    X21 = data21.iloc[:, 1: -1]
    y21 = data21["label"]

    # # rf0: 训练15号的数据生成的模型
    # rf0 = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    # rf0.fit(X15, y15)
    # y21_p = rf0.predict(X21)
    # tools.plot_cm(y21, y21_p)
    # # rf1: 训练21号的数据生成的模型
    # rf1 = RandomForestClassifier(random_state=1, n_jobs=-1)
    # rf1.fit(X21, y21)
    # y15_p = rf1.predict(X15)
    # tools.plot_cm(y15, y15_p)
    #
    # gbm0 = GradientBoostingClassifier(random_state=10)
    # gbm0.fit(X15, y15)
    # y_pred = gbm0.predict(X21)
    # y_predprob = gbm0.predict_proba(X21)[:, 1]
    # print("Accuracy : %.4g" % accuracy_score(y21.values, y_pred))
    # print("AUC Score (Train): %f" % roc_auc_score(y21, y_predprob))
    # tools.plot_cm(y21, y_pred)
    #
    # param_test1 = {'n_estimators': range(20, 81, 10)}
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
    #                         min_samples_leaf=20, max_depth=8, max_features='sqrt', subsample=0.8, random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    # gsearch1.fit(X15, y15)
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    # 合并15，21数据
    data = pd.concat([data15, data21])
    X = data.iloc[:, 1: -2]
    y = data["label"]

    # 热图，相关系数矩阵
    # corrmat = data15.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=.8, square=True)
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    #
    # plt.figure()
    # k = 10
    # cols = corrmat.nlargest(k, 'label')['label'].index
    data = pd.concat([data15, data21])
    data = data.iloc[:, 1:]
    cm = np.corrcoef(data[data.columns].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                yticklabels=data.columns.values,xticklabels=data.columns.values)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    clf0 = RandomForestClassifier(random_state=1)
    clf1 = GradientBoostingClassifier(random_state=1)
    clf2 = XGBClassifier()
    clf3 = LogisticRegression()

    # # 训练集训练
    # X_train, X_test, y_train, y_test = data_prep(over)
    # clf1.fit(X_train, y_train)
    # y_p = clf1.predict(X_train)
    # tools.plot_cm(y_train, y_p)
    # y_p = clf1.predict(X_test)
    # tools.plot_cm(y_test, y_p)
    # y_p = clf1.predict(test)
    # output(y_p)

    # # 全集训练
    # # model = SMOTE(random_state=0, n_jobs=-1)
    # # model = ADASYN(random_state=0, n_jobs=-1)
    # # model = RandomOverSampler(random_state=0)
    # over = oversample(data, model)
    # over = oversample1(data)  # 没有消去时间列
    #
    # X = over.iloc[:, :-1]
    # y = over.label
    #
    # clf1.fit(X, y)
    # y_p = clf1.predict(X)
    # get_score(y, y_p)
    # tools.plot_cm(y, y_p)
    # y_p = clf1.predict(test)
    # output(y_p)


    # model = SMOTE(random_state=0, n_jobs=-1)
    # model = ADASYN(random_state=0, n_jobs=-1)
    # model = RandomOverSampler(random_state=0)
    # over = oversample(data, model) # 已去掉时间列
    over = oversample1(data)
    X_o = over.iloc[:, 1: -2]
    y_o = over["label"]
    # X_train, X_test, y_train, y_test = data_prep(over)
    # clf4 = XGBClassifier(
    #     max_depth=5,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     seed=10)
    # param_range = range(40, 81, 10)
    # param_grid = {'n_estimators': param_range}
    # gs = GridSearchCV(clf4,
    #                   param_grid,
    #                   cv=5,
    #                   scoring='roc_auc')
    # param_test1 = {'n_estimators': range(10, 71, 10)}
    # gs = GridSearchCV(clf0,
    #                   param_test1,
    #                   cv=5,
    #                   scoring='roc_auc')
    # # modelfit(xgb1, X_train, y_train)
