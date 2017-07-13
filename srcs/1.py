# coding=utf-8
import srcs.tools as tools
import time
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from scipy.stats import pearsonr
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


# 合并15，21号风机的数据
def combine_data(*args, depart=True):
    df = pd.concat(args)
    if depart is False:
        return df
    df_x = df.iloc[:, :-1]
    df_y = df["label"]
    return df_x, df_y


# train, test分割
def data_prep(df, size=0.3):
    # if len(df.columns) == 29:  # 过采样数据集已经消去了时间列
    #     df_x = df.iloc[:, 1: -1]
    # else:
    df_x = df.iloc[:, : -1]
    df_y = df["label"]
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=size, random_state=10)
    print("训练集大小：%d" % len(df_x_train))
    print("测试集大小：%d" % len(df_x_test))
    return df_x_train, df_x_test, df_y_train, df_y_test


# 构造欠采样训练集
def undersample(df, times=1):
    # 获取正负样本索引
    fail_index = np.array(df[df["label"] == 1].index)
    fail_num = len(fail_index)
    norm_index = np.array(df[df["label"] == 0].index)
    # 默认1：1欠采样， rate设定欠采样比率
    np.random.seed(1)
    undersample_norm_index = np.random.choice(norm_index, times * fail_num, replace=False)
    # undersample_norm_index = np.array(undersample_norm_index)
    undersample_index = np.hstack((undersample_norm_index, fail_index))
    undersample_index.sort()
    undersample_df = df.iloc[undersample_index, :]
    print("欠采样正常样本大小：%d" % len(undersample_norm_index))
    print("欠采样结冰样本大小：%d" % len(fail_index))
    return undersample_df


# 构造过采样训练集
def oversample1(df):
    df_fail = df[df["label"] == 1]
    times = int(len(df) / len(df_fail)) - 1
    for i in range(times):
        df = df.append(df_fail)
    print("过采样正常样本大小：%d" % len(df[df["label"] == 0]))
    print("过采样结冰样本大小：%d" % len(df[df["label"] == 1]))
    df_x = df.iloc[:, : -1]
    df_y = df["label"]
    return df_x, df_y


# 过采样，多种可选方式
def oversample(df, method=0):
    methods = ["随机", "SMOTE", "ADASYN"]
    model = RandomOverSampler(random_state=0)
    if method == 0:
        methods = methods[0]
    elif method == 1:
        model = SMOTE(random_state=0, n_jobs=-1)
        methods = methods[1]
    elif method == 2:
        model = ADASYN(random_state=0, n_jobs=-1)
        methods = methods[2]
    columns = df.columns[1: -1]
    df_x, df_y = model.fit_sample(df.iloc[:, 1: -1], df["label"])
    df_x = pd.DataFrame(df_x, columns=columns)
    df_y = pd.DataFrame(df_y, columns=["label"])
    print("%s过采样总样本大小：%d" % (methods, len(df_y)))
    print("%s过采样正常样本大小：%d" % (methods, len(df_y[df_y["label"] == 0])))
    print("%s过采样结冰样本大小：%d" % (methods, len(df_y[df_y["label"] == 1])))
    # 注意，返回的数据集已经去掉了时间列
    # return pd.concat([df_x, df_y], adf_xis=1)
    return df_x, df_y


# 获取8号风机测试集
def get_test():
    df = pd.read_csv(test_path, index_col="time")
    return df


# 特征线性相关程度（皮尔逊相关系数）
def col_sim(df):
    features = [i for i in df.columns if i not in ["time", "group", "label"]]
    plt.figure()
    cm = np.corrcoef(df[features].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                yticklabels=features, xticklabels=features)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


# 获取评分
def get_score(df_y, y_pred):
    cnf = metrics.confusion_matrix(df_y, y_pred)
    p = len(df_y[df_y == 0])
    n = len(df_y[df_y == 1])
    fn = cnf[0][1]
    fp = cnf[1][0]
    pred_score = 1 - fn * 0.5 / float(p) - fp * 0.5 / float(n)
    return pred_score


# 生成结果
def output(y_pred):
    # 测试集中的time = index + 1, 重置结果索引
    y_pred = pd.Series(y_pred, index=[j for j in range(1, len(y_pred)+1)])
    fail_index = np.array(y_pred[y_pred == 1].index)
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

if __name__ == '__main__':
    data_15, norm_15, fail_15 = get_data(15)
    data_21, norm_21, fail_21 = get_data(21)
    data15 = add_label(data_15, norm_15, fail_15)
    data21 = add_label(data_21, norm_21, fail_21)
    test = get_test()
    # 查看正负样本数目-->非平衡数据集
    # 15号风机label：
    # 0.0: 350255
    # 1.0: 23892
    # data_15["label"].value_counts(dropna=False)
    # data_21["label"].value_counts(dropna=False)

    # X15 = data15.iloc[:, 1: -1]
    # y15 = data15["label"]
    # X21 = data21.iloc[:, 1: -1]
    # y21 = data21["label"]

    data = combine_data(data15, data21, depart=False)
    x0, y0 = oversample(data, method=0)
    x1, y1 = oversample(data, method=1)

    clf0 = RandomForestClassifier(random_state=1)
    clf1 = GradientBoostingClassifier(random_state=1)
    clf2 = XGBClassifier(max_depth=5)
    clf = XGBClassifier(learning_rate=0.05, n_estimators=100, max_depth=3, min_child_weight=4,
                        reg_alpha=0.005, colsample_bytree=0.8, colsample_bylevel=0.8)

    # 特征选择
    # score = clf2.feature_importances_
    # plt.figure()
    # plt.bar(range(len(predictors)), score)
    # plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    # plt.show()
    predictors = ['wind_speed', 'generator_speed', 'power', 'wind_direction', 'yaw_position', 'pitch1_angle',
                  'pitch2_angle', 'pitch3_angle',  'pitch1_moto_tmp', 'pitch2_moto_tmp', 'pitch3_moto_tmp',
                  'environment_tmp', 'int_tmp', 'pitch1_ng5_tmp', 'pitch2_ng5_tmp', 'pitch3_ng5_tmp']
    param_test = {'n_estimators': [160, 180, 200]}
    gs = GridSearchCV(estimator=clf2,
                      param_grid=param_test,
                      scoring='roc_auc',
                      n_jobs=-1,
                      cv=4)
    gs.fit(x0[predictors], y0)
    # print(gs.grid_scores_, gs.best_params_, gs.best_score_)






