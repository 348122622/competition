# coding=utf-8
import pickle, pprint
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from sklearn .grid_search import GridSearchCV

def dump(data_pd):
    with open('creaditcard1.pkl', 'wb') as file:
        pickle.dump(data_pd, file)


def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def printing_Kfold_scores(x_train_data, y_train_data, visual=False):
    fold = KFold(len(y_train_data), 5, shuffle=False)
    # kf = KFold(n_splits=5)
    # fold = kf.get_n_splits(y_train_data)

    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        # if visual:
        #     print'-------------------------------------------'
        #     print'C parameter: ', c_param
        #     print'-------------------------------------------'
        #     print''

        recall_accs = []

        for iteration, indices in enumerate(fold, start=1):

            lr = LogisticRegression(C=c_param, penalty='l1')
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)

            # if visual:
            #     print 'iteration:', iteration, '召回率 = %f' % recall_acc

        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1

        # if visual:
        #     print''
        #     print'平均召回率: ', np.mean(recall_accs)
        #     print''

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    # if visual:
    #     print'*********************************************************************************'
    #     print'交叉验证结果显示最好的参数C为：', best_c
    #     print'*********************************************************************************'

    # print results_table
    # print '正则化参数c最优值:', best_c
    return best_c


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_cm(y, y_pred):
    cnf_matrix = confusion_matrix(y, y_pred)
    # np.set_printoptions(precision=2)
    plt.figure()
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names)


def recall_proba(y, y_pred_proba):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure(figsize=(10, 10))
    j = 1
    for i in thresholds:
        y_test_predictions = y_pred_proba[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y, y_test_predictions)
        np.set_printoptions(precision=2)
        print("Recall metric in the testing dataset: ", float(cnf_matrix[1, 1]) / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

        # Plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)


def tune_parm(X_undersample, y_undersample, X, y, xgbmodel, param_grid, cv_folds=5):
    gs = GridSearchCV(estimator=xgbmodel, param_grid=param_grid, scoring='roc_auc',
                      cv=cv_folds, n_jobs=-1)
    gs = gs.fit(X_undersample, y_undersample.values.ravel())
    print(gs.grid_scores_)
    print(gs.best_params_, gs.best_score_)
    y_pred = gs.best_estimator_.predict(X)
    print("Accuracy : %.4g" % accuracy_score(y.values.ravel(), y_pred))
    print("AUC Score : %f" % roc_auc_score(y.values.ravel(), y_pred))
    return gs.best_estimator_


def modelfit(xgbmodel, X_undersample, y_undersample, X, y):
    xgbmodel =xgbmodel.fit(X_undersample, y_undersample.values.ravel())
    y_pred = xgbmodel.predict(X)
    # y_predprob = xgbmodel.predict_proba(X)
    print("Accuracy : %.4g" % accuracy_score(y.values.ravel(), y_pred))
    print("AUC Score : %f" % roc_auc_score(y.values.ravel(), y_pred))
    return y_pred
