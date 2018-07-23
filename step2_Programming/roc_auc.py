# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 00:16:29 2018

@author: pb061
Programing to calculate AUC and learning curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler 
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.learning_curve import learning_curve,validation_curve
from sklearn import grid_search,linear_model,svm
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score,roc_curve, auc  
from imblearn.combine import SMOTEENN

def plot_validation_curve(estimator, X, y, param_name, param_range,
                          ylim=(0, 1.1), cv=3, n_jobs=4, scoring=None):
    estimator_name = type(estimator).__name__
    plt.title("Validation curves for %s on %s"
              % (param_name, estimator_name))
    plt.ylim(*ylim); plt.grid()
    plt.xlim(min(param_range), max(param_range))
    plt.xlabel(param_name)
    plt.ylabel("Score")

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range,
        cv=cv, n_jobs=n_jobs, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    print("Best test score: {:.4f}".format(test_scores_mean[-1]))


#----------------------------------------------------------------------------------
print("_________________________________________")
print("Get dataset and use SMOTE to oversampling ")
#data2 = './temp22.xlsx'
data2 = r"E:\EconometricsTermPaper\step2_Programing\temp22.xlsx"
base = pd.read_excel(data2)
scaler = StandardScaler()
scaler.fit(base)
trans_data = scaler.transform(base)

X = base.iloc[:,1:]
y = base['lemon']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, 
                                                                     y, 
                                                                     test_size=.8,
                                                                     random_state=0) 
y_test = pd.DataFrame(y_test) 
#sm = SMOTEENN()
#X_train,y_train = sm.fit_sample(X_train,y_train)

#----------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:20:07 2018

@author: Administrator
"""

print("_________________________________________")
print("评估CART模型的预测能力")
clf2 = DecisionTreeClassifier(max_depth=8)
scores = cross_validation.cross_val_score(clf2, X_train, y_train, cv=5, scoring='roc_auc')
print("ROC AUC CART: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))

tree = DecisionTreeClassifier()
tree = tree.fit(X_train,y_train)
probas_tree = tree.predict_proba(X_test)[:, 1]
fpr_tree,tpr_tree,thresholds = roc_curve(y_test,probas_tree,pos_label=1)

tree8 = DecisionTreeClassifier(max_depth=8)
tree8 = tree8.fit(X_train,y_train)
probas_tree8 = tree8.predict_proba(X_test)[:, 1]
fpr_tree8,tpr_tree8,thresholds = roc_curve(y_test,probas_tree8,pos_label=1)

print("_________________________________________")
print("评估RF模型的预测能力")
clf = RandomForestClassifier()
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
print("ROC AUC FOREST: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))

rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
probas_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf,tpr_rf,thresholds = roc_curve(y_test,probas_rf,pos_label=1)

gradrf = RandomForestClassifier(max_depth = 12,
                                max_features = 5,
                                n_estimators = 31)
grad = gradrf.fit(X_train,y_train)
probas_gradrf = gradrf.predict_proba(X_test)[:, 1]
fpr_gradrf,tpr_gradrf,thresholds = roc_curve(y_test,probas_gradrf,pos_label=1)

#----------------------------------------------------------------------------------
print("_________________________________________")
print("利用validation_curve计算不同深度训练集和测试集交叉验证得分")
clf = DecisionTreeClassifier(max_depth=8)
param_name = 'max_depth'
param_range = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15]

plot_validation_curve(clf, X_train, y_train,
                      param_name, param_range, scoring='roc_auc')

print("_________________________________________")
print("随机森林以及调参")
print("1. 先利用随机森里来提升分类效果:")
clf = RandomForestClassifier(n_estimators=27, 
                             max_features=15, # 参数n_estimators设置森林中树的个数
                             max_depth=10)
scores = cross_validation.cross_val_score(clf, 
                                          X_train, y_train, 
                                          cv=3, scoring='roc_auc',
                                          n_jobs=4)
print("ROC Random Forest: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))

print("2. 再利用梯度法进行调参")
parameters = {'n_estimators':[5,11,15,21,25,31], 
              'max_features':[5, 10,15,20],
              'max_depth':[3,6,9,12],
              'criterion':['gini','entropy']}
clf = grid_search.GridSearchCV(RandomForestClassifier(), 
                               parameters, 
                               cv=3)
clf.fit(X_train, y_train)

grid_search.GridSearchCV(cv=3, error_score='raise',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=4,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
       fit_params={}, iid=True, n_jobs=4,
       param_grid={'n_estimators': [5, 11, 15, 21, 25, 31], 
                   'max_features': [5, 10, 15, 20], 
                   'criterion': ['gini', 'entropy'], 
                   'max_depth': [3, 6, 9, 12]},
       pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)

y_pred_proba = clf.predict_proba(X_test)[:, 1]
print("ROC AUC: %0.4f" % roc_auc_score(y_test, y_pred_proba))

print(clf.best_params_)
print(clf.best_score_)

fpr_gradrf,tpr_gradrf,thresholds = roc_curve(y_test,y_pred_proba,pos_label=4)

print("_________________________________________")
print("绘制ROC曲线")

plt.title('ROC Curves For Models')
plt.plot(fpr_tree,tpr_tree,label="ROC_Tree")
plt.plot(fpr_tree8,tpr_tree8,label="ROC_Tree with Max_Depth=8")
plt.plot(fpr_rf,tpr_rf,linewidth=2,label="ROC_RandomForest")
plt.plot(fpr_gradrf,tpr_gradrf,linewidth=2,label="ROC_RandomForest with grad_search")
plt.xlabel("false presitive rate")
plt.ylabel("true presitive rate")
plt.legend(loc=4)#图例的位置
plt.show()





















