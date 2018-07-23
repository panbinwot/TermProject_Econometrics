# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:08:40 2018

@author: pb061
Programming for logit model
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from numpy.random import permutation
from sklearn import linear_model 
from numpy import array_split, concatenate
from sklearn.metrics import accuracy_score, classification_report,roc_curve, auc, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from imblearn.combine import SMOTEENN


class FraudDetection:
    def __init__(self, data_file):
        self.dataFrame = pd.read_excel(data_file)
        for k in self.dataFrame.columns[1:]:
            self.dataFrame[k], _ = pd.factorize(self.dataFrame[k])
        
        sorted_cats = sorted(pd.Categorical(self.dataFrame['lemon']).categories)
        self.classes = np.array(sorted_cats)
        self.features = self.dataFrame.columns[self.dataFrame.columns != 'lemon']

    def __factorize(self,data):
        y, _ = pd.factorize(pd.Categorical(data['lemon']), sort=True)
        return y
    
    def validation_data(self,folds):
        df = self.dataFrame
        response = []
        
        assert len(df) > folds
        
        perms = array_split(permutation(len(df)), folds)
        
        for  i in range(folds):
            train_idxs = list(range(folds))
            train_idxs.pop(i) 
            train = []
            for idx in train_idxs:
                train.append(perms[idx])
            
            train = concatenate(train)
            
            test_idx = perms[i]
            
            training = df.iloc[train]
            test_data = df.iloc[test_idx]
            
            y = self.__factorize(training)
            
            sm = SMOTEENN()                            #利用SMOTE算法扩容 
            axis = training[self.features]
            tt, y = sm.fit_sample(axis,y)
            axis = pd.DataFrame(np.array(tt)) #.ravel() transform ndarray to dataframe
                 
            classifier = self.train(axis, y)
            predictions = classifier.predict(test_data[self.features]) # predict the outcome
            
            expected = self.__factorize(test_data)
            response.append([predictions, expected])
            
        return response

class Logreg(FraudDetection):
    def train(self,X,Y):
        logreg = linear_model.LogisticRegression(C=1e5, 
                                                 solver='lbfgs', 
                                                 random_state=0)  
        classifier = logreg.fit(X, Y)       
        return classifier 
    
    def validate(self,folds):
        report = []
        #acc = []
        for y_true, y_pred in self.validation_data(folds):
            report.append(classification_report(y_true,y_pred))
            #acc.append(accuracy_score(y_true,y_pred))
        return report
    
class FraudClassifier(FraudDetection):
  """
  Partial implementation of frauds classification problem
  """

  def validate(self, folds):
    """
    Evaluate classifier using confusion matrices
    :param folds: number of folds
    :return: 
        1. list of confusion matrices per fold
        2. 5 pictures of importance
    """
    confusion_matrices = []

    for test, training in self.validation_data(folds):
      confusion_matrices.append(self.confusion_matrix(training, test))

    return confusion_matrices

  @staticmethod
  def confusion_matrix(train, test):
    return pd.crosstab(test, train, rownames=['actual'], colnames=['preds'])

  def figimp(importances,X,y,labels):
    indices = np.argsort(importances)[::-1]
    plt.title("Plot of Importances")
    plt.bar(range(X.shape[1]),
            importances[indices],
            color = 'lightblue',
            align='center')
    plt.xlim([-1,X.shape[1]])
    plt.tight_layout
    plt.xticks(range(X.shape[1]),labels,rotation=90)
    plt.show() 

class FraudForest(FraudClassifier):
  """
  Implementation of frauds classification problem with sklearn.RandomForestClassifier
  """
  def train(self, X, Y):
    """
    Train classifier.
    :param X: training input samples
    :param Y: target values
    :return: classifier / a picture of importance
    """
    classifier = RandomForestClassifier(n_jobs=4,
                                        random_state=0,
                                        n_estimators = 500,
                                        max_features= 5,
                                        max_depth = 12) # n_job=x, x is the number of working CPU
    classifier = classifier.fit(X, Y)
    
    importances = classifier.feature_importances_
    labels = self.dataFrame.columns[1:]
    FraudClassifier.figimp(importances,X,Y,labels)
    

    return classifier


class FraudTree(FraudClassifier):
  """
  Implementation of frauds classification problem with sklearn.DecisionTreeClassifier
  """

  def train(self, X, Y):
    """
    Train classifier.
    :param X: training input samples
    :param Y: target values
    :return: classifier
    """
    classifier = DecisionTreeClassifier(random_state=0,
                                        max_depth=40)
    classifier = classifier.fit(X, Y)
    
    importances = classifier.feature_importances_
    labels = self.dataFrame.columns[1:]
    FraudClassifier.figimp(importances,X,Y,labels)  
    dot_data = export_graphviz(classifier, 
                               out_file=None,
                               feature_names=X.columns)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("C:/Users/pb061/Desktop/output/tree.pdf")
    return classifier
    
    
    
    
    
    