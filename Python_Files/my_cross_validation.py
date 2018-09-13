from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import linearsvm as my_clfs
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

def my_cross_validation(X,y, use_sklearn= True, N_SPLITS = 5, epsilon=0.0001):
    """
    Returns optimal lambda found, where lambdas searched
       are 10^-4 to 10^3
    Args:
        X (numpy array): Training matrix
        y (numpy array): training labels (can be {0,1} or {-1,1}) format
        use_sklearn (boolean): If TRUE, uses scikit learn LinearSVC().
                                  Else: uses Rahul Birmiwal's implemententation of
                                      weighted-Linear SVM (and/or multiclass ensemble)
        N_SPLITS (int): number of splits for the cross-validation
        epsilon (float): training tolerance / stopping criterion for gradient descent
    Returns:
        best_lamb (float): best lambda found (lambda that maximizes average prediction score)
    """

    kf = KFold(N_SPLITS)
    best_score = -99999
    best_lam = 0
    lambs = [0.0001]
    lambs = np.append(lambs, np.arange(0.1, 1, 0.1))
    lambs = np.append(lambs, [10,100,1000])
    for lam in lambs:
        sum_score = 0.0
        print("...Running classifier with lambda={}".format(lam))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if (use_sklearn):
                classifier = LinearSVC(penalty='l2', loss='squared_hinge', tol=epsilon, C=lam, class_weight='balanced')
                classifier.fit(X_train, y_train)
                score = classifier.score(X_test, y_test)
            else:

                classifier = my_clfs.LinearSVM(loss_function='huberized_hinge', lamb=lam, rho=0.8)
                y_train = np.where(y_train==0,-1,1)
                y_test = np.where(y_test==0,-1,1)
                classifier.fit(X_train.T, y_train)
                score = classifier.score(X_test.T, y_test)


            sum_score += score
        avg_score = sum_score / N_SPLITS

        print("Lambda {} had accuracy score of {}".format(lam, avg_score))
        if (avg_score > best_score):
            best_score = avg_score
            best_lam = lam
    return best_lam


def my_multi_cross_validation(X,y, use_sklearn= True, N_SPLITS = 5, epsilon=0.0001):
    """
    Returns optimal lambda found, where lambdas searched
       are 10^-4 to 10^3
    Args:
        X (numpy array): Training matrix
        y (numpy array): training labels (can be {0,1} or {-1,1}) format
        use_sklearn (boolean): If TRUE, uses scikit learn LinearSVC().
                                  Else: uses Rahul Birmiwal's implemententation of
                                      weighted-Linear SVM (and/or multiclass ensemble)
        N_SPLITS (int): number of splits for the cross-validation
        epsilon (float): training tolerance / stopping criterion for gradient descent
    Returns:
        best_lamb (float): best lambda found (lambda that maximizes average prediction score)
    """

    kf = KFold(N_SPLITS)
    best_score = -99999
    best_lam = 0
    lambs = [0.0001,0.01,1]
    lambs = np.append(lambs, [10,100,1000])
    for lam in lambs:
        sum_score = 0.0
        print("...Running classifier with lambda={}".format(lam))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if (use_sklearn):
                classifier = LinearSVC(penalty='l2', loss='squared_hinge', multi_class='ovr',tol=epsilon, C=lam, class_weight='balanced')
                classifier.fit(X_train, y_train)
                score = classifier.score(X_test, y_test)
            else:
                pass

            sum_score += score
        avg_score = sum_score / N_SPLITS

        print("Lambda {} had accuracy score of {}".format(lam, avg_score))
        if (avg_score > best_score):
            best_score = avg_score
            best_lam = lam
    return best_lam 
