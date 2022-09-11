import imp
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import KFold
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from base_class import *

# Base functions
# bootstrapping method for dividing dataset
def bootstrap_with_dup(xset, yset, seed):
    m = len(xset[:,0]); xtrain_set, ytrain_set  = [], [];  indexlist = []
    if seed != None: np.random.seed(seed); seeds = np.arange(10*m); np.random.shuffle(seeds)
    for k in range(m):
        np.random.seed(seeds[k])
        random_index = np.random.randint(0, m)
        xtrain_set.append(xset[random_index,:])  
        ytrain_set.append(yset[random_index,:])  
        indexlist.append(random_index)
    x_train_set = np.array(xtrain_set)
    y_train_set = np.array(ytrain_set)
    indexlist = np.unique(np.array(indexlist))
    x_test_set = np.delete(xset, indexlist, axis = 0)
    y_test_set = np.delete(yset, indexlist, axis = 0)
    return x_train_set, x_test_set, y_train_set, y_test_set


def bootstrap(xset, yset, seed):
    m = len(xset[:,0]);   indexlist = []
    if seed != None: np.random.seed(seed); seeds = np.arange(10*m); np.random.shuffle(seeds)
    for k in range(m):
        np.random.seed(seeds[k])
        random_index = np.random.randint(0, m) 
        indexlist.append(random_index)
    indexlist = np.unique(np.array(indexlist))
    x_train_set = xset[indexlist,:]
    x_test_set = np.delete(xset, indexlist, axis = 0)
    y_train_set = yset[indexlist,:]
    y_test_set = np.delete(yset, indexlist, axis = 0)
    return x_train_set, x_test_set, y_train_set, y_test_set

# KFold method for experiments
def k_fold(xset, yset, k, randomstate):
    splits = KFold(n_splits = k, shuffle=True, random_state= randomstate)
    x_train_set, x_test_set, y_train_set, y_test_set = [],[],[],[]
    for train_index, test_index in splits.split(xset):
        x_train_set.append(xset[train_index])
        x_test_set.append(xset[test_index])
        y_train_set.append(yset[train_index])
        y_test_set.append(yset[test_index])
    return x_train_set, x_test_set, y_train_set, y_test_set

# Random Forest method
def RF(data_x, data_y, method = 'default', test_size = 0.25, k = 10):
    if method == 'k-fold':
        forest_reg = RandomForestRegressor()
        cv_result = cross_validate(forest_reg, data_x, data_y, cv = k, scoring = 'r2', return_train_score= True, return_estimator = True)
        return cv_result
    if method == 'default':
        """split the train and test set"""
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size= test_size, random_state = 120)

    if method == 'bootstrap':
        x_train, x_test, y_train, y_test = bootstrap(data_x, data_y, seed = 150)

    forest_reg = RandomForestRegressor()
    forest_reg.fit(x_train, y_train)
    train_score = forest_reg.score(x_train, y_train)
    test_score = forest_reg.score(x_test, y_test)
    return forest_reg, train_score, test_score

# Linear Regression method
def LR(data_x, data_y, method = 'default', test_size = 0.25, k = 10):
    if method == 'k-fold':
        forest_reg = LinearRegression()
        cv_result = cross_validate(forest_reg, data_x, data_y, cv = k, scoring = 'r2', return_train_score= True, return_estimator = True)
        return cv_result
    if method == 'default':
        """split the train and test set"""
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size= test_size, random_state = 120)

    if method == 'bootstrap':
        x_train, x_test, y_train, y_test = bootstrap(data_x, data_y, seed = 150)

    forest_reg = LinearRegression()
    forest_reg.fit(x_train, y_train)
    train_score = forest_reg.score(x_train, y_train)
    test_score = forest_reg.score(x_test, y_test)
    return forest_reg, train_score, test_score

# Support Vector Regression method for dimension=1
def SVR1(data_x, data_y, method = 'default', test_size = 0.25, k = 10): # svr只能做一维的因变量
    if method == 'k-fold':
        forest_reg = SVR()
        cv_result = cross_validate(forest_reg, data_x, data_y, cv = k,scoring = 'r2', return_train_score= True, return_estimator = True)
        return cv_result
    if method == 'default':
        """split the train and test set"""
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size= test_size, random_state = 120)

    if method == 'bootstrap':
        x_train, x_test, y_train, y_test = bootstrap(data_x, data_y, seed = 150)

    forest_reg = SVR()
    forest_reg.fit(x_train, y_train)
    train_score = forest_reg.score(x_train, y_train)
    test_score = forest_reg.score(x_test, y_test)
    return forest_reg, train_score, test_score

# Zscore
def zscore(data):
    mean, std = np.zeros(len(data.columns)), np.zeros(len(data.columns))
    for i in range(len(data.columns)):
        mean[i] = np.mean(data.iloc[:,i]); std[i] = np.std(data.iloc[:,i])
        data.iloc[:,i] = (data.iloc[:,i] - mean[i])/std[i]
    return data, mean, std


# Turn pred values into integer
def turn2int(pred, threshold = 0.5):
    pred2 = pred.copy()
    for k in range(len(pred)):
        pred2[k] = 0 if pred[k] < threshold else 1
    return pred2



"""Figures"""

def scatter_plot_TM_CW(data, col_x, pred = None, backend = 'Temperature'):
    fig, ax = plt.subplots(len(col_x), figsize = (6,50))
    if backend == 'Temperature':
        for k in range(len(col_x)):
            chem = col_x[k]
            df_chem_T = data[['Temperature', chem]]
            sns.scatterplot(x = 'Temperature', y = chem, data = df_chem_T, ax = ax[k])
            if isinstance(pred ,np.ndarray):
                sns.scatterplot(x = pred[:,1], y = data[chem], ax = ax[k])
        plt.tight_layout()
        plt.savefig("temperature+distribute.png")
    if backend != 'Temperature':
        for k in range(len(col_x)):
            chem = col_x[k]
            df_chem_T = data[['Coffe/WaterRatio', chem]]
            sns.scatterplot(x = 'Coffe/WaterRatio', y = chem, data = df_chem_T, ax = ax[k])
            if isinstance(pred ,np.ndarray):
                sns.scatterplot(x = pred[:,0], y = data[chem], ax = ax[k])
        plt.tight_layout()
        plt.savefig("Coffe/WaterRatio+distribute.png")        

