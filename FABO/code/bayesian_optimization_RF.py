import os
import sys
import csv
import time
import torch
import pickle
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error as mae

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel

from code import data_load
from code import feature_selection_funcs





def feature_selection_loop(data,labels,ids_acquired, FS_method = 'mRMR', min_features = 5, max_features = 30):
    old_error = 1000000
    waiting = 0
    error_dict = {}
    feature_dict = {}
    for N in range(min_features,max_features):
        # Get top features
        if FS_method == 'mRMR':
            acq_features = feature_selection_funcs.get_mrmr_top_N(data.copy().loc[ids_acquired,:],labels.copy()[ids_acquired],N)
        elif FS_method == 'spearman':
            acq_features = feature_selection_funcs.get_spearman_top_N(data.copy().loc[ids_acquired,:],labels.copy()[ids_acquired],N)

        feature_dict[N] = acq_features
        
        X = np.array(data[acq_features])
        y = np.array(labels)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        y = y.unsqueeze(1)

        X_acquired = X[ids_acquired,:]

        y_acquired = y[ids_acquired]
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
        y_acquired = y[ids_acquired,:]

        error = five_fold_cross_val(X_acquired,y_acquired)
        
        error_dict[N] = error

    top_feature_size = min(error_dict, key=error_dict.get)
    top_features = feature_dict[top_feature_size]
    feature_error = error_dict[top_feature_size]
    

    return top_features , feature_error

###############################################################################################################################

def five_fold_cross_val(X_train,y_train):

    fold_size = int(X_train.shape[0]/5)

    X_fold_1 = X_train[:fold_size,:]
    X_fold_2 = X_train[fold_size:2*fold_size,:]
    X_fold_3 = X_train[2*fold_size:3*fold_size,:]
    X_fold_4 = X_train[3*fold_size:4*fold_size,:]
    X_fold_5 = X_train[4*fold_size:5*fold_size,:]

    y_fold_1 = y_train[:fold_size,:]
    y_fold_2 = y_train[fold_size:2*fold_size,:]
    y_fold_3 = y_train[2*fold_size:3*fold_size,:]
    y_fold_4 = y_train[3*fold_size:4*fold_size,:]
    y_fold_5 = y_train[4*fold_size:5*fold_size,:]

    folds = [(X_fold_1,y_fold_1),(X_fold_2,y_fold_2),(X_fold_3,y_fold_3),(X_fold_4,y_fold_4),(X_fold_5,y_fold_5)]

    error = []

    for i in range(len(folds)):
        X_val = folds[i][0]
        y_val = folds[i][1]

        X_train = torch.zeros((1,X_val.shape[1]))
        y_train = torch.zeros((1,y_train.shape[1]))
        for j in range(len(folds)):

            if j != i:
                X_train = torch.concatenate((X_train,folds[j][0]),axis=0)
                y_train = torch.concatenate((y_train,folds[j][1]),axis=0)
        X_train = X_train[1:,:]
        y_train = y_train[1:,:]
        error.append(eval_GP_model(X_train,y_train,X_val,y_val))
    return sum(error)/len(error)
    
###############################################################################################################################

def eval_GP_model(X_train,y_train,X_val,y_val):


    model = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    posterior = model.posterior(X_val)
    y_true = np.array(y_val)
    y_pred = posterior.mean.detach().numpy()

    return float(mae(y_true,y_pred))


###############################################################################################################################

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm


class RandomForestSurrogate:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='auto', bootstrap=True):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           max_features=max_features, bootstrap=bootstrap)
    
    def fit(self, X, y):
        self.model.fit(X, y)        
    
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = preds.mean(axis=0)
        variance = preds.var(axis=0)
        return mean, variance


def expected_improvement(X_candidates, model, y_max):
    mean, variance = model.predict(X_candidates)
    std = np.sqrt(variance)
    with np.errstate(divide='warn'):
        z = (mean - y_max) / std
        ei = (mean - y_max) * norm.cdf(z) + std * norm.pdf(z)
        ei[std == 0.0] = 0.0
    return ei

###############################################################################################################################

def bo_run(X,y,nb_MOFs, nb_iterations, nb_initialization):
    assert nb_iterations > nb_initialization
    

    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_initialization, replace=False)

    df_X = X.copy()
    df_y = y.copy() 
    
    
    rf_surrogate = RandomForestSurrogate()
    for i in range(nb_initialization, nb_iterations):
        print("iteration:", i, end="\r")

        X = np.array(df_X)
        y = np.array(df_y)
        y_acquired = y[ids_acquired]
        y_acquired = (y_acquired - y_acquired.mean()) / y_acquired.std()

        rf_surrogate.fit(X[ids_acquired], y_acquired)
        ids_candidates = np.setdiff1d(np.arange(nb_MOFs), ids_acquired)
        ids_candidates = np.setdiff1d(np.arange(nb_MOFs), ids_acquired)
        X_candidates = X[ids_candidates]
        y_max = y[ids_acquired].max()
        ei_values = expected_improvement(X_candidates, rf_surrogate, y_max)
        
        sorted_candidate_indices = np.argsort(-ei_values) 
        for idx in sorted_candidate_indices:
            next_query_idx = ids_candidates[idx]
            if next_query_idx not in ids_acquired:
                ids_acquired = np.append(ids_acquired, next_query_idx)
                break
        
        y_acquired = y[ids_acquired] 
        y_acquired = (y_acquired - y_acquired.mean()) / y_acquired.std()
        
        
    assert np.size(ids_acquired) == nb_iterations
    return ids_acquired


    
###############################################################################################################################

def main(X,y,nb_MOFs, nb_iterations, nb_initialization, experiment_name, i = 1):

    save_path = experiment_name

    

    print("# MOFs in initialization:", nb_initialization)
    bo_res = dict() 
    bo_res['ids_acquired']            = []
    t0 = time.time()
    ids_acquired = bo_run(X.copy(), y.copy(), nb_MOFs, nb_iterations, nb_initialization)
    bo_res['ids_acquired'].append(ids_acquired)

    print("took time t = ", 100 * ((time.time() - t0) / 60), "min\n")

    ids = ids_acquired
    rankings = []
    in_top_250 = 0

    y = torch.tensor(y).unsqueeze(1)

    sort = torch.sort(y[:,0],descending=True,axis=0)
    for i in ids:
        position = int(np.where(i == sort.indices)[0])
        if rankings == []:
            rankings = [position]
        elif position < rankings[-1]:
            rankings.append(position)
        else:
            rankings.append(rankings[-1])

        if position < 250:
            in_top_250 += 1

    min_val = max(y[ids_acquired][:,0])
    min_ind = np.where(y[:,0] == min_val)[0]
    found = np.where(y[ids_acquired][:,0] == min_val)[0]

    output = np.array([int(i)]+ [experiment_name] + [int(rankings[-1]) + 1] + [float(min_val)] + [int(min_ind)] + [int(found)] + [float(in_top_250/100)])

    rank = np.array(rankings)
    plot_info = {"Rank": rank, "IDS": ids_acquired}

    file = open("Pickle_Files/" + experiment_name + ".pkl", 'wb')

    pickle.dump(plot_info,file)
    file.close()

    return output, rank, ids_acquired, plot_info

###############################################################################################################################


