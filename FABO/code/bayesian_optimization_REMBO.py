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

def bo_run(X,y,nb_MOFs, nb_iterations, nb_initialization, which_acquisition, store_explore_exploit_terms=True, cross_val_stats=False, lower_dim = 40):
    assert nb_iterations > nb_initialization
    assert which_acquisition in ['max y_hat', 'EI', 'max sigma']
    

    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_initialization, replace=False)

    df_X = X.copy()
    df_y = y.copy() 
    
    if which_acquisition == "EI" and store_explore_exploit_terms:
        explore_exploit_balance = np.array([(np.nan, np.nan) for i in range(nb_iterations)])
    else:
        explore_exploit_balance = []

    error = []
    # REMBO Initialization
    D = df_X.shape[1] # Original high dimension
    d = lower_dim   # Lower effective dimension
    A = np.random.randn(d, D)
    
    for i in range(nb_initialization, nb_iterations):
        print("iteration:", i, end="\r")


        df_X_low = np.dot(A, np.array(df_X.T)).T # Resulting shape will be (n, d)
        X = torch.from_numpy(df_X_low)
        y = torch.from_numpy(np.array(df_y))
        y = y.unsqueeze(1)
        X_unsqueezed = X.unsqueeze(1)
    
        y_acquired = y[ids_acquired]
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)

        if cross_val_stats:
            if (i+1)%5 == 0 :
                error.append(five_fold_cross_val(X[ids_acquired, :],y_acquired))
        
        model = SingleTaskGP(X[ids_acquired, :], y_acquired) 
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        if which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(model, best_f=y_acquired.max().item())
            acquisition_values = acquisition_function.forward(X_unsqueezed) 
        elif which_acquisition == "max y_hat":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).mean.squeeze()
        elif which_acquisition == "max sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).variance.squeeze()
        else:
            raise Exception("not a valid acquisition function")

        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        for id_max_aquisition_all in ids_sorted_by_aquisition:
            if not id_max_aquisition_all.item() in ids_acquired:
                id_max_aquisition = id_max_aquisition_all.item()
                break

        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
        assert np.size(ids_acquired) == i + 1
        
        if which_acquisition == "EI" and store_explore_exploit_terms:
            y_pred = model.posterior(X_unsqueezed[id_max_aquisition]).mean.squeeze().detach().numpy()
            sigma_pred = np.sqrt(model.posterior(X_unsqueezed[id_max_aquisition]).variance.squeeze().detach().numpy())
            
            y_max = y_acquired.max().item()
            
            z = (y_pred - y_max) / sigma_pred
            explore_term = sigma_pred * norm.pdf(z)
            exploit_term = (y_pred - y_max) * norm.cdf(z)            
            assert np.isclose(explore_term + exploit_term, acquisition_values[id_max_aquisition].item())
            explore_exploit_balance[i] = (explore_term, exploit_term)

        y_acquired = y[ids_acquired] 
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
        
        
    assert np.size(ids_acquired) == nb_iterations
    return ids_acquired, explore_exploit_balance, error


    
###############################################################################################################################

def main(X,y,nb_MOFs, nb_iterations, nb_initialization, which_acquisition, experiment_name, cross_val_stats=False, i = 1, lower_dim = 40):

    save_path = experiment_name

    which_acquisition = "EI"
    nb_initializations = {"EI": [nb_initialization], 
                            "max y_hat": [10], 
                            "max sigma": [10]}
    for nb_initialization in nb_initializations[which_acquisition]:
        print("# MOFs in initialization:", nb_initialization)
        bo_res = dict() 
        bo_res['ids_acquired']            = []
        bo_res['explore_exploit_balance'] = []
        store_explore_exploit_terms = True
        t0 = time.time()
        
        ids_acquired, explore_exploit_balance, err= bo_run(X.copy(), y.copy(), nb_MOFs, nb_iterations, nb_initialization, which_acquisition,  store_explore_exploit_terms=store_explore_exploit_terms,cross_val_stats=cross_val_stats, lower_dim = lower_dim)
        
        bo_res['ids_acquired'].append(ids_acquired)
        bo_res['explore_exploit_balance'].append(explore_exploit_balance)
        
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
    plot_info = {"Rank": rank, "IDS": ids_acquired, "Error": err, "explore_exploit_balance" : bo_res['explore_exploit_balance']}


    file = open("Pickle_Files/" + experiment_name + ".pkl", 'wb')
    pickle.dump(plot_info,file)
    file.close()

    return output, rank, ids_acquired, plot_info

###############################################################################################################################

