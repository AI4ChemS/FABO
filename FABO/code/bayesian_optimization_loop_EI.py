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

def bo_run(X,y,nb_MOFs, nb_iterations, nb_initialization, which_acquisition, kernel_type, min_features = 5, max_features = 30, FABO=False, store_explore_exploit_terms=True, cross_val_stats=False, FS_method = 'spearman'):
    assert nb_iterations > nb_initialization
    assert which_acquisition in ['max y_hat', 'EI', 'max sigma']
    

    ids_acquired = np.random.choice(np.arange((nb_MOFs)), size=nb_initialization, replace=False)

    df_X = X.copy()
    df_y = y.copy() 
    
    if which_acquisition == "EI" and store_explore_exploit_terms:
        explore_exploit_balance = np.array([(np.nan, np.nan) for i in range(nb_iterations)])
    else:
        explore_exploit_balance = []

    feature_count = [] 
    error = []
    top_features_dict = dict()
    feature_error_dict = dict()
    
    for i in range(nb_initialization, nb_iterations):
        print("iteration:", i, end="\r")

        if FABO:

            top_features , feature_error = feature_selection_loop(df_X,df_y,ids_acquired, FS_method = FS_method, min_features = min_features, max_features = max_features)

            top_features_dict[i] = top_features
            feature_error_dict[i] = feature_error

            X = torch.from_numpy(np.array(df_X[top_features]))
            y = torch.from_numpy(np.array(df_y))
            y = y.unsqueeze(1)
            X_unsqueezed = X.unsqueeze(1)
            feature_count.append(len(top_features))
        else:
            X = torch.from_numpy(np.array(df_X))
            y = torch.from_numpy(np.array(df_y))
            y = y.unsqueeze(1)
            X_unsqueezed = X.unsqueeze(1)
    
        y_acquired = y[ids_acquired]
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)

        if cross_val_stats:
            if (i+1)%5 == 0 :
                error.append(five_fold_cross_val(X[ids_acquired, :],y_acquired))
        
        model = SingleTaskGP(X[ids_acquired, :], y_acquired)
        if kernel_type == 'Default':
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        else:
            custom_kernel = ScaleKernel(kernel_type)
            model.covar_module = custom_kernel
            likelihood = GaussianLikelihood()
            mll = ExactMarginalLogLikelihood(likelihood, model)
            fit_gpytorch_mll(mll)
            
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

        y_acquired = y[ids_acquired] # start over to normalize y properly
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
        
        
    assert np.size(ids_acquired) == nb_iterations
    return ids_acquired, explore_exploit_balance, feature_count, error, top_features_dict , feature_error_dict 


    
###############################################################################################################################

def main(X,y,nb_MOFs, nb_iterations, nb_initialization, which_acquisition, min_features, max_features, FABO, experiment_name, cross_val_stats=False, kernel_type = 'Default', i = 1, FS_method = 'spearman'):

    # Save Path
    save_path = experiment_name

    # RUN TRAINING
    which_acquisition = "EI"
    nb_initializations = {"EI": [nb_initialization], 
                            "max y_hat": [10], 
                            "max sigma": [10]}
    for nb_initialization in nb_initializations[which_acquisition]:
        print("# MOFs in initialization:", nb_initialization)
        # store results here.
        bo_res = dict() 
        bo_res['ids_acquired']            = []
        bo_res['explore_exploit_balance'] = []
        store_explore_exploit_terms = True
        t0 = time.time()
        
        ids_acquired, explore_exploit_balance, feature_count, err, top_features_dict , feature_error_dict = bo_run(X.copy(), y.copy(), nb_MOFs, nb_iterations, nb_initialization, which_acquisition,kernel_type, min_features, max_features, FABO, store_explore_exploit_terms=store_explore_exploit_terms,cross_val_stats=cross_val_stats, FS_method = FS_method)
        
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
    plot_info = {"Rank": rank, "IDS": ids_acquired, "Feature Count" : feature_count, "Error": err, "top_features": top_features_dict , "feature_error": feature_error_dict, "explore_exploit_balance" : bo_res['explore_exploit_balance']}

#     if os.path.isdir("Pickle_Files/" + "FABO_" + save_path) == False:
#         os.mkdir("Pickle_Files/" + "FABO_" + save_path)
#     file = open("Pickle_Files/" + "FABO_" + save_path + "/" + experiment_name + ".pkl", 'wb')
    file = open("Pickle_Files/" + experiment_name + ".pkl", 'wb')

    pickle.dump(plot_info,file)
    # close the file
    file.close()

    return output, rank, ids_acquired, feature_count, plot_info

###############################################################################################################################

def feature_evaluation(X,y,feature_set_name, user_name,features, nb_iterations = 250, seed=3,cross_val_stats=False):

    np.random.seed(seed)
    torch.manual_seed(seed)

    your_name = user_name
    feature_set = features
    experiment_name = feature_set_name
    nb_MOFs = len(X.index)

    X = X[features]

    print("Experiment Name: {}".format(experiment_name))
    print("===================")
    print("Features: {}".format(feature_set))
    print("shape of X:", np.shape(X))
    print("shape of y:", np.shape(y))
    print("# MOFs:", nb_MOFs)
    print("# iterations:", nb_iterations)
    print("# Random Seed", seed)

    which_acquisition = "EI"
    nb_initializations = {"EI": [10], 
                            "max y_hat": [10], 
                            "max sigma": [10]}
    for nb_initialization in nb_initializations[which_acquisition]:
        print("# MOFs in initialization:", nb_initialization)
        bo_res = dict() 
        bo_res['ids_acquired']            = []
        bo_res['explore_exploit_balance'] = []
        if nb_initialization == 10 and which_acquisition == 'EI':
            store_explore_exploit_terms = True
        else:
            store_explore_exploit_terms = False
        
        t0 = time.time()            
        ids_acquired, explore_exploit_balance, feature_count, err = bo_run(X,y,nb_MOFs,nb_iterations, nb_initialization, which_acquisition, store_explore_exploit_terms=store_explore_exploit_terms,cross_val_stats=cross_val_stats)
        
        bo_res['ids_acquired'].append(ids_acquired)
        bo_res['explore_exploit_balance'].append(explore_exploit_balance)
        
        print("took time t = ", (time.time() - t0) / 60, "min\n")

    import pickle
    ids = ids_acquired
    rankings = []
    in_top_250 = 0
    y = torch.tensor(y).unsqueeze(1)
    sorted = torch.sort(y[:,0],descending=True,axis=0)
    for i in ids:
        position = int(np.where(i == sorted.indices)[0])
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

    if len(min_ind) > 1:
        min_ind = min_ind[-1]
        warnings.warn("There may be multiple entries with the same label value")
        
    output = np.array([experiment_name] + [int(rankings[-1]) + 1] + [float(min_val)] + [int(min_ind)] + [int(found)] + [float(in_top_250/100)] + feature_set)
        

    return output, rankings, ids_acquired, err


###############################################################################################################################


def baseline(path, label, nb_iterations, n_seed):
    df_CORE160 = pd.read_csv(path)
    CO2_uptake = list(df_CORE160[label].values) 

    ranks = df_CORE160[label].rank(method='min', ascending=False)
    np.random.seed(n_seed)  
    num_rows = nb_iterations
    num_cols = n_seed
    value_range = df_CORE160.shape[0]-1
    data_random = np.zeros((num_rows, num_cols), dtype=int)
    for col in range(num_cols):
        data_random[:, col] = np.random.choice(range(1, value_range + 1), num_rows, replace=False)

    rank_list = []
    data_Random = {}
    for i in range(num_rows):
        temp_list = []
        for j in range(num_cols):
            temp_list.append(int(ranks[data_random[i,j]]))
        rank_list.append(temp_list)
    highest_rank_Random = np.array(rank_list)
    for i in range(1, highest_rank_Random.shape[0]):  
        for j in range(highest_rank_Random.shape[1]):  
            if highest_rank_Random[i, j] > highest_rank_Random[i - 1, j]:
                highest_rank_Random[i, j] = highest_rank_Random[i - 1, j]
    rank_avg = pd.Series(np.mean(highest_rank_Random, axis=1))
    id_methods = pd.DataFrame(data_random)
    return rank_avg, id_methods

