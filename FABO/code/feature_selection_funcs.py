import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
# from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import mutual_info_regression
# import xgboost as xgb
from mrmr import mrmr_regression


def get_spearman_top_N(X,y,N):
    X.insert(0, 'Band Gap', y)
    correlation = X.copy()
    correlation = correlation.corr(method="spearman")["Band Gap"] # Compute using Spearman Correlation
    correlation = correlation.drop("Band Gap")
    spear = list(abs(correlation).nlargest(N).index)
    return(spear)


def get_mutual_info_top_N(X,y,N):
    mutual_info = mutual_info_regression(np.array(X),np.array(y))
    l = list(mutual_info.argsort()[-N:])
    Top = []
    colnames = X.columns

    for i in l:
        Top.append(colnames[i])
    return(Top)


def get_mrmr_top_N(X, y, N):
    selected_indices = mrmr_regression(X, y, K=N)  # Select top N features    
    selected_feature_names = X[selected_indices].columns.tolist()
    return selected_feature_names


def XGBOOST_Grid_Search(X,y):
    tunable = {}
    tunable["gamma"] = np.linspace(0,2,5)
    tunable["alpha"] = np.linspace(0,1000,5)
    tunable["max_depth"] = list(range(5,9,2))
    tunable["min_child_weight"] = np.linspace(0.5,1.5,5)
    xgb = GridSearchCV(XGBRegressor(tree_method="gpu_hist",predictor="gpu_predictor"),param_grid=tunable,cv=3)
    xgb.fit(X,y,verbose=3) 
    return xgb.best_params_


def XG_BOOST_Perm_Importance(X,y,hyperparameters,N):
    features = X.columns
    nb_MOFs = len(X.index)    
    X = np.array(X)
    y = np.array(y)    
    ids_train, ids_test = train_test_split(np.arange(nb_MOFs), train_size=0.8)
    print("# training: ", len(ids_train))
    print("# test: ", len(ids_test))
    X_train = X[ids_train, :]
    y_train = y[ids_train]
    X_test  = X[ids_test, :]
    y_test  = y[ids_test]
    print(X_train.shape)
    print(X_test.shape)
    xgb = XGBRegressor(alpha=hyperparameters["alpha"],gamma=hyperparameters["gamma"],max_depth=hyperparameters["max_depth"],min_child_weight=hyperparameters["min_child_weight"],tree_method = hyperparameters["tree_method"],predictor = hyperparameters["predictor"])
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = r2_score(y_test, y_pred)
    print("METRICS")
    print("RMSE: {}\nR2: {}".format(rmse,r2))
    feature_importances = permutation_importance(xgb, X_test, y_test)
    ind = np.argpartition(feature_importances["importances_mean"], -N)[-N:]
    top = features[ind]
    return(top)


def XGboost_feature_selection(X,y,hyperparameters,N):
    dtrain = xgb.DMatrix(X, label=y)
    num_round = 100
    bst = xgb.train(hyperparameters, dtrain, num_round)    
    importance = bst.get_score(importance_type='weight')
    importance_df = pd.DataFrame({'feature': list(importance.keys()), 'importance': list(importance.values())})    
    importance_df = importance_df.sort_values(by='importance', ascending=False)    
    acq_features = list(importance_df['feature'][:N].values)
    return acq_features



