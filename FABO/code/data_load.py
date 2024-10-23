import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


def remove_repeated_col(df):
     from itertools import combinations
     repeating = []
     for i,j in combinations(df, 2):
          if df[i].equals(df[j]):
               repeating.append(j)

     for i in repeating:
          df = df.drop(i,axis=1)
     return df

def read_data(file_name, standardize=True, to_drop = ["number", "MOF"], label="BG_PBE", normalize=True):

     X = pd.read_csv(file_name)
     for i in to_drop:
          if i in X:
               X = X.drop(i,axis=1)
     if label in X.columns and label != None:
          y = X.pop(label)
     else:
          y = None
     colnames = X.columns
     if standardize:
          from sklearn.preprocessing import StandardScaler
          scaler = StandardScaler()
          print(scaler.fit(X))
          X = scaler.transform(X)
     X = np.array(X)     
     if normalize:
          for i in range(np.shape(X)[1]):
               X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))

     X = pd.DataFrame(X,columns=colnames)
     X = X.dropna(axis=1)
     X = X.loc[:, (X != 0).any(axis=0)]
     return X, y

