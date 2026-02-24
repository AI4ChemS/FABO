import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from mrmr import mrmr_regression


def get_spearman_top_N(X: pd.DataFrame, y: pd.Series, N: int):
    Xc = X.copy()
    Xc.insert(0, "_target_", y.values)
    corr = Xc.corr(method="spearman")["_target_"].drop("_target_")
    return list(corr.abs().nlargest(N).index)


def get_mutual_info_top_N(X: pd.DataFrame, y: pd.Series, N: int):
    mi = mutual_info_regression(X.to_numpy(), np.asarray(y))
    idx = np.argsort(mi)[-N:]
    return X.columns[idx].tolist()


def get_mrmr_top_N(X: pd.DataFrame, y: pd.Series, N: int):
    return mrmr_regression(X, y, K=N)


def get_rf_top_N(
    X: pd.DataFrame,
    y: pd.Series,
    N: int,
    n_estimators: int = 50,
    max_depth=None,
    max_features="sqrt",
    sample_size=None,
):
    if sample_size is not None:
        Xs = X.sample(n=min(sample_size, len(X)), random_state=42)
        ys = y.loc[Xs.index]
    else:
        Xs, ys = X, y

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(Xs, ys)
    importances = pd.Series(model.feature_importances_, index=Xs.columns)
    return importances.nlargest(N).index.tolist()
