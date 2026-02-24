from __future__ import annotations

import ast
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import feature_selection as fs


def _safe_literal_eval(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def _to_2d_y(y: torch.Tensor) -> torch.Tensor:
    return y.unsqueeze(1) if y.ndim == 1 else y


def _get_acquired_pos(df_y: pd.DataFrame) -> np.ndarray:
    if "value" not in df_y.columns:
        raise ValueError("labels dataframe must contain a 'value' column.")
    return np.where(~df_y["value"].isna().to_numpy())[0]


def feature_selection_loop(
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    acquired_pos: np.ndarray,
    *,
    FS_method: str = "mRMR",
    min_features: int = 5,
    max_features: int = 30,
) -> Tuple[List[str], float]:
    X_acq = df_X.iloc[acquired_pos, :].copy()
    y_acq = df_y.iloc[acquired_pos]["value"].copy()

    if X_acq.shape[1] > 3:
        desc = X_acq.iloc[:, :-3]
        fracs = X_acq.iloc[:, -3:]
        desc = (desc - desc.mean()) / desc.std().replace(0, 1)
        X_acq = pd.concat([desc, fracs], axis=1)

    y_std = y_acq.std()
    y_acq = (y_acq - y_acq.mean()) / (y_std if y_std != 0 else 1.0)

    error_dict = {}
    feat_dict = {}

    for N in range(min_features, max_features + 1):
        if FS_method == "mRMR":
            feats = fs.get_mrmr_top_N(X_acq.copy(), y_acq.copy(), N)
        elif FS_method == "spearman":
            feats = fs.get_spearman_top_N(X_acq.copy(), y_acq.copy(), N)
        elif FS_method == "RF":
            feats = fs.get_rf_top_N(X_acq.copy(), y_acq.copy(), N, n_estimators=50, max_depth=10)
        else:
            raise ValueError(f"Unknown FS_method: {FS_method}")

        feat_dict[N] = feats

        X_full = torch.from_numpy(df_X[feats].to_numpy(dtype=np.float32))
        y_full = torch.from_numpy(df_y["value"].to_numpy(dtype=np.float32))
        y_full = _to_2d_y(y_full)

        X_train = X_full[acquired_pos]
        y_train = y_full[acquired_pos]

        err = five_fold_cross_val(X_train, y_train)
        error_dict[N] = float(err)

    best_N = min(error_dict, key=error_dict.get)
    return feat_dict[best_N], float(error_dict[best_N])


def five_fold_cross_val(X: torch.Tensor, y: torch.Tensor) -> float:
    n = X.shape[0]
    if n < 10:
        return float(eval_gp_model(X, y, X, y))

    fold_size = n // 5
    errs = []
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n
        X_val, y_val = X[start:end], y[start:end]
        X_tr = torch.cat([X[:start], X[end:]], dim=0)
        y_tr = torch.cat([y[:start], y[end:]], dim=0)
        errs.append(eval_gp_model(X_tr, y_tr, X_val, y_val))
    return float(sum(errs) / len(errs))


def eval_gp_model(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
    model = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    with torch.no_grad():
        post = model.posterior(X_val)
        pred = post.mean.detach().cpu().numpy()
        true = y_val.detach().cpu().numpy()
        return float(np.mean(np.abs(true - pred)))


def bo_run(
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    linearity_flag: Sequence[bool],
    which_acquisition: str,
    *,
    FABO: bool = False,
    FS_method: Optional[str] = None,
    min_features: Optional[int] = None,
    max_features: Optional[int] = None,
    kernel_function="default",
    syn_effect_AF: bool = False,
    teta: float = 0.5,
    n_batch: int = 1,
    AF_batch_size: int = 10000,
    default_top_features: Optional[List[str]] = None,
) -> Tuple[List[int], List[str], float]:

    assert which_acquisition in ["max y_hat", "EI", "max sigma"]

    acquired_pos = _get_acquired_pos(df_y)
    if acquired_pos.size == 0:
        raise ValueError("No acquired labels found (df_y['value'] all NaN).")

    feature_error = 0.0
    if FABO:
        if FS_method is None or min_features is None or max_features is None:
            raise ValueError("FABO=True requires FS_method, min_features, max_features.")
        top_features, feature_error = feature_selection_loop(
            df_X, df_y, acquired_pos, FS_method=FS_method, min_features=min_features, max_features=max_features
        )

        # Expand base features across cmp1..3 + mole fractions
        filtered = [f for f in top_features if not f.startswith("mole_fraction")]
        base_features = sorted(set(f.rsplit("_", 1)[0] for f in filtered))
        top_features = [f"{feat}_cmp{i}" for i in range(1, 4) for feat in base_features]
        top_features += ["mole_fraction_cmp1", "mole_fraction_cmp2", "mole_fraction_cmp3"]
    else:
        top_features = default_top_features or list(df_X.columns)

    X = torch.from_numpy(df_X[top_features].to_numpy(dtype=np.float32))
    y = torch.from_numpy(df_y["value"].to_numpy(dtype=np.float32))
    y = _to_2d_y(y)

    acq_idx = torch.from_numpy(acquired_pos.astype(np.int64))
    y_acq = y[acq_idx]
    mu, sd = y_acq.mean(), y_acq.std()
    if torch.isclose(sd, torch.tensor(0.0, device=sd.device)):
        sd = torch.tensor(1.0, device=sd.device)
    y_std = (y - mu) / sd
    y_acq_std = y_std[acq_idx]
    X_acq = X[acq_idx]

    if kernel_function == "default":
        model = SingleTaskGP(X_acq, y_acq_std)
    else:
        model = SingleTaskGP(X_acq, y_acq_std, covar_module=kernel_function)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    Xq = X.unsqueeze(1)  # [N, 1, d]

    with torch.no_grad():
        if which_acquisition == "EI":
            acq = ExpectedImprovement(model, best_f=y_acq_std.max().item())
            vals = []
            for start in range(0, Xq.shape[0], AF_batch_size):
                vals.append(acq(Xq[start : start + AF_batch_size]))
            acquisition_values = torch.cat(vals, dim=0).squeeze()
        elif which_acquisition == "max y_hat":
            acquisition_values = model.posterior(Xq).mean.squeeze()
        elif which_acquisition == "max sigma":
            acquisition_values = model.posterior(Xq).variance.squeeze()
        else:
            raise RuntimeError("Invalid acquisition function.")

    if syn_effect_AF:
        mask = torch.tensor(np.asarray(linearity_flag, dtype=bool))
        acquisition_values = acquisition_values.clone()
        acquisition_values[mask] *= (1.0 - float(teta))

    acquired_set = set(acquired_pos.tolist())
    ids_sorted = acquisition_values.argsort(descending=True).tolist()
    queries_ids = [i for i in ids_sorted if i not in acquired_set][: int(n_batch)]

    return queries_ids, top_features, float(feature_error)


def main_bo(args):
    df = pd.read_csv(args.df_path)
    compounds = pd.read_csv(args.compounds_path)
    df_X = pd.read_csv(args.df_featurized_path)
    df_y = pd.read_csv(args.labels_path)

    linearity_flag = df.get("linearity_flag", pd.Series([False] * len(df))).to_numpy(dtype=bool)

    queries_ids, top_features, feature_error = bo_run(
        df_X,
        df_y,
        linearity_flag,
        args.which_acquisition,
        FABO=args.FABO,
        FS_method=args.FS_method,
        min_features=args.min_features,
        max_features=args.max_features,
        kernel_function=args.kernel_function,
        syn_effect_AF=args.syn_effect_AF,
        teta=args.teta,
        n_batch=args.n_batch,
        AF_batch_size=args.AF_batch_size,
    )

    selected = df.iloc[queries_ids]
    cmp_ids = _safe_literal_eval(selected.iloc[0]["cmp_ids"])
    fracs = _safe_literal_eval(selected.iloc[0]["cmp_mole_fractions"])

    cmp_ids = tuple(cmp_ids) + (None, None, None)
    fracs = tuple(fracs) + (0.0, 0.0, 0.0)

    cmp0, cmp1, cmp2 = cmp_ids[:3]
    x1, x2, x3 = fracs[:3]

    # index compounds by id if needed
    if "id" in compounds.columns and compounds.index.name != "id":
        compounds = compounds.set_index("id")

    name0 = compounds.loc[cmp0, "name"] if cmp0 in compounds.index else None
    name1 = compounds.loc[cmp1, "name"] if cmp1 in compounds.index else None
    name2 = compounds.loc[cmp2, "name"] if cmp2 in compounds.index else None

    return name0, name1, name2, x1, x2, x3, queries_ids, top_features
