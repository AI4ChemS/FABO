"""
Bayesian Optimization (BO) dataset curation and search-space initialization.

This module refactors the original utilities you shared in `curation.py`
(e.g., Mordred component featurization, dataset curation, and search-space init)
into a cleaner, configurable, and reusable API.

Typical usage in a notebook:
    from bo_curation import SearchSpaceConfig, init_search_space

    cfg = SearchSpaceConfig(
        compounds_csv="data/compounds.csv",
        out_df_csv="data/processed.csv",
        out_X_csv="data/featurized_processed.csv",
        out_y_csv="data/labels.csv",
        max_components=3,
        allowed_ids=None,  # defaults to all compound ids in compounds.csv
        include_ternary=True,
    )
    X, y, df = init_search_space(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import itertools
import ast

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None  # type: ignore

try:
    import mordred
    import mordred.descriptors
except Exception:
    mordred = None  # type: ignore


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SearchSpaceConfig:
    # Inputs
    compounds_csv: PathLike

    # Optional hot-start (seed dataset)
    processed_seed_csv: Optional[PathLike] = None  # e.g., processed_v2.csv

    # Optional: precomputed component features
    cmp_features_csv: Optional[PathLike] = None

    # Outputs (optional)
    out_df_csv: Optional[PathLike] = None
    out_X_csv: Optional[PathLike] = None
    out_y_csv: Optional[PathLike] = None

    # Column conventions
    compounds_id_col: str = "id"
    compounds_smiles_col: str = "smiles"

    mixture_ids_col: str = "cmp_ids"
    mixture_fracs_col: str = "cmp_mole_fractions"
    label_col: str = "value"  # BO objective (y)

    # Search space options
    allowed_ids: Optional[Sequence[int]] = None
    max_components: int = 3
    include_binary: bool = True
    include_ternary: bool = True

    # Fraction patterns
    binary_fractions: Tuple[Tuple[float, float], ...] = ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25))
    ternary_fractions: Tuple[Tuple[float, float, float], ...] = ((0.2, 0.2, 0.6), (0.1, 0.45, 0.45), (1/3, 1/3, 1/3))

    # Representation options
    normalize_cmp_features: bool = True
    fake_id: int = 200  # used for padding (removed before returning df)
    fake_smiles: str = "Fake"
    fake_name: str = "Fake"

    # Linearity filtering
    compute_linearity_flag: bool = True
    linearity_threshold: float = 0.02  # NL_score < threshold gets flagged
    compounds_value_col: str = "value"  # optional per-component reference value


def _safe_literal_eval(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        mu = out[c].mean()
        sd = out[c].std()
        if sd and not np.isclose(sd, 0):
            out[c] = (out[c] - mu) / sd
        else:
            out[c] = 0.0
    return out


def _pad_to_k(
    ids: Sequence[int],
    fracs: Sequence[float],
    k: int,
    fake_id: int,
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    ids2 = list(ids)
    fr2 = list(fracs)
    while len(ids2) < k:
        ids2.append(fake_id)
        fr2.append(0.0)
    return tuple(ids2[:k]), tuple(fr2[:k])


def _remove_fake(
    ids: Sequence[int],
    fracs: Sequence[float],
    fake_id: int,
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    kept = [(i, f) for i, f in zip(ids, fracs) if i != fake_id and not np.isclose(f, 0.0)]
    if not kept:
        return tuple(), tuple()
    ids2, fr2 = zip(*kept)
    return tuple(ids2), tuple(fr2)


def featurize_components_mordred(
    compounds_csv: PathLike,
    *,
    smiles_col: str = "smiles",
    out_csv: Optional[PathLike] = None,
    drop_na_cols: bool = True,
) -> pd.DataFrame:
    """
    Compute Mordred descriptors for components listed in compounds.csv.

    Requires RDKit + Mordred.
    Returns descriptors in the same row order as compounds.csv.
    """
    if mordred is None or Chem is None:
        raise ImportError("featurize_components_mordred requires `rdkit` and `mordred`.")

    compounds_csv = Path(compounds_csv)
    df = pd.read_csv(compounds_csv)

    calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col].astype(str).tolist()]
    feats = calc.pandas(mols)

    feats = feats.apply(pd.to_numeric, errors="coerce")
    if drop_na_cols:
        feats = feats.dropna(axis=1)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out_csv, index=False)

    return feats


def generate_unique_mixtures(
    component_ids: Sequence[int],
    fraction_patterns: Sequence[Sequence[float]],
) -> List[dict]:
    """
    Generate unique mixtures by canonicalizing (id, fraction) pairs via sorting by id.
    """
    seen = set()
    mixtures: List[dict] = []

    for pattern in fraction_patterns:
        k = len(pattern)
        for comp in itertools.combinations(component_ids, k):
            for perm_fracs in set(itertools.permutations(pattern)):
                paired_sorted = tuple(sorted(zip(comp, perm_fracs), key=lambda x: x[0]))
                if paired_sorted in seen:
                    continue
                seen.add(paired_sorted)
                ids, fracs = zip(*paired_sorted)
                mixtures.append({"cmp_ids": tuple(ids), "cmp_mole_fractions": tuple(fracs)})
    return mixtures


def build_search_space_df(cfg: SearchSpaceConfig) -> pd.DataFrame:
    """
    Build an unlabeled search space DataFrame with cmp_ids and cmp_mole_fractions.
    """
    compounds = pd.read_csv(cfg.compounds_csv)
    ids = compounds[cfg.compounds_id_col].astype(int).tolist()
    allowed = list(cfg.allowed_ids) if cfg.allowed_ids is not None else ids

    mixtures: List[dict] = []
    if cfg.include_binary:
        mixtures += generate_unique_mixtures(allowed, cfg.binary_fractions)
    if cfg.include_ternary:
        mixtures += generate_unique_mixtures(allowed, cfg.ternary_fractions)

    return pd.DataFrame(mixtures)


def featurize_mixtures(
    df: pd.DataFrame,
    *,
    cmp_features: pd.DataFrame,
    cfg: SearchSpaceConfig,
) -> pd.DataFrame:
    """
    Fixed-length mixture representation:
      [desc(id1)..., x1, desc(id2)..., x2, desc(id3)..., x3]
    """
    df2 = df.copy()

    df2[cfg.mixture_ids_col] = df2[cfg.mixture_ids_col].apply(_safe_literal_eval).apply(tuple)
    df2[cfg.mixture_fracs_col] = df2[cfg.mixture_fracs_col].apply(_safe_literal_eval).apply(tuple)

    # Add fake row to allow padding
    if cfg.fake_id not in cmp_features.index:
        fake_row = pd.Series(0.0, index=cmp_features.columns, name=cfg.fake_id)
        cmp_features = pd.concat([cmp_features, fake_row.to_frame().T])

    # Normalize component features
    if cfg.normalize_cmp_features:
        cmp_features = _zscore(cmp_features)

    # Pad to max_components
    padded = df2.apply(
        lambda r: _pad_to_k(
            r[cfg.mixture_ids_col],
            r[cfg.mixture_fracs_col],
            cfg.max_components,
            cfg.fake_id,
        ),
        axis=1,
        result_type="expand",
    )
    padded.columns = [cfg.mixture_ids_col, cfg.mixture_fracs_col]
    df2[[cfg.mixture_ids_col, cfg.mixture_fracs_col]] = padded

    def row_to_vec(r) -> List[float]:
        ids = list(r[cfg.mixture_ids_col])
        frs = list(r[cfg.mixture_fracs_col])
        vec: List[float] = []
        for cid, frac in zip(ids, frs):
            vec.extend(cmp_features.loc[int(cid)].astype(float).tolist())
            vec.append(float(frac))
        return vec

    vectors = df2.apply(row_to_vec, axis=1)

    feat_names = list(cmp_features.columns)
    cols: List[str] = []
    for k in range(1, cfg.max_components + 1):
        cols += [f"{f}_cmp{k}" for f in feat_names]
        cols += [f"mole_fraction_cmp{k}"]

    return pd.DataFrame(vectors.tolist(), columns=cols)


def attach_linearity_flag(df: pd.DataFrame, *, cfg: SearchSpaceConfig) -> pd.DataFrame:
    """
    Compute linear mixing baseline and a linearity flag using per-component values in compounds.csv.
    If `cfg.compounds_value_col` is missing in compounds.csv, this becomes a no-op.
    """
    out = df.copy()
    compounds = pd.read_csv(cfg.compounds_csv)

    if cfg.compounds_value_col not in compounds.columns:
        out["linear_value"] = np.nan
        out["NL_score"] = np.nan
        out["linearity_flag"] = False
        return out

    id_to_value = compounds.set_index(cfg.compounds_id_col)[cfg.compounds_value_col].to_dict()

    out[cfg.mixture_ids_col] = out[cfg.mixture_ids_col].apply(_safe_literal_eval).apply(tuple)
    out[cfg.mixture_fracs_col] = out[cfg.mixture_fracs_col].apply(_safe_literal_eval).apply(tuple)

    def linear_value_row(r) -> float:
        ids = r[cfg.mixture_ids_col]
        frs = r[cfg.mixture_fracs_col]
        return float(sum(float(f) * float(id_to_value.get(i, 0.0)) for i, f in zip(ids, frs)))

    out["linear_value"] = out.apply(linear_value_row, axis=1)
    out["NL_score"] = (out[cfg.label_col] - out["linear_value"]) / out["linear_value"]
    out["linearity_flag"] = out["NL_score"] < float(cfg.linearity_threshold)
    return out


def init_search_space(cfg: SearchSpaceConfig) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create (X, y, df) for Bayesian Optimization initialization.

    - Optionally merges an existing processed dataset (`processed_seed_csv`) for hot-start.
    - Builds an unlabeled search space from compounds + fraction patterns.
    - Loads component features from `cmp_features_csv` (preferred) or computes them with Mordred.
    - Featurizes mixtures into a fixed-length vector X and extracts y from `label_col`.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    df : pd.DataFrame
    """
    # Seed (hot start)
    if cfg.processed_seed_csv is not None:
        seed = pd.read_csv(cfg.processed_seed_csv)
        seed[cfg.mixture_ids_col] = seed[cfg.mixture_ids_col].apply(_safe_literal_eval).apply(tuple)
        seed[cfg.mixture_fracs_col] = seed[cfg.mixture_fracs_col].apply(_safe_literal_eval).apply(tuple)
    else:
        seed = None

    # Generated search space
    space = build_search_space_df(cfg)

    if seed is not None:
        # Make schema-compatible with seed
        for col in seed.columns:
            if col not in space.columns:
                space[col] = None
        space = space[seed.columns]
        df_all = pd.concat([seed, space], ignore_index=True)
    else:
        df_all = space.copy()
        if cfg.label_col not in df_all.columns:
            df_all[cfg.label_col] = None

    # Component features
    if cfg.cmp_features_csv is not None:
        cmp_feat = pd.read_csv(cfg.cmp_features_csv)
        if cfg.compounds_id_col in cmp_feat.columns:
            cmp_feat = cmp_feat.set_index(cfg.compounds_id_col)
        else:
            compounds = pd.read_csv(cfg.compounds_csv)
            cmp_feat.index = compounds[cfg.compounds_id_col].astype(int).tolist()
    else:
        feats = featurize_components_mordred(cfg.compounds_csv, smiles_col=cfg.compounds_smiles_col)
        compounds = pd.read_csv(cfg.compounds_csv)
        feats.index = compounds[cfg.compounds_id_col].astype(int).tolist()
        cmp_feat = feats

    # Featurize mixtures and labels
    X = featurize_mixtures(df_all, cmp_features=cmp_feat, cfg=cfg)
    y = df_all[cfg.label_col]

    # Optional linearity flag
    if cfg.compute_linearity_flag and cfg.label_col in df_all.columns:
        df_all = attach_linearity_flag(df_all, cfg=cfg)

    # Remove fake padding from returned df
    df_all[cfg.mixture_ids_col] = df_all[cfg.mixture_ids_col].apply(_safe_literal_eval).apply(tuple)
    df_all[cfg.mixture_fracs_col] = df_all[cfg.mixture_fracs_col].apply(_safe_literal_eval).apply(tuple)
    df_all[[cfg.mixture_ids_col, cfg.mixture_fracs_col]] = df_all.apply(
        lambda r: pd.Series(_remove_fake(r[cfg.mixture_ids_col], r[cfg.mixture_fracs_col], cfg.fake_id)),
        axis=1,
    )

    # Optional saves
    if cfg.out_df_csv is not None:
        p = Path(cfg.out_df_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(p, index=False)

    if cfg.out_X_csv is not None:
        p = Path(cfg.out_X_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(p, index=False)

    if cfg.out_y_csv is not None:
        p = Path(cfg.out_y_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        y.to_csv(p, index=True)

    return X, y, df_all
