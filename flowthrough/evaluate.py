import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .data import REMOVAL, STUDY, feature_columns
from .models import ROUTES, fit_predict

SEEDS = (0, 1, 2, 3, 4)


def scores(y, pred):
    return {
        "R2": r2_score(y, pred),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
    }


def cross_validate(data, dictionary, route, seeds=SEEDS, n_splits=5):
    """Fold scores of one route under shuffled 5-fold CV, repeated for every seed."""
    numeric, categorical = feature_columns(dictionary, ROUTES[route][0])
    y = data[REMOVAL].to_numpy(float)
    rows = []
    for seed in seeds:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(folds.split(data)):
            pred = fit_predict(route, data, numeric, categorical, train, test, seed)
            rows.append({"route": route, "seed": seed, "fold": fold, **scores(y[test], pred)})
    return pd.DataFrame(rows)


def summarize(folds):
    """Average the five folds of each seed, then mean and SD over the seeds."""
    per_seed = folds.groupby(["route", "seed"])[["R2", "MAE", "RMSE"]].mean()
    return per_seed.groupby("route").agg(["mean", "std"])


def leave_one_study_out(data, dictionary, route, seed=0):
    """Predict every record with a model that has not seen any record of its study."""
    numeric, categorical = feature_columns(dictionary, ROUTES[route][0])
    studies = data[STUDY].to_numpy()
    pred = np.full(len(data), np.nan)
    for study in pd.unique(studies):
        test = np.flatnonzero(studies == study)
        train = np.flatnonzero(studies != study)
        pred[test] = fit_predict(route, data, numeric, categorical, train, test, seed)
    return pred


def _study_index(studies):
    return [np.asarray(idx) for idx in pd.Series(np.arange(len(studies))).groupby(studies).groups.values()]


def offset_share(y, pred, studies):
    """Share of the squared error that a constant offset per study accounts for."""
    err = pd.Series(y - pred)
    offset = err.groupby(np.asarray(studies)).transform("mean")
    return float(np.sum(offset ** 2) / np.sum(err ** 2))


def one_point_calibration(y, pred, studies):
    """Correct each study with the error of a single measured record.

    Every record takes a turn as the measured one and the shift is applied to the
    other records of its study. Studies with one record are skipped. Returns the
    observed and corrected values of the records that were not measured.
    """
    observed, corrected = [], []
    for idx in _study_index(studies):
        if len(idx) < 2:
            continue
        err = y[idx] - pred[idx]
        for i in range(len(idx)):
            rest = np.delete(idx, i)
            observed.append(y[rest])
            corrected.append(pred[rest] + err[i])
    return np.concatenate(observed), np.concatenate(corrected)


def within_study_rank(y, pred, studies, min_records=5):
    """Median Spearman correlation inside studies with at least `min_records` records.

    Studies where either the observations or the predictions are constant have no
    defined rank correlation and are left out.
    """
    rhos = []
    for idx in _study_index(studies):
        if len(idx) >= min_records and np.std(y[idx]) > 0 and np.std(pred[idx]) > 0:
            rhos.append(spearmanr(y[idx], pred[idx]).statistic)
    return float(np.median(rhos))


def icc(values, studies):
    """One-way random-effects intraclass correlation ICC(1) for unequal group sizes."""
    values = pd.Series(np.asarray(values, float))
    groups = values.groupby(np.asarray(studies))
    n, k = len(values), groups.ngroups
    sizes = groups.size().to_numpy()
    ss_between = float(np.sum(sizes * (groups.mean().to_numpy() - values.mean()) ** 2))
    ss_within = float(np.sum((values - groups.transform("mean")) ** 2))
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)
    n0 = (n - np.sum(sizes ** 2) / n) / (k - 1)
    return (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
