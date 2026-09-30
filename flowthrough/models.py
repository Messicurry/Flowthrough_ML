import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import HRT, LN_K, REMOVAL

# route: (feature set, first-stage target, residual model)
ROUTES = {
    "R1": ("raw", "removal", None),
    "R2": ("physics", "removal", None),
    "R3": ("raw", "rate", None),
    "R4": ("physics", "rate", None),
    "R5": ("physics", "rate", "hgb"),
    "R6": ("physics", "rate", "catboost"),
}

BASE_COLUMN = "base_removal"


def decode(ln_k, hrt_s):
    """Removal (%) from ln k_app and residence time for first-order decay in one pass."""
    k = np.exp(np.clip(ln_k, -25, 10))
    return 100.0 * (1.0 - np.exp(-np.clip(k * hrt_s, 0, 30)))


def make_pipeline(model, numeric, categorical):
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", model)])


def random_forest(seed):
    return RandomForestRegressor(n_estimators=80, max_depth=7, min_samples_leaf=6, random_state=seed)


def rate_model(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=15,
                                         l2_regularization=0.5, random_state=seed)


def base_model(seed):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, min_samples_leaf=8,
                                         random_state=seed)


def residual_model(kind, seed):
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10,
                                             random_state=seed)
    return CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, random_state=seed,
                             verbose=False, allow_writing_files=False)


def fit_predict(route, data, numeric, categorical, train, test, seed):
    """Fit a route on the rows in `train` and return predicted removal (%) for `test`."""
    _, target, residual = ROUTES[route]
    X = data[numeric + categorical]
    removal = data[REMOVAL].to_numpy(float)
    hrt = data[HRT].to_numpy(float)

    if target == "removal":
        model = make_pipeline(random_forest(seed), numeric, categorical)
        model.fit(X.iloc[train], removal[train])
        return model.predict(X.iloc[test])

    ln_k = data[LN_K].to_numpy(float)
    if residual is None:
        model = make_pipeline(rate_model(seed), numeric, categorical)
        model.fit(X.iloc[train], ln_k[train])
        return decode(model.predict(X.iloc[test]), hrt[test])

    base = make_pipeline(base_model(seed), numeric, categorical)
    base.fit(X.iloc[train], ln_k[train])
    base_train = decode(base.predict(X.iloc[train]), hrt[train])
    base_test = decode(base.predict(X.iloc[test]), hrt[test])

    # Second stage: learn what the decoded base prediction misses, in percentage
    # points, with the base prediction as one more input.
    X_train = X.iloc[train].assign(**{BASE_COLUMN: base_train})
    X_test = X.iloc[test].assign(**{BASE_COLUMN: base_test})
    closure = make_pipeline(residual_model(residual, seed), numeric + [BASE_COLUMN], categorical)
    closure.fit(X_train, removal[train] - base_train)
    return np.clip(base_test + closure.predict(X_test), 0.0, 100.0)
