from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "FTM18_222rows_dataset.xlsx"

# A study is identified by its full title. ref_short is only a short label and is
# not unique per publication (grouping by it gives 56 groups instead of 40).
STUDY = "paper_title_full"
REMOVAL = "removal_pct_py"      # single-pass removal of the parent compound, %
LN_K = "ln_kapp_HRT"            # ln k_app, k_app = -ln(1 - removal/100) / t_HRT, in 1/s
HRT = "contact_time_s_model"    # hydraulic residence time t_HRT, s


def load(path=DATA_FILE):
    """Return the 222 records and the feature dictionary."""
    data = pd.read_excel(path, sheet_name="data")
    dictionary = pd.read_excel(path, sheet_name="feature_dictionary")
    return data, dictionary


def feature_columns(dictionary, feature_set):
    """Numeric and categorical columns of the "raw" or "physics" feature set.

    Columns come back in the order recorded in the dictionary. Tree models break
    ties between equally good splits by column position, so another order gives
    slightly different fold scores.
    """
    if feature_set not in ("raw", "physics"):
        raise ValueError(f"unknown feature set: {feature_set}")
    rows = dictionary[dictionary[f"in_{feature_set}_set"]].sort_values(f"{feature_set}_set_order")
    numeric = rows.loc[rows["type"] == "numeric", "column"].tolist()
    categorical = rows.loc[rows["type"] == "categorical", "column"].tolist()
    return numeric, categorical
