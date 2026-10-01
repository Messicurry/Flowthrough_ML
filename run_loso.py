"""Leave-one-study-out evaluation and one-point calibration (Section 3.7, Table S17).

    python run_loso.py

Each of the 40 studies is held out in turn. For R1, R4 and R6 the script reports
the pooled scores on the held-out studies, how much of the squared error is a
constant offset per study, what correcting that offset with one measured record
gives, and how well the order of records inside a study is reproduced.
Writes results/loso_predictions.csv.
"""
from pathlib import Path

import pandas as pd

from flowthrough.data import LN_K, REMOVAL, STUDY, load
from flowthrough.evaluate import (icc, leave_one_study_out, offset_share, one_point_calibration,
                                  scores, within_study_rank)

OUT = Path(__file__).resolve().parent / "results"
ROUTES = ("R1", "R4", "R6")


def main():
    data, dictionary = load()
    y = data[REMOVAL].to_numpy(float)
    studies = data[STUDY].to_numpy()
    print(f"intraclass correlation of ln k_app across studies: {icc(data[LN_K], studies):.3f}\n")

    rows, predictions = [], {"row_id": data["row_id"], "study": studies, "removal": y}
    for route in ROUTES:
        pred = leave_one_study_out(data, dictionary, route)
        predictions[route] = pred
        s = scores(y, pred)
        calibrated = scores(*one_point_calibration(y, pred, studies))
        rows.append({
            "route": route,
            "R2": s["R2"],
            "MAE": s["MAE"],
            "offset share": offset_share(y, pred, studies),
            "R2 after 1-point": calibrated["R2"],
            "MAE after 1-point": calibrated["MAE"],
            "median rho": within_study_rank(y, pred, studies),
        })

    table = pd.DataFrame(rows).set_index("route")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(predictions).to_csv(OUT / "loso_predictions.csv", index=False)


if __name__ == "__main__":
    main()
