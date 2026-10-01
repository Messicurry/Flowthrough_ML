"""R1-R6 under shuffled 5-fold CV repeated with seeds 0-4 (Table 4 of the paper).

    python run_ladder.py

Takes about two minutes. Writes results/ladder_folds.csv (150 fold scores) and
results/ladder.csv (the summary table).
"""
import time
from pathlib import Path

import pandas as pd

from flowthrough.data import STUDY, load
from flowthrough.evaluate import cross_validate, summarize
from flowthrough.models import ROUTES

OUT = Path(__file__).resolve().parent / "results"


def main():
    data, dictionary = load()
    print(f"{len(data)} records, {data[STUDY].nunique()} studies, "
          f"{data['pollutant_name_std'].nunique()} pollutants\n")

    folds = []
    for route in ROUTES:
        start = time.time()
        folds.append(cross_validate(data, dictionary, route))
        print(f"  {route}  {time.time() - start:4.0f} s")
    folds = pd.concat(folds, ignore_index=True)
    table = summarize(folds)

    print(f"\n{'route':<7}{'R2':<16}{'MAE (pp)':<16}{'RMSE (pp)'}")
    for route, row in table.iterrows():
        cells = [f"{row[(m, 'mean')]:.{d}f} ± {row[(m, 'std')]:.{d}f}"
                 for m, d in (("R2", 3), ("MAE", 2), ("RMSE", 2))]
        print(f"{route:<7}{cells[0]:<16}{cells[1]:<16}{cells[2]}")

    OUT.mkdir(exist_ok=True)
    folds.to_csv(OUT / "ladder_folds.csv", index=False)
    table.to_csv(OUT / "ladder.csv")


if __name__ == "__main__":
    main()
