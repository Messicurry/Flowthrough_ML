# Flowthrough_ML

Code for

> Y. He, Y. Guo, Z. Wang, L. Wang. Physics-informed grey-box prediction of single-pass
> pollutant removal in electrochemical flow-through reactors. *Desalination*, 2026, 120834.
> https://doi.org/10.1016/j.desal.2026.120834

It reproduces, from the public dataset, the comparison of the six modelling routes
(Table 4) and the leave-one-study-out analysis with one-point calibration
(Section 3.7, Table S17).

## The idea

Single-pass removal X in a flow-through reactor is bounded between 0 and 100 %, and in the
literature the hydraulic residence time t_HRT ranges from below 0.1 s to about 10 min.
Instead of regressing X directly, the grey-box routes predict the apparent first-order
rate constant

    k_app = -ln(1 - X/100) / t_HRT

and turn it back into removal with

    X = 100 * (1 - exp(-k_app * t_HRT))

so residence time acts through the kinetics rather than as one more input. R5 and R6
then add a second model that learns the error left after this decoding.

| Route | Features | Target | Model |
|---|---|---|---|
| R1 | raw (36 numeric + 10 categorical) | removal | random forest |
| R2 | physics (40 + 10) | removal | random forest |
| R3 | raw | ln k_app, decoded | gradient boosting (HGB) |
| R4 | physics | ln k_app, decoded | HGB |
| R5 | physics | ln k_app, decoded, plus residual model | HGB + HGB |
| R6 | physics | ln k_app, decoded, plus residual model | HGB + CatBoost |

R1 to R2 and R3 to R4 isolate the transport descriptors of the physics set; R5 to R6
changes only the residual learner. Hyperparameters are in `flowthrough/models.py` and in
the `route_contract` sheet of the dataset.

## Data

`data/FTM18_222rows_dataset.xlsx` is the Zenodo deposit
([doi:10.5281/zenodo.21980522](https://doi.org/10.5281/zenodo.21980522), CC BY 4.0):
222 single-pass records from 40 studies covering 29 pollutants, with a feature dictionary
and the route definitions. See [data/README.md](data/README.md).

## Running

Tested with Python 3.12.

```
pip install -r requirements.txt
python run_ladder.py      # Table 4, about 1 min
python run_loso.py        # leave-one-study-out, about 2 min
pytest                    # checks the published numbers, about 1.5 min
```

`run_ladder.py` (mean ± SD over five fold seeds):

```
route  R2              MAE (pp)        RMSE (pp)
R1     0.766 ± 0.019   9.76 ± 0.26     15.28 ± 0.38
R2     0.767 ± 0.034   9.13 ± 0.27     15.15 ± 0.51
R3     0.774 ± 0.024   8.25 ± 0.54     14.83 ± 1.22
R4     0.783 ± 0.016   8.20 ± 0.54     14.49 ± 1.00
R5     0.804 ± 0.015   7.44 ± 0.26     13.82 ± 0.78
R6     0.818 ± 0.019   7.33 ± 0.29     13.38 ± 0.87
```

`run_loso.py`:

```
intraclass correlation of ln k_app across studies: 0.914

          R2    MAE  offset share  R2 after 1-point  MAE after 1-point  median rho
route
R1    -0.849 32.044         0.759            -0.831             27.771       0.411
R4    -0.243 29.450         0.812             0.323             20.950      -0.252
R6    -0.104 26.713         0.822             0.474             17.806       0.100
```

When a whole study is held out, the pooled R² is negative for all three routes. For R6,
82 % of the squared error is a constant offset per study. Correcting that offset with a
single measured record from the new study brings R² to 0.474, while the same correction
does almost nothing for the black-box R1. The low median rank correlation inside studies
means the level of an unseen study can be calibrated, but the ordering of its conditions
is not reliably predicted.

These values match the paper exactly with the versions pinned in `requirements.txt`.
Other releases of scikit-learn or CatBoost can shift the last digit.

## Notes

- Evaluation is shuffled 5-fold cross-validation repeated with seeds 0 to 4. All 25
  folds are kept; the SD is taken over the five seed means.
- Preprocessing (median or most-frequent imputation, scaling, one-hot encoding) is
  fitted inside each training fold.
- In R5 and R6 the residual target is the observed removal minus the decoded base
  prediction on the same training fold, and the base prediction is also given to the
  residual model as an input. The final prediction is clipped to 0–100 %.
- Studies are grouped by `paper_title_full`. `ref_short` is not unique per study.
- Column order is kept as recorded in the dataset: trees break ties between equally good
  splits by column position, so reordering changes the scores slightly.
- Leave-one-study-out uses seed 0.

## Layout

```
flowthrough/data.py       dataset loading and feature sets
flowthrough/models.py     rate decoder and the six routes
flowthrough/evaluate.py   cross-validation, leave-one-study-out, calibration
run_ladder.py             Table 4
run_loso.py               Section 3.7
tests/
```

## Citation and licence

Please cite the paper above if you use the code or the data (`CITATION.cff`).
Code: MIT. Data: CC BY 4.0.
