# Data

`FTM18_222rows_dataset.xlsx` is an unchanged copy of the Zenodo deposit
[doi:10.5281/zenodo.21980522](https://doi.org/10.5281/zenodo.21980522) (CC BY 4.0,
md5 `2ec3b613bb12f65218711424bc68aa8f`).

It has three sheets:

- `data`: 222 single-pass records from 40 studies and 29 pollutants. Four identifier
  columns, the 50 descriptors of the physics feature set, 10 variables used only in the
  raw set, and two targets.
- `feature_dictionary`: type, unit, definition, provenance and percentage missing for
  every column, plus which feature set it belongs to and its position there.
- `route_contract`: features, target, models, hyperparameters and published scores of
  R1 to R6.

Targets:

- `removal_pct_py`: single-pass removal of the parent compound, %.
- `ln_kapp_HRT`: ln k_app with k_app = -ln(1 - removal/100) / t_HRT in 1/s, where t_HRT
  is `contact_time_s_model`. A removal of 100 % is taken as 99.99 %.

Two descriptors are worth knowing about before using them elsewhere. `pore_size_um` is
a representative value assigned from the electrode material, not a value reported in the
source studies, and the specific surface area `a_s_m2_m3_py` is derived from it. No study
reported tortuosity, so the effective diffusivity uses D0 * porosity^1.5 throughout.
