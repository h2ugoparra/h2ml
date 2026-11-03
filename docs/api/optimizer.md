# Optimizer API

## run_study

::: h2ml.optimization.optimizer.run_study

## optimize_all

::: h2ml.optimization.optimizer.optimize_all

## Available metrics

| Dict | Keys |
|------|------|
| `CLF_METRICS` | `"AUC"`, `"AUCPR"`, `"LogLoss"`, `"F1"`, `"Brier"` |
| `REG_METRICS` | `"R2"`, `"MAE"`, `"RMSE"`, `"PoissonDev"` |
| `POOLED_REG_METRICS` | the same regression keys, scored once over all out-of-fold rows |

All metrics are maximised internally — error metrics (`LogLoss`, `Brier`, `MAE`, `RMSE`, `PoissonDev`) are negated before optimisation and displayed with their natural (positive) values. Regression trials use the pooled scorers by default (`selection_scoring="pooled"`) and the fold mean with `"fold_mean"`.
