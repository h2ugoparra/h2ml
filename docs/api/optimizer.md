# Optimizer API

## run_study

::: h2ml.optimization.optimizer.run_study

## Available metrics

| Dict | Keys |
|------|------|
| `CLF_METRICS` | `"AUC"`, `"AUC_PR"`, `"LogLoss"`, `"F1"`, `"Brier"` |
| `REG_METRICS` | `"R2"`, `"MAE"`, `"RMSE"` |

All metrics are maximised internally — error metrics (`LogLoss`, `Brier`, `MAE`, `RMSE`) are negated before optimisation and displayed with their natural (positive) values.
