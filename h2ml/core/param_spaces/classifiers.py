"""Optimization Parameters and search ranges for Classifiers

Each function returns sampled values only, plus runtime settings (n_jobs, verbosity)
that don't change predictions. study.best_params keeps only the sampled values, so a
constant that affects the model would apply during trials but not to the tuned or
final model; such settings belong in the registry's default_kwargs.
"""


def lightgbm_c_params(trial):
    """LightGBM Optuna trials"""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 10, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "n_jobs": -1,  # runtime only: speeds up trials, same predictions
    }


def xgboost_c_params(trial):
    """XGBoost Optuna trials"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),  # For imbalance
        "n_jobs": -1,  # runtime only: speeds up trials, same predictions
    }


def histgradientboosting_c_params(trial):
    """HistGradientBoosting Optuna trials"""
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 32, 255),
    }


def svc_c_params(trial):
    """SVC Optuna trials.

    poly/sigmoid and C > 10 are left out: on overlapping classes libsvm needs
    millions of iterations there (minutes per fit at ~1k rows); the registry's
    max_iter default bounds whatever slow tail remains. gamma is tuned continuously because after the
    in-fold scaling "scale" and "auto" resolve to nearly the same value.
    """
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    params = {
        "C": trial.suggest_float("C", 0.01, 10, log=True),
        "kernel": kernel,
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    if kernel == "rbf":
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    return params


def randomforest_c_params(trial):
    """Random Forest Optuna trials"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": 1,  # runtime only: same predictions
    }


def decisiontree_c_params(trial):
    """Decision Tree Optuna trials"""
    return {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }


def gradientboosting_c_params(trial):
    """Gradient Boosting Optuna trials"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def catboost_c_params(trial):
    """CatBoost Optuna trials"""
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "od_wait": trial.suggest_int("od_wait", 20, 50),
        "thread_count": -1,  # runtime only: speeds up trials, same predictions
        "silent": True,  # runtime only: quiet trials
    }


def extratrees_c_params(trial):
    """Extra Trees Optuna trials"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),  # Fraction of features
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }
