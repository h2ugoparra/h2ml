"""Optimization Parameters and search ranges for Regressors"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor




def adaboost_r_params(trial):
    """AdaBoost Optuna trials"""
    return {
        "estimator": trial.suggest_categorical("estimator", [DecisionTreeRegressor(), RandomForestRegressor()]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        "random_state": 42
    }


def bagging_r_params(trial):
    """Bagging Optuna trials"""
    return {
        "estimator": trial.suggest_categorical("estimator", [DecisionTreeRegressor(), RandomForestRegressor()]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "bootstrap_features": trial.suggest_categorical("bootstrap_features", [True, False]),
        "random_state": 42
    }


def extratrees_r_params(trial):
    """Extra Trees Optuna trials"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "random_state": 42
    }


def histgradientboosting_r_params(trial):
    """Hist Gradient Boosting Optuna trials"""
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 32),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "max_iter": trial.suggest_int("max_iter", 50, 1000),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-5, 0.1, log=True),
        "random_state": 42
    }


def decisiontree_r_params(trial):
    """Decision Tree Optuna trials"""
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "poisson"]),
        "random_state": 42
    }

def gradientboosting_r_params(trial):
    """Gradient Boosting Optuna trials"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42
    }

def svr_r_params(trial):
    """Support Vector Machine (SVM) Optuna trials"""
    # Define kernel first
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])

    # define degree but will only be used for 'poly'
    degree = trial.suggest_int("degree", 2, 5)

    params = {
        "C": trial.suggest_float("C", 0.1, 100, log=True),  # Regularization parameter
        "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),  # Epsilon-tube within which no penalty is given
        "kernel": kernel,
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "degree": degree
    }

    # Pass `degree=3` for non-poly kernels (ignored by SVR)
    if kernel != "poly":
        params["degree"] = 3  # Default value (ignored for non-poly kernels)

    return params


def catboost_r_params(trial):
    """Catboost regression Optuna trials"""
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-5, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "od_type": "Iter",
        "od_wait": trial.suggest_int("od_wait", 20, 50),
        "thread_count": -1,
        "loss_function": "RMSE",
        "silent": True,
        "random_seed": 42,
    }


def lightgbm_r_params(trial):
    """LightGBM Regression Optuna trials"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200, step=10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50, step=5),
        'subssample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }


def randomforest_r_params(trial):
    """Random Forest Regression Optuna trials"""
    return {
        'n_estimators': trial.suggest_int("n_estimators", 100, 500, step=50),
        'max_depth': trial.suggest_int("max_depth", 5, 20, step=1),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'n_jobs': -1,
        'random_state': 42
    }


def xgboost_r_params(trial):
    """XGBoost Regression Optuna trials"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 50, step=5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }
