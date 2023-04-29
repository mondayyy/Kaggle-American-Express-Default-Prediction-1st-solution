import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc, os, random
import time, datetime
import shap
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from utils import *
from model import *
from A1_models import *

root = args.root
seed = args.seed

df = pd.read_feather(f"{root}/all_feature.feather")

train_y = pd.read_csv(f"{root}/train_labels.csv")
train = df[: train_y.shape[0]]
train["target"] = train_y["target"]
test = df[train_y.shape[0] :].reset_index(drop=True)
del df

print(train.shape, test.shape)

lgb_config = {
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "dart",
        "max_depth": -1,
        "num_leaves": 64,
        "learning_rate": 0.035,
        "bagging_freq": 5,
        "bagging_fraction": 0.75,
        "feature_fraction": 0.05,
        "min_data_in_leaf": 256,
        "max_bin": 63,
        "min_data_in_bin": 256,
        # 'min_sum_heassian_in_leaf': 10,
        "tree_learner": "serial",
        "boost_from_average": "false",
        "lambda_l1": 0.1,
        "lambda_l2": 30,
        "num_threads": 24,
        "verbosity": 1,
    },
    "feature_name": [],
    "rounds": 4500,
    "early_stopping_rounds": 100,
    "verbose_eval": 50,
    "folds": 5,
    "seed": seed,
}

lgb_config = {
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "dart",
        "max_depth": -1,
        "num_leaves": 64,
        "learning_rate": 0.035,
        "bagging_freq": 5,
        "bagging_fraction": 0.75,
        "feature_fraction": 0.05,
        "min_data_in_leaf": 256,
        "max_bin": 63,
        "min_data_in_bin": 256,
        # 'min_sum_heassian_in_leaf': 10,
        "tree_learner": "serial",
        "boost_from_average": "false",
        "lambda_l1": 0.1,
        "lambda_l2": 30,
        "num_threads": 24,
        "verbosity": 1,
    },
    "feature_name": [
        col
        for col in train.columns
        if col not in [id_name, label_name, "S_2"]
        and "skew" not in col
        and "kurt" not in col
        and "sub_mean" not in col
        and "div_mean" not in col
    ],
    "rounds": 4500,
    "early_stopping_rounds": 100,
    "verbose_eval": 50,
    "folds": 5,
    "seed": seed,
}

lgb_config["feature_name"] = [
    col
    for col in train.columns
    if col not in [id_name, label_name, "S_2"] and "target" not in col
]

model = Lgb_model(
    train, test, lgb_config, aug=None, run_id="LGB_with_manual_feature"
)
explainer = shap.LinearExplainer(
        model, train, feature_perturbation="interventional"
)
shap_values = explainer(test)
shap.plots.waterfall(shap_values[0])

lgb_config["feature_name"] = [
    col for col in train.columns if col not in [id_name, label_name, "S_2"]
]

model = Lgb_model(
    train, test, lgb_config, aug=None, run_id="LGB_with_manual_feature_and_series_oof"
)
explainer = shap.LinearExplainer(
  model, train, feature_perturbation="interventional"
)
shap_values = explainer(test)
shap.plots.waterfall(shap_values[0])

