import pandas as pd
import numpy as np
import os, random
import datetime
import pickle
from contextlib import contextmanager
from tqdm import tqdm


from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    average_precision_score,
    log_loss,
)
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler

import torch.cuda.amp as amp

from scheduler import *

import argparse

from utils import *


def Lgb_model(train, config, gkf=False, aug=None, output_root="./output/", run_id=None):
    if not run_id:
        run_id = "run_lgb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        while os.path.exists(output_root + run_id + "/"):
            time.sleep(1)
            run_id = "run_lgb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_root + f"{args.save_dir}/"
    else:
        output_path = output_root + run_id + "/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    os.system(f"cp ./*.py {output_path}")
    os.system(f"cp ./*.sh {output_path}")
    config["lgb_params"]["seed"] = config["seed"]
    oof, sub = None, None

    log = open(output_path + "/train.log", "w", buffering=1)
    log.write(str(config) + "\n")
    features = config["feature_name"]
    params = config["lgb_params"]
    rounds = config["rounds"]
    verbose = config["verbose_eval"]
    early_stopping_rounds = config["early_stopping_rounds"]
    folds = config["folds"]
    seed = config["seed"]
    oof = train[[id_name]]
    oof[label_name] = 0

    all_valid_metric, feature_importance = [], []
    if gkf:
        tmp = (
            train[[id_name, label_name]].drop_duplicates(id_name).reset_index(drop=True)
        )
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = skf.split(tmp, tmp[label_name])
        new_split = []
        for trn_index, val_index in split:
            trn_uids = tmp.loc[trn_index, id_name].values
            val_uids = tmp.loc[val_index, id_name].values
            new_split.append(
                (
                    train.loc[train[id_name].isin(trn_uids)].index,
                    train.loc[train[id_name].isin(val_uids)].index,
                )
            )
        split = new_split

        # skf = GroupKFold(n_splits=folds)
        # split = skf.split(train,train[label_name],train[id_name])
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = skf.split(train, train[label_name])
    for fold, (trn_index, val_index) in enumerate(split):
        evals_result_dic = {}
        train_cids = train.loc[trn_index, id_name].values
        if aug:
            train_aug = aug.loc[aug[id_name].isin(train_cids)]
            trn_data = lgb.Dataset(
                train.loc[trn_index, features].append(train_aug[features]),
                label=train.loc[trn_index, label_name].append(train_aug[label_name]),
            )
        else:
            trn_data = lgb.Dataset(
                train.loc[trn_index, features],
                label=train.loc[trn_index, label_name],
            )

        val_data = lgb.Dataset(
            train.loc[val_index, features], label=train.loc[val_index, label_name]
        )
        model = lgb.train(
            params,
            train_set=trn_data,
            num_boost_round=rounds,
            valid_sets=[trn_data, val_data],
            evals_result=evals_result_dic,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose,
        )
        model.save_model(output_path + "/fold%s.ckpt" % fold)

        valid_preds = model.predict(
            train.loc[val_index, features], num_iteration=model.best_iteration
        )
        oof.loc[val_index, label_name] = valid_preds

        for i in range(len(evals_result_dic["valid_1"][params["metric"]]) // verbose):
            Write_log(
                log,
                " - %i round - train_metric: %.6f - valid_metric: %.6f\n"
                % (
                    i * verbose,
                    evals_result_dic["training"][params["metric"]][i * verbose],
                    evals_result_dic["valid_1"][params["metric"]][i * verbose],
                ),
            )
        all_valid_metric.append(Metric(train.loc[val_index, label_name], valid_preds))
        Write_log(log, "- fold%s valid metric: %.6f\n" % (fold, all_valid_metric[-1]))

        importance_gain = model.feature_importance(importance_type="gain")
        importance_split = model.feature_importance(importance_type="split")
        feature_name = model.feature_name()
        feature_importance.append(
            pd.DataFrame(
                {
                    "feature_name": feature_name,
                    "importance_gain": importance_gain,
                    "importance_split": importance_split,
                }
            )
        )

    feature_importance_df = pd.concat(feature_importance)
    feature_importance_df = (
        feature_importance_df.groupby(["feature_name"]).mean().reset_index()
    )
    feature_importance_df = feature_importance_df.sort_values(
        by=["importance_gain"], ascending=False
    )
    feature_importance_df.to_csv(output_path + "/feature_importance.csv", index=False)

    mean_valid_metric = np.mean(all_valid_metric)
    global_valid_metric = Metric(train[label_name].values, oof[label_name].values)
    Write_log(
        log,
        "all valid mean metric:%.6f, global valid metric:%.6f"
        % (mean_valid_metric, global_valid_metric),
    )

    oof.to_csv(output_path + "/oof.csv", index=False)

    log.close()
    os.rename(
        output_path + "/train.log",
        output_path + "/train_%.6f.log" % mean_valid_metric,
    )

    log_df = pd.DataFrame(
        {
            "run_id": [run_id],
            "mean metric": [round(mean_valid_metric, 6)],
            "global metric": [round(global_valid_metric, 6)],
            "remark": [args.remark],
        }
    )
    if not os.path.exists(output_root + "/experiment_log.csv"):
        log_df.to_csv(output_root + "/experiment_log.csv", index=False)
    else:
        log_df.to_csv(
            output_root + "/experiment_log.csv", index=False, header=None, mode="a"
        )

    return lgb.Booster(model_file=output_path + "/fold%s.ckpt" % fold)
