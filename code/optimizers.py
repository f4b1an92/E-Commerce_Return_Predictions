import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

repo_path = os.path.abspath("")
data_path = repo_path + "\\data"
code_path = repo_path + "\\code"

sys.path.extend([
    code_path
])
from utils import *


def optimizer_xgb(space: dict, x: pd.DataFrame, y: pd.Series, feature_list: list, woe: bool = True, n_splits: int = 5, random_state: int = 2022, woe_correction: int = 0.01) -> int:
    np.random.seed(random_state)
    model = xgb.XGBClassifier(
        learning_rate=space['learning_rate']
        , gamma=space['gamma']
        , max_depth=int(space['max_depth'])
        , subsample=space['subsample']
        , n_estimators=int(space['n_estimators'])
        , reg_lambda=space['reg_lambda']
        , reg_alpha=space['reg_alpha']
        , min_child_weight=space['min_child_weight']
        , random_state=random_state
        , use_label_encoder=False
        , eval_metric='auc'
        , objective='binary:logistic'
        , n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    roc_list = []

    for j, idx in enumerate(skf.split(X=x, y=y)):
        X_tr, X_val = x.iloc[idx[0], :], x.iloc[idx[1], :]
        y_tr, y_val = y.iloc[idx[0]], y.iloc[idx[1]]

        if woe:
            # Compute WOE-features for every fold from scratch to minimize leakage
            X_tr = compute_woe(data=pd.concat([X_tr, y_tr], axis=1), feature_list=feature_list, correction=woe_correction)
            X_tr = X_tr.drop(labels='return', axis=1)
            for col in feature_list:
                # Part 2: Join WOE-scores computed on the current fold of the train set to the validation set
                X_val = X_val.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
                X_val[f'woe_{col}'] = X_val[f'woe_{col}'].fillna(0)
            X_tr.drop(labels=feature_list, axis=1, inplace=True)
            X_val.drop(labels=feature_list, axis=1, inplace=True)
        else:
            pass

        model.fit(X_tr, y_tr)
        y_hat = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_hat)
        roc_list.append(score)

    return -1 * np.mean(roc_list)


def optimizer_rf(space: dict, x: pd.DataFrame, y: pd.Series, feature_list: list, n_splits: int = 5
                 , random_state: int = 2022, crit: str = 'entropy', woe: bool = True, woe_correction: int = 0.01) -> int:
    np.random.seed(random_state)
    model = RandomForestClassifier(
        n_estimators=int(space['n_estimators'])
        , max_depth=int(space['max_depth'])
        , min_samples_split=space['min_samples_split']
        , min_samples_leaf=space['min_samples_leaf']
        , max_features=space['max_features']
        , min_impurity_decrease=space['min_impurity_decrease']
        , max_samples=space['max_samples']
        , random_state=random_state
        , criterion=crit
        , n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    roc_list = []

    for j, idx in enumerate(skf.split(X=x, y=y)):
        X_tr, X_val = x.iloc[idx[0], :], x.iloc[idx[1], :]
        y_tr, y_val = y.iloc[idx[0]], y.iloc[idx[1]]

        if woe:
            # Compute WOE-features for every fold from scratch to minimize leakage
            X_tr = compute_woe(data=pd.concat([X_tr, y_tr], axis=1), feature_list=feature_list, correction=woe_correction)
            X_tr = X_tr.drop(labels='return', axis=1)
            for col in feature_list:
                # Part 2: Join WOE-scores computed on the current fold of the train set to the validation set
                X_val = X_val.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
                X_val[f'woe_{col}'] = X_val[f'woe_{col}'].fillna(0)
            X_tr.drop(labels=feature_list, axis=1, inplace=True)
            X_val.drop(labels=feature_list, axis=1, inplace=True)
        else:
            pass

        model.fit(X_tr, y_tr)
        y_hat = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_hat)
        roc_list.append(score)

    return -1 * np.mean(roc_list)

