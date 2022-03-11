import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from hyperopt import Trials, fmin, hp, tpe
from functools import partial
import pickle as pkl

pd.set_option('display.max_columns', None)

repo_path = os.path.abspath("")
data_path = repo_path + "\\data"
code_path = repo_path + "\\code"
param_path = repo_path + "\\tuned_parameters"
model_path = repo_path + "\\models"

sys.path.extend([
    code_path
])

import feature_gen as fg
from utils import *
from optimizers import optimizer_rf

# 1. Prepare features
arg_dict = {
    'agg_fun': ['mean', 'median', 'max']
    , 'window': 7
    , 'feature_list': ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id']
}
fgen = fg.FeatureGen(known_filename='BADS_WS1819_known', unknown_filename='BADS_WS1819_unknown', load_path=data_path, save_path=data_path)
fgen.prep_wrapper(**arg_dict, cat_trans=None)

# 2. Reload cleaned data
fl_woe = ['user_id', 'brand_id', 'item_size', 'item_color', 'item_id']
fl_to_drop = [
    'most_exp_item', 'rel_pfreq', 'title_mr', 'title_mrs', 'title_family', 'title_not_reported', 'color_keeper'
    , 'unsized_dummy', 'mean_ref_price', '7d_rolling_mean_ref_price', '7d_rolling_median_ref_price', 'max_ref_price'
    , 'user_state'
]
thr_list = [2, 1, 1, 1, 1]
correction = 0.00001
random_state = 2022

known = pd.read_parquet(data_path + '\\known_cleaned.parquet')
unknown = pd.read_parquet(data_path + '\\unknown_cleaned.parquet')
known.drop(labels=fl_to_drop, axis=1, inplace=True)
unknown.drop(labels=fl_to_drop, axis=1, inplace=True)
known.fillna(-1, inplace=True)
unknown.fillna(-1, inplace=True)


# 3. Prepare data for the models
truncator(known, feature_list=fl_woe, thr=thr_list)

"""
X_train, X_test, y_train, y_test = train_test_split(
    known.loc[known['dd_null'] == 0, np.setdiff1d(known.columns, ['order_item_id', 'dd_null', 'return'])]
    , known.loc[known['dd_null'] == 0, 'return']
    , test_size=0.2
    , random_state=2021
)
"""
X_train, X_test, y_train, y_test = train_test_split(
    known.loc[:, np.setdiff1d(known.columns, ['order_item_id', 'return'])]
    , known['return']
    , test_size=0.2
    , random_state=random_state - 1
)


# 4. Hyperparameter Tuning
params = {
    'n_estimators': hp.quniform('n_estimators', 50, 800, 25)
    , 'max_depth': hp.quniform('max_depth', 2, 8, 1)
    , 'max_features': hp.uniform('max_features', 0.3, 0.9)
    , 'max_samples': hp.uniform('max_samples', 0.3, 0.9)
    , 'min_samples_split': hp.uniform('min_samples_split', 0.00001, 0.1)
    , 'min_samples_leaf': hp.uniform('min_samples_leaf', 0.00001, 0.1)
    , 'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.000001, 0.001)
}

fmin_objective = partial(
    optimizer_rf
    , x=X_train
    , y=y_train
    , feature_list=fl_woe
    , crit='entropy'
    , random_state=random_state
    , n_splits=5
    , woe=True
    , woe_correction=correction
)

trials = Trials()

best_hyperparams = fmin(
    fn=fmin_objective
    , space=params
    , algo=tpe.suggest
    , max_evals=30
    , trials=trials
    , rstate=np.random.RandomState(random_state) #np.random.default_rng(42)
)


# 5. Extract best hyperparameters and save them in a pickle-file
np.random.seed(random_state)
new_param_dict = {}

for key, val in best_hyperparams.items():
    if key in ['n_estimators', 'max_depth']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val

with open(param_path + '\\tuned_parameters_rf.pickle', 'wb') as handle:
    pkl.dump(new_param_dict, handle)


# 6. Train full model and evaluate on Test Set
# 6.1. Implementation of single validation for WOE (cf.https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
dict_list = []
for j, idx in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[idx[0], :], X_train.iloc[idx[1], :]
    y_tr, y_val = y_train.iloc[idx[0]], y_train.iloc[idx[1]]

    # Compute WOE-features for every fold from scratch to minimize leakage
    X_tr = compute_woe(pd.concat([X_tr, y_tr], axis=1), feature_list=fl_woe, correction=correction)
    X_tr = X_tr.drop(labels='return', axis=1)
    df_dict = {}
    for col in fl_woe:
        # Saves WOE-scores per fold in order to blend them together later which is supposed to work as regularization
        df = pd.DataFrame({col: X_train[col].unique()})
        df = df.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
        df[f'woe_{col}'] = df[f'woe_{col}'].fillna(0)
        df_dict[col] = df
    dict_list.append(df_dict)

for col in fl_woe:
    # Blend catgorical variables' WOE-scores across different folds together for the final model training
    tmp = [dict_list[i][col].loc[:, f'woe_{col}'] for i in range(5)]
    tmp = pd.concat(tmp, axis=1).mean(axis=1) # row mean
    df = pd.DataFrame({col: X_train[col].unique(), f'woe_{col}': tmp.values})

    # Join blended WOE-scores of the train set folds to the train set
    X_train = X_train.merge(df, how='left')
    X_train[f'woe_{col}'].fillna(0, inplace=True)
    X_train.drop(labels=col, axis=1, inplace=True)

    # Join blended WOE-scores of the train set folds to the test set
    X_test = X_test.merge(df, how='left')
    X_test[f'woe_{col}'].fillna(0, inplace=True)
    X_test.drop(labels=col, axis=1, inplace=True)

# 6.2. Train classifier with tuned hyperparameters
with open(param_path + '\\tuned_parameters_rf.pickle', 'rb') as handle:
    new_param_dict = pkl.load(handle)

rf = RandomForestClassifier(**new_param_dict)
rf.fit(X_train, y_train)

# 6.3. Make predictions using classifier with tuned hyperparameters & evaluate on test set
y_hat_train = rf.predict_proba(X_train)[:, 1]
y_hat_test = rf.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score on the Train set: {roc_auc_score(y_train, y_hat_train):.4f}")
print(f"ROC-AUC Score on the Test set: {roc_auc_score(y_test, y_hat_test):.4f}")

# 6.4. Save the model
with open(model_path + '\\rf_model.pickle', 'wb') as handle:
    pkl.dump(rf, handle)
