import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle as pkl
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)

repo_path = os.path.abspath("")
data_path = repo_path + "\\data"
code_path = repo_path + "\\code"
model_path = repo_path + "\\models"
param_path = repo_path + "\\tuned_parameters"
pred_path = repo_path + "\\predictions"

sys.path.extend([
    code_path
])

import feature_gen as fg
from utils import *


# 0. Define name parameters and model_type:
model_type = 'rf'
params_name = f'tuned_parameters_{model_type}'
model_name = f'{model_type}_model_full'
preds_name = f'final_preds_{model_type}'


# 1. Prepare features
arg_dict = {
    'agg_fun': ['mean', 'median', 'max']
    , 'window': 7
    , 'feature_list': ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id']
}
fgen = fg.FeatureGen(known_filename='BADS_WS1819_known', unknown_filename='BADS_WS1819_unknown', load_path=data_path, save_path=data_path)
fgen.prep_wrapper(**arg_dict, cat_trans=None)

fl_woe = ['user_id', 'brand_id', 'item_size', 'item_color', 'item_id']
fl_to_drop = [
    'most_exp_item', 'rel_pfreq', 'title_mr', 'title_mrs', 'title_family', 'title_not_reported', 'color_keeper'
    , 'unsized_dummy', 'mean_ref_price', '7d_rolling_mean_ref_price', '7d_rolling_median_ref_price', 'max_ref_price'
    , 'user_state'
]
thr_list = [2, 1, 1, 1, 1]
correction = 0.00001

known = pd.read_parquet(data_path + '\\known_cleaned.parquet')
unknown = pd.read_parquet(data_path + '\\unknown_cleaned.parquet')
known.drop(labels=fl_to_drop, axis=1, inplace=True)
unknown.drop(labels=fl_to_drop, axis=1, inplace=True)
known.fillna(-1, inplace=True)
unknown.fillna(-1, inplace=True)

truncator(known, feature_list=fl_woe, thr=thr_list)

y_known = known['return']
known.drop(labels='return', axis=1, inplace=True)
unknown.drop(labels='return', axis=1, inplace=True)

"""
# Conventional WoE-calculation
known = compute_woe(data=pd.concat([known, y_known], axis=1), feature_list=fl_woe, correction=correction)
for col in fl_wo):
    unknown = unknown.merge(known.loc[:, [col, f"woe_{col}"]].drop_duplicates(), how='left')
    unknown[f'woe_{col}'].fillna(0, inplace=True)
    unknown.drop(labels=col, axis=1, inplace=True)
    known.drop(labels=col, axis=1, inplace=True)
known.drop(labels='return', axis=1, inplace=True)
"""

# 2. Prepare WoE-columns using single validation (https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
dict_list = []
for j, idx in enumerate(skf.split(known, y_known)):
    X_tr, X_val = known.iloc[idx[0], :], known.iloc[idx[1], :]
    y_tr, y_val = y_known.iloc[idx[0]], y_known.iloc[idx[1]]

    # Compute WOE-features for every fold from scratch to minimize leakage
    X_tr = compute_woe(data=pd.concat([X_tr, y_tr], axis=1), feature_list=fl_woe, correction=correction)
    df_dict = {}
    for col in fl_woe:
        # Saves WOE-scores per fold in order to blend them together later which is supposed to work as regularization
        df = pd.DataFrame({col: known[col].unique()})
        df = df.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
        df[f'woe_{col}'] = df[f'woe_{col}'].fillna(0)
        df_dict[col] = df
    dict_list.append(df_dict)

for col in fl_woe:
    # Blend catgorical variables' WOE-scores across different folds together for the final model training
    tmp = [dict_list[i][col].loc[:, f'woe_{col}'] for i in range(5)]
    tmp = pd.concat(tmp, axis=1).mean(axis=1) # row mean
    df = pd.DataFrame({col: known[col].unique(), f'woe_{col}': tmp.values})

    # Join blended WOE-scores of the train set folds to the test set
    unknown = unknown.merge(df, how='left')
    unknown[f'woe_{col}'].fillna(0, inplace=True)
    unknown.drop(labels=col, axis=1, inplace=True)

    # Join blended WOE-scores of the train set folds to the test set
    known = known.merge(df, how='left')
    known[f'woe_{col}'].fillna(0, inplace=True)
    known.drop(labels=col, axis=1, inplace=True)

# 3. Train model on all data points in "known" & save it
with open(param_path + f'\\{params_name}.pickle', 'rb') as handle:
    params = pkl.load(handle)

if model_type == 'xgb':
    model = xgb.XGBClassifier(
        **params
        , random_state=2022
        , use_label_encoder=False
        , eval_metric='auc'
        , objective='binary:logistic'
        , n_jobs=-1
    )
elif model_type == 'knn':
    print("knn-code not yet implemented")

elif model_type == 'rf':
    model = RandomForestClassifier(**params)

elif model_type == 'svm':
    print("svm-code not yet implemented")

elif model_type == 'lg':
    print("log_reg-code not yet implemented")

elif model_type == 'nn':
    print("nn-code not yet implemented")

else:
    raise ValueError("'model_type' must be one of the following strings: 'xgb', 'knn', 'rf', 'svm', 'lg', 'nn'")


"""
mask = known['dd_null'] == 0

model.fit(known.loc[mask, known.columns != 'order_item_id'], y_known.loc[mask])
"""
model.fit(known.loc[:, known.columns != 'order_item_id'], y_known)

with open(model_path + f"\\{model_name}.pickle", 'wb') as handle:
    pkl.dump(model, handle)


# 4. Make final predictions & save the results in a csv-file
"""
mask = unknown['dd_null'] == 0
y_hat = model.predict_proba(unknown.loc[mask, unknown.columns != 'order_item_id'])[:, 1]
df1 = pd.DataFrame({'order_item_id': unknown.loc[mask, 'order_item_id'].values, 'return': y_hat})
df2 = pd.DataFrame({'order_item_id': unknown.loc[~mask, 'order_item_id'].values, 'return': np.tile(0, 4053)})
df = pd.concat([df1, df2], axis=0).sort_values(by='order_item_id').reset_index(drop=True)
df.to_csv(pred_path + f'\\{preds_name}.csv', index=False)
"""

y_hat = model.predict_proba(unknown.loc[:, unknown.columns != 'order_item_id'])[:, 1]
df = pd.DataFrame({'order_item_id': unknown['order_item_id'].values, 'return': y_hat})
df.to_csv(pred_path + f'\\{preds_name}.csv', index=False)
