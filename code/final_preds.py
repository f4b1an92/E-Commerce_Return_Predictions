import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle as pkl
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', None)

sys.path.extend([
    'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/1.) WS_1819/BADS2/code'
])
#import feature_gen as fg
from utils import *

data_path = 'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/1.) WS_1819/BADS2/data'
model_name = 'xgb_model' #'gbm_model_rs'

# 1. Load cleaned data & the model
known = pd.read_parquet(data_path + '/known_cleaned.parquet')
unknown = pd.read_parquet(data_path + '/unknown_cleaned.parquet')

fl = ['user_id','brand_id','item_size','item_color','item_id'] #['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
truncator(data=known, feature_list=fl, thr=[6,10,10,10,10]) #,100


# 2. Prepare data...
known.drop(labels=['orderday', 'clt', 'rel_pfreq'], axis=1, inplace=True)
known = pd.concat([
    known.loc[:, 'order_item_id':'color_returner']
    , known['max_ref_price']
], axis=1)
y_known = known['return']
known.drop(labels='return', axis=1, inplace=True)

unknown.drop(labels=['orderday', 'clt', 'rel_pfreq'], axis=1, inplace=True)
unknown = pd.concat([
    unknown.loc[:, 'order_item_id':'color_returner']
    , unknown['max_ref_price']
], axis=1)
unknown.drop(labels='return', axis=1, inplace=True)

# Implementation of single validation for WOE (cf. https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)
#fl = ['brand_id']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
dict_list = []
for j, idx in enumerate(skf.split(known, y_known)):
    X_tr, X_val = known.iloc[idx[0], :], known.iloc[idx[1], :]
    y_tr, y_val = y_known.iloc[idx[0]], y_known.iloc[idx[1]]

    # Compute WOE-features for every fold from scratch to minimize leakage
    X_tr = compute_woe(x=X_tr, y=y_tr, feature_list=fl, correction=0.01)
    df_dict = {}
    for col in fl:
        # Saves WOE-scores per fold in order to blend them together later which is supposed to work as regularization
        df = pd.DataFrame({col: known[col].unique()})
        df = df.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
        df[f'woe_{col}'] = df[f'woe_{col}'].fillna(0)
        df_dict[col] = df
    dict_list.append(df_dict)

# Blend catgorical variables' WOE-scores across different folds together for the final model training
for col in fl:
    tmp = [dict_list[i][col].loc[:, f'woe_{col}'] for i in range(5)]
    tmp = pd.concat(tmp, axis=1).mean(axis=1) # row mean
    df = pd.DataFrame({col: known[col].unique(), f'woe_{col}': tmp.values})

    # Join blended WOE-scores of the train set folds to the test set
    unknown = unknown.merge(df, how='left')
    unknown[f'woe_{col}'].fillna(0, inplace=True)
    #unknown.drop(labels=col, axis=1, inplace=True)

    # Join blended WOE-scores of the train set folds to the test set
    known = known.merge(df, how='left')
    known[f'woe_{col}'].fillna(0, inplace=True)
    #unknown.drop(labels=col, axis=1, inplace=True)


"""
# ... and also join WOE's to unknown-df
known = compute_woe(x=known, y=y_known, feature_list=fl, correction=0.01)

for col in fl: # ['item_color', 'item_size', 'brand_id', 'user_state']:
    df = known.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True)
    unknown = unknown.merge(df, how='left')
    unknown[f'woe_{col}'] = unknown[f'woe_{col}'].fillna(0)
"""

fl = ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
unknown.drop(labels=fl, axis=1, inplace=True)

# 3. Make final predictions & save the results in a csv-file
y_hat = gbm.predict_proba(unknown.loc[:, 'item_price':])[:, 1]
df = pd.DataFrame({'order_item_id': unknown['order_item_id'].values, 'return': y_hat})
df.to_csv(data_path + '/final_preds.csv', index=False)

tmp = gbm.get_booster().get_score(importance_type='gain')
df = pd.DataFrame({'feature':[key for key in tmp.keys()],'gain':[val for val in tmp.values()]})
df = df.sort_values(by='gain',ascending=False).reset_index(drop=True)

import matplotlib.pyplot as plt
import seaborn as sns

fig,ax = plt.subplots(1,1,figsize=(12,8))

sns.barplot(data=df,x='gain',y='feature')
plt.show()

gbm


