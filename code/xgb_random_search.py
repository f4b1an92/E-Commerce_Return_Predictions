import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import pickle as pkl
pd.set_option('display.max_columns', None)

sys.path.extend([
    'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/1.) WS_1819/BADS2/code'
])
import feature_gen as fg
from utils import *

data_path = 'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/1.) WS_1819/BADS2/data'

# 1. Prepare features
## 1.1. Features including WOE/CE
arg_dict = {
    'agg_fun': ['mean', 'median', 'max']
    , 'window': 7
    , 'feature_list': ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
    , 'thr': 10
    , 'correction': 0.01
}
fgen = fg.FeatureGen(known_filename='BADS_WS1819_known', unknown_filename='BADS_WS1819_unknown', load_path=data_path, save_path=data_path)
fgen.prep_wrapper(**arg_dict, cat_trans='WOE') # incl. WOE-features
fgen = fg.FeatureGen(known_filename='BADS_WS1819_known', unknown_filename='BADS_WS1819_unknown', load_path=data_path, save_path=data_path)
fgen.prep_wrapper(**arg_dict, cat_trans='CE') # incl. CE-features

## 1.2. Features excluding WOE/CE
arg_dict = {
    'agg_fun': ['mean', 'median', 'max']
    , 'window': 7
    , 'feature_list': ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
    , 'thr': 10
    , 'correction': 0.01
}
fgen = fg.FeatureGen(known_filename='BADS_WS1819_known', unknown_filename='BADS_WS1819_unknown', load_path=data_path, save_path=data_path)
fgen.prep_wrapper(**arg_dict, cat_trans=None)


# 2. Reload cleaned data
known = pd.read_parquet(data_path + '/known_cleaned.parquet')
unknown = pd.read_parquet(data_path + '/unknown_cleaned.parquet')

fl = ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
truncator(data=known, feature_list=fl, thr=10)


# 3. Prepare data for the models
X_train, X_test, y_train, y_test = train_test_split(
    known.loc[:, known.columns != 'return']
    , known['return']
    , test_size=0.2
    , random_state=2021
)
# Make final selection of ref. price variable
X_train = pd.concat([
    X_train.loc[:, 'item_id':'title_not_reported']
    , X_train['mean_ref_price']
], axis=1)
X_test = pd.concat([
    X_test.loc[:, 'item_id':'title_not_reported']
    , X_test['mean_ref_price']
], axis=1)

# 4. Random Search
np.random.seed(2022)
iters = 5
woe = True
#fl = ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
fl1 = ['item_color', 'item_size', 'brand_id', 'user_state']
fl2 = ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']

param_storage = {}
roc_storage = {}
log_storage = {}

for i in range(1, iters + 1):
    print(80 * '=')
    print(f"Starting iteration {i}/{iters}:")
    params = {
        'learning_rate': np.random.uniform(0.01, 0.2),
        'gamma': np.random.uniform(0.01, 2),
        'max_depth': np.random.choice(np.arange(3, 8, 1)),
        'subsample': np.random.uniform(0.5, 0.9),
        'n_estimators': np.random.choice(np.arange(200, 600, 25)),
        'random_state': 2022,
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'objective': 'binary:logistic'
    }
    if i != 5:
        continue
    param_storage[f'iter_{i}'] = params
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    log_list_train = []
    log_list_val = []
    roc_list_train = []
    roc_list_val = []
    for k, idx in enumerate(skf.split(X=X_train, y=y_train)):
        X_tr, X_val = X_train.iloc[idx[0], :], X_train.iloc[idx[1], :]
        y_tr, y_val = y_train.iloc[idx[0]], y_train.iloc[idx[1]]
        if woe:
            # Compute WOE-features for every fold from scratch to minimize leakage
            X_tr = compute_woe(x=X_tr, y=y_tr, feature_list=fl, correction=0.01)
            for col in fl:
                X_val = X_val.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
                X_val[f'woe_{col}'] = X_val[f'woe_{col}'].fillna(0)
            X_tr.drop(labels=fl, axis=1, inplace=True)
            X_val.drop(labels=fl, axis=1, inplace=True)

        else:
            # Compute CE-features for every fold from scratch to minimize leakage
            X_tr = compute_ce(x=X_tr, y=y_tr, feature_list=fl)
            X_tr.drop(labels=fl, axis=1, inplace=True)
            X_val = compute_ce(x=X_val, y=y_val, feature_list=fl)
            X_val.drop(labels=fl, axis=1, inplace=True)

        # Fit the model
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        y_hat_tr = model.predict_proba(X_tr)[:, 1]
        y_hat_val = model.predict_proba(X_val)[:, 1]
        roc_list_train.append(roc_auc_score(y_tr, y_hat_tr))
        roc_list_val.append(roc_auc_score(y_val, y_hat_val))
        log_list_train.append(log_loss(y_tr, y_hat_tr))
        log_list_val.append(log_loss(y_val, y_hat_val))
        print(f"\t\t\t Fold {k+1} | ROC-AUC: {roc_list_train[-1]:.4f} / {roc_list_val[-1]:.4f} | Log-Loss: {log_list_train[-1]:.4f} / {log_list_val[-1]:.4f}")
    roc_storage[f'iter_{i+1}'] = np.mean(roc_list_val)
    log_storage[f'iter_{i+1}'] = np.mean(log_list_val)
    print(f"\n\t Mean Val. ROC-AUC: {np.mean(roc_list_val):.4f} | Mean Val. Log-Loss: {np.mean(log_list_val):.4f} \n")


# 5. Fitting final model
np.random.seed(2022)
fl1 = ['item_color', 'item_size', 'brand_id', 'user_state']
fl2 = ['item_id', 'user_id', 'item_color', 'item_size', 'brand_id', 'user_state']
X_train = compute_woe(x=X_train, y=y_train, feature_list=fl, correction=0.01)
for col in fl:
    X_test = X_test.merge(X_train.loc[:,[col,f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True),how='left')
    X_test[f'woe_{col}'] = X_test[f'woe_{col}'].fillna(0)
X_train.drop(labels=fl, axis=1, inplace=True)
X_test.drop(labels=fl, axis=1, inplace=True)

idx_best = np.where([val for val in roc_storage.values()] == max([val for val in roc_storage.values()]))

#gbm = xgb.XGBClassifier(**param_storage[f'iter_{idx_best[0][0]+1}'])
gbm = xgb.XGBClassifier(**param_storage[f'iter_{5}'])
gbm.fit(X_train, y_train)
y_hat = gbm.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score on test set: {roc_auc_score(y_test, y_hat):.4f}")

with open(data_path + '/gbm_model_rs.pickle', 'wb') as handle:
    pkl.dump(gbm, handle)


# 6. Check Variable importance
with open(data_path + '/gbm_model_rs.pickle', 'rb') as handle:
    gbm = pkl.load(handle)

wdict = gbm.get_booster().get_score(importance_type='weight')

df = pd.DataFrame({'var': [key for key in wdict.keys()], 'weight': [val for val in wdict.values()]})
df.sort_values(by='weight', ascending=False, inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

fig,ax = plt.subplots(1,1,figsize=(12,8))

sns.barplot(data=df,y='var',x='weight',ax=ax)
#ax.set_xticklabels(labels=df['var'], rotation=45,ha='right')
plt.show()
