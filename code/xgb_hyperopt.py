import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
from hyperopt import Trials, fmin, hp, tpe
from functools import partial
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

fl1 = ['user_id', 'brand_id','item_size','item_color','item_id'] #['item_size', 'brand_id', 'user_id', 'item_color', 'item_id', 'user_state']
fl2 = ['user_state'] # user_id, 'brand_id','item_color','item_size'
truncator(known, feature_list=fl1, thr=[6,10,10,10])
known.drop(labels=fl2, axis=1, inplace=True)
known.drop(labels=['orderday','clt','rel_pfreq'], axis=1, inplace=True)

# 3. Prepare data for the models
X_train, X_test, y_train, y_test = train_test_split(
    known.loc[:, known.columns != 'return']
    , known['return']
    , test_size=0.2
    , random_state=2021
)

# Make final selection of ref. price variable
X_train = pd.concat([
    #X_train.loc[:, 'item_size':'title_not_reported']
    X_train.loc[:, 'item_id':'color_returner']
    , X_train['max_ref_price']
], axis=1)
X_test = pd.concat([
    #X_test.loc[:, 'item_size':'title_not_reported']
    X_test.loc[:, 'item_id':'color_returner']
    , X_test['max_ref_price']
], axis=1)


def optimizer_xgb(space, x, y, feature_list, woe=True, n_splits=5, random_state=2022):
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
            X_tr = compute_woe(x=X_tr, y=y_tr, feature_list=feature_list, correction=0.01)
            for col in feature_list:
                # Part 2: Join WOE-scores computed on the current fold of the train set to the validation set
                X_val = X_val.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
                X_val[f'woe_{col}'] = X_val[f'woe_{col}'].fillna(0)
            X_tr.drop(labels=feature_list, axis=1, inplace=True)
            X_val.drop(labels=feature_list, axis=1, inplace=True)
        else:
            pass
            """
            # Compute CE-features for every fold from scratch to minimize leakage
            X_tr = compute_ce(x=X_tr, y=y_tr, feature_list=feature_list)
            X_tr.drop(labels=feature_list, axis=1, inplace=True)
            X_val = compute_ce(x=X_val, y=y_val, feature_list=feature_list)
            X_val.drop(labels=feature_list, axis=1, inplace=True)
            """

        model.fit(X_tr, y_tr)
        y_hat = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_hat)
        roc_list.append(score)
    return -1 * np.mean(roc_list)


params = {
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01)
    , 'colsample_bylevel': hp.uniform('colsample_bylevel', 0, 1)
    , 'colsample_bynode': hp.uniform('colsample_bynode', 0, 1)
    , 'gamma': hp.uniform('gamma', 0, 3)
    , 'max_depth': hp.quniform('max_depth', 2, 8, 1)
    , 'subsample': hp.uniform('subsample', 0.5, 1)
    , 'n_estimators': hp.quniform('n_estimators', 250, 800, 25)
    , 'reg_lambda': hp.uniform('reg_lambda', 0, 1)
    , 'reg_alpha': hp.uniform('reg_alpha', 0, 1)
    , 'min_child_weight': hp.uniform('min_child_weight', 1, 5)
}

fmin_objective = partial(optimizer_xgb, x=X_train, y=y_train, feature_list=fl1, woe=True, random_state=2022, n_splits=5)

trials = Trials()

best_hyperparams = fmin(
    fn=fmin_objective
    , space=params
    , algo=tpe.suggest
    , max_evals=3
    , trials=trials
    , rstate=np.random.default_rng(42)
)

# Extract best hyperparameters and train model on full train set
np.random.seed(2022)
new_param_dict = {}

for key, val in best_hyperparams.items():
    if key in ['n_estimators', 'max_depth']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val


# Implementation of single validation for WOE (cf.https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)
fl = ['user_id','brand_id','item_color','item_size','item_id']
#fl = ['brand_id','item_size','user_id', 'item_color', 'item_id', 'user_state']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
dict_list = []
for j, idx in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[idx[0], :], X_train.iloc[idx[1], :]
    y_tr, y_val = y_train.iloc[idx[0]], y_train.iloc[idx[1]]

    # Compute WOE-features for every fold from scratch to minimize leakage
    X_tr = compute_woe(x=X_tr, y=y_tr, feature_list=fl, correction=0.01)
    df_dict = {}
    for col in fl:
        # Saves WOE-scores per fold in order to blend them together later which is supposed to work as regularization
        df = pd.DataFrame({col: X_train[col].unique()})
        df = df.merge(X_tr.loc[:, [col, f'woe_{col}']].drop_duplicates(subset=col).reset_index(drop=True), how='left')
        df[f'woe_{col}'] = df[f'woe_{col}'].fillna(0)
        df_dict[col] = df
    dict_list.append(df_dict)

# Blend catgorical variables' WOE-scores across different folds together for the final model training
for col in fl:
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


gbm = xgb.XGBClassifier(
    **new_param_dict
    , random_state=2022
    , use_label_encoder=False
    , eval_metric='auc'
    , objective='binary:logistic'
    , n_jobs=-1
)
gbm.fit(X_train, y_train)
y_hat_train = gbm.predict_proba(X_train)[:, 1]
y_hat_test = gbm.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score on the Test set: {roc_auc_score(y_train, y_hat_train):.4f}")
print(f"ROC-AUC Score on the Test set: {roc_auc_score(y_test, y_hat_test):.4f}")

# Save the model
with open(data_path + '/xgb_model.pickle', 'wb') as handle:
    pkl.dump(gbm, handle)


# 6. Check Variable importance
with open(data_path + '/xgb_model.pickle', 'rb') as handle:
    gbm = pkl.load(handle)

wdict = gbm.get_booster().get_score(importance_type='gain')

df = pd.DataFrame({'var': [key for key in wdict.keys()], 'gain': [val for val in wdict.values()]})
df.sort_values(by='gain', ascending=False, inplace=True)

fig,ax = plt.subplots(1,1,figsize=(12,8))

sns.barplot(data=df,y='var',x='gain',ax=ax)
#ax.set_xticklabels(labels=df['var'], rotation=45,ha='right')
plt.show()

