import numpy as np
import pandas as pd
from typing import Union


def truncator(data: pd.DataFrame, feature_list: Union[str, list], thr: Union[int, list]):
    if isinstance(feature_list, str):
        feature_list = [feature_list]
    if isinstance(thr, int):
        thr = np.repeat(thr, len(feature_list))

    for feature, t in zip(feature_list, thr):
        count_df = data.groupby(feature).agg({
            'order_item_id': 'count'
        }).reset_index().rename(columns={
            'order_item_id': 'count'
        })

        cond = count_df['count'] < t
        vals = count_df.loc[cond, feature].values
        data.loc[data[feature].isin(vals), feature] = 'misc'


def compute_ce(x: pd.DataFrame, y: pd.Series, feature_list: list) -> pd.DataFrame:
    y_col = y.name
    data = pd.concat([x, y], axis=1)
    for feature in feature_list:
        ce_df = data.groupby(feature).agg({
            'return': 'mean'
        }).reset_index().rename(columns={
            'return': f'ce_{feature}'
        })
        ce_df[f'ce_{feature}'].fillna(0.5, inplace=True)
        data = data.merge(ce_df,how='left')
    return data.loc[:, data.columns != y_col]


def compute_woe(data: pd.DataFrame, feature_list: Union[str, list], correction: float = 0.01) -> pd.DataFrame:
    if isinstance(feature_list, str):
        feature_list = [feature_list]

    for feature in feature_list:
        woe_df = data.groupby(feature).agg({
            'order_item_id': 'count'
            , 'return': 'sum'
        }).reset_index().rename(columns={
            'order_item_id': 'total'
            , 'return': 'events'
        })
        woe_df['non_events'] = woe_df['total'] - woe_df['events']
        woe_df['numerator'] = (woe_df['non_events'] / sum(data['return'] == 0)).apply(
            lambda a: a + correction if a == 0 else a)
        woe_df['denominator'] = (woe_df['events'] / sum(data['return'] == 1)).apply(
            lambda a: a + correction if a == 0 else a)

        # WOE-calculation
        woe_df[f'woe_{feature}'] = np.log(woe_df['numerator'] / woe_df['denominator'])
        woe_df[f'woe_{feature}'] = woe_df[f'woe_{feature}'].fillna(0)
        woe_df[f'woe_{feature}'] = woe_df[f'woe_{feature}'].apply(lambda x: np.round(x, 4))
        woe_df = woe_df.loc[:, [feature, f'woe_{feature}']]
        data = data.merge(woe_df, how='left')

    return data


def compute_iv(data: pd.DataFrame, woe_vars: Union[str, list], correction: int = 0.01) -> dict:
    if isinstance(woe_vars, str):
        woe_vars = [woe_vars]

    data_with_woe = compute_woe(data=data, feature_list=[*woe_vars], correction=correction)
    result_dict = {}
    for var in woe_vars:
        tmp = data_with_woe.groupby(f'woe_{var}').agg({
            'order_item_id': 'count'
            , 'return': 'sum'
        }).reset_index()
        tmp.columns = ['woe', 'count', 'events']
        tmp['non_events'] = tmp['count'] - tmp['events']
        tmp['perc_event'] = tmp['events'] / sum(data_with_woe['return'] == 1)
        tmp['perc_non_event'] = tmp['non_events'] / sum(data_with_woe['return'] == 0)
        tmp['iv'] = (tmp['perc_non_event'] - tmp['perc_event']) * tmp['woe']
        result_dict[var] = tmp['iv'].sum()

    return result_dict


# WOE binning
def _woe_bin_prepper(data: pd.DataFrame, var: str) -> pd.DataFrame:
    tmp = data.groupby(var).agg({
        'order_item_id': 'count'
    }).reset_index().rename(columns={
        'order_item_id': 'counts'
    })
    tmp = tmp.sort_values(by=var).reset_index(drop=True)
    tmp['share'] = tmp['counts'] / data.shape[0]
    tmp['cum_share'] = tmp['share'].cumsum()
    return tmp


def _woe_q_finder(data: pd.DataFrame, var: str) -> pd.DataFrame:
    tmp = _woe_bin_prepper(data, var)
    for q in np.arange(20)[::-1]:
        tmp['qcats'] = pd.qcut(tmp['cum_share'], labels=[i for i in np.arange(q)], q=q)
        check_df = tmp.groupby('qcats').agg({
            var: 'mean'
            , 'counts': 'sum'
        }).reset_index()
        check_df['woe_cond'] = check_df['counts'] > int(data.shape[0] / 20)
        if (check_df['woe_cond'].sum() / check_df.shape[0]) != 1:
            continue
        else:
            return q, check_df


def woe_binning(data: pd.DataFrame, var: str) -> pd.DataFrame:
    q, check_df = _woe_q_finder(data, var)
    prepped_df = _woe_bin_prepper(data, var)
    prepped_df['qcats'] = pd.qcut(prepped_df['cum_share'], labels=[i for i in np.arange(q)], q=q)
    data = data.merge(prepped_df.loc[:, [var, 'qcats']], how='left').drop(labels=var, axis=1)
    data = data.merge(check_df.loc[:, ['qcats', var]], how='left')
    data.drop(labels=['qcats'], axis=1, inplace=True)
    return data
