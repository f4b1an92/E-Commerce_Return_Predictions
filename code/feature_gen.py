import time
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import datetime
from typing import Union


class FeatureGen(object):
    def __init__(self, known_filename: str, unknown_filename: str, load_path: str, save_path: str = None):
        # Load and clean known data set
        self.known = pd.read_csv(load_path + f'/{known_filename}.csv')
        self.known['known'] = 1
        self.known['brand_id'] = self.known['brand_id'] - 100
        self.known['item_id'] = self.known['item_id'].apply(lambda x: int((x/2) - 1))
        # Load and clean unknown data set
        self.unknown = pd.read_csv(load_path + f'/{unknown_filename}.csv')
        self.unknown['known'] = 0
        self.unknown['return'] = np.nan
        # Fuse known and unknown to facilitate feature creation
        self.data = pd.concat([self.known, self.unknown], axis=0).reset_index(drop=True)
        cond_dob = pd.to_datetime(self.data['user_dob']).dt.year <= 1901
        self.data.loc[cond_dob, 'user_dob'] = np.nan
        self.save_path = save_path

    def _delivery_null_feature(self):
        self.data['dd_null'] = self.data['delivery_date'].apply(lambda x: 1 if pd.isna(x) else 0)

    def _delivery_1994_feature(self):
        self.data['dd_1994'] = pd.to_datetime(self.data['delivery_date']).apply(lambda x: 1 if x.year == 1994 else 0)

    def _time_features(self):
        # Orderday (ranges from 0 to 365) to reflect seasonal effects
        tmp = self.data.groupby('order_date').agg({
            'order_item_id': 'count'
        }).reset_index().drop(labels='order_item_id', axis=1)
        tmp['orderday'] = tmp.index

        self.data = self.data.merge(tmp, how='left')

        # Age + median imputation of missing entries
        tmp = pd.to_datetime(self.data['order_date']) - pd.to_datetime(self.data['user_dob'])
        self.data['age'] = np.round(tmp.dt.days / 365)
        self.data['age'].fillna(self.data['age'].median(), inplace=True)

        # Customer Lifetime
        tmp = pd.to_datetime(self.data['order_date']) - pd.to_datetime(self.data['user_reg_date'])
        self.data['clt'] = tmp.dt.days + 1

        # Delivery time
        self.data['delivery_time'] = (pd.to_datetime(self.data['delivery_date']) - pd.to_datetime(self.data['order_date'])).dt.days
        median_del_time = self.data.loc[self.data['delivery_time'] >= 0, 'delivery_time'].median()
        self.data.loc[self.data['delivery_time'] < 0, 'delivery_time'] = median_del_time

    def _rel_purchase_freq(self):
        tmp = self.data.drop_duplicates(subset=['user_id', 'order_date'])
        tmp = tmp.sort_values(by=['user_id', 'order_date']).reset_index(drop=True)
        
        tmp['reg_dummy1'] = pd.to_datetime(tmp['user_reg_date']) < pd.to_datetime('2016-04-01')
        tmp['reg_dummy2'] = pd.to_datetime(tmp['user_reg_date']) >= pd.to_datetime('2016-04-01')
        tmp['day_diff1'] = (pd.to_datetime(tmp['order_date']) - pd.to_datetime('2016-04-01')).dt.days + 2 # +2 to prevent zero-division as some trx. have a reg date < order_date (by 1 day)
        tmp['day_diff2'] = (pd.to_datetime(tmp['order_date']) - pd.to_datetime(tmp['user_reg_date'])).dt.days + 2
        tmp['day_diff'] = tmp['reg_dummy1'] * tmp['day_diff1'] + tmp['reg_dummy2'] * tmp['day_diff2']
        tmp['purchase_event'] = tmp.groupby('user_id').cumcount() + 1
        tmp['rel_pfreq'] = tmp['purchase_event'] / tmp['day_diff']
        
        tmp = tmp.loc[:, ['user_id', 'order_date', 'rel_pfreq']]
        self.data = self.data.merge(tmp, how='left')

    def _ref_price(self, agg_fun: str = 'mean'):
        item_df = self.data.groupby(['item_id']).agg({
            'item_price': agg_fun
        }).reset_index().rename(columns={
            'item_price': f'{agg_fun}_price'
        })
        self.data = self.data.merge(item_df, how='left')
        self.data[f'{agg_fun}_ref_price'] = (self.data['item_price'] / self.data[f'{agg_fun}_price']).fillna(0)
        self.data[f'{agg_fun}_ref_price'] = self.data[f'{agg_fun}_ref_price'].replace([-np.inf, np.inf], 0)
        self.data.drop(labels=f'{agg_fun}_price', axis=1, inplace=True)

    def _rolling_ref_price(self, agg_fun: str = 'mean', window: int = 7):
        tmp = self.data.loc[:, ['item_id', 'order_date', 'item_price']]
        tmp = pd.pivot_table(data=tmp, values='item_price', index='order_date', columns='item_id', aggfunc=agg_fun).reset_index()
        tmp = pd.melt(tmp, id_vars='order_date', var_name='item_id', value_name=f'{agg_fun}_price').reset_index(drop=True)
        tmp[f'{window}d_rolling_{agg_fun}_price'] = tmp.groupby('item_id').rolling(window, min_periods=1).agg({
            f'{agg_fun}_price': agg_fun
        }).values
        tmp = tmp.drop(labels=f'{agg_fun}_price', axis=1)

        self.data = self.data.merge(tmp, how='left')
        self.data[f'{window}d_rolling_{agg_fun}_ref_price'] = (self.data['item_price'] / self.data[f'{window}d_rolling_{agg_fun}_price']).fillna(0)
        self.data.drop(labels=f'{window}d_rolling_{agg_fun}_price', axis=1, inplace=True)

    def _mep_in_basket(self):
        tmp = self.data.groupby(['user_id', 'order_date']).agg({
            'item_price': 'max'
        }).reset_index()

        tmp['most_exp_item'] = 1

        self.data = self.data.merge(tmp, how='left')
        self.data['most_exp_item'] = self.data['most_exp_item'].fillna(0)

    def _solo_item_purchase(self):
        tmp = self.data.groupby(['user_id', 'order_date']).agg({
            'order_item_id': 'count'
        }).reset_index().rename(columns={
            'order_item_id': 'counts'
        })
        tmp['solo_purchase'] = tmp['counts'].apply(lambda x: 1 if x == 1 else 0)
        self.data = self.data.merge(tmp.loc[:, ['user_id', 'order_date', 'solo_purchase']], how='left')

    def _cat_dummies(self):
        # unsized dummy
        self.data['unsized_dummy'] = self.data['item_size'].apply(lambda x: 1 if x == 'unsized' else 0)

        # sales dummy
        cond1 = pd.to_datetime(self.data['order_date']) >= pd.to_datetime('2016-06-26')
        cond2 = pd.to_datetime(self.data['order_date']) <= pd.to_datetime('2016-07-09')

        self.data['summer_sale'] = 0
        self.data.loc[cond1 & cond2, 'summer_sale'] = 1

        # color dummies
        keeper_colors = ['?', 'gold', 'mahagoni', 'pallid']
        self.data['color_keeper'] = 0
        self.data.loc[self.data['item_color'].isin(keeper_colors), 'color_keeper'] = 1

        return_colors = ['dark garnet', 'champagner', 'ingwer', 'floral', 'dark denim', 'dark navy']
        self.data['color_returner'] = 0
        self.data.loc[self.data['item_color'].isin(return_colors), 'color_returner'] = 1

        # user title dummies
        tmp = pd.get_dummies(data=self.data['user_title'], prefix='title', drop_first=True)
        tmp.columns = [i.lower().replace(" ", "_") for i in tmp.columns]
        self.data = pd.concat([self.data, tmp], axis=1)

    def _truncator(self, feature_list: list, thr: Union[int, list] = 10):
        if isinstance(thr, int):
            thr = np.tile(thr, len(feature_list))

        for idx, feature in enumerate(feature_list):
            count_df = self.data.groupby(feature).agg({
                'order_item_id': 'count'
            }).reset_index().rename(columns={
                'order_item_id': 'count'
            })

            cond = count_df['count'] < thr[idx]
            vals = count_df.loc[cond, feature].apply(str).values
            self.data[feature] = self.data[feature].apply(str)
            self.data.loc[self.data[feature].isin(vals), feature] = 'misc'

    def _compute_ce(self, feature_list: list):
        for feature in feature_list:
            ce_df = self.data.loc[self.data['known'] == 1, :].groupby(feature).agg({
                'return': 'mean'
            }).reset_index().rename(columns={
                'return': f'ce_{feature}'
            })
            self.data = self.data.merge(ce_df, how='left')
            self.data[f'ce_{feature}'] = self.data[f'ce_{feature}'].fillna(0.5)

    def _compute_woe(self, feature_list: list, correction: float = 0.01):
        for feature in feature_list:
            woe_df = self.data.loc[self.data['known'] == 1, :].groupby(feature).agg({
                'return': 'mean'
            }).reset_index().rename(columns={
                'return': f'return_rate'
            })

            # 0-correction
            woe_df['return_rate'] = woe_df['return_rate'].apply(lambda a: a + correction if a == 0 else a)
            # 1-correction
            woe_df['return_rate'] = woe_df['return_rate'].apply(lambda a: a - correction if a == 1 else a)

            # WOE-calculation
            woe_df[f'woe_{feature}'] = woe_df['return_rate'].apply(lambda a: np.log((1-a)/a))
            woe_df = woe_df.drop(labels='return_rate', axis=1)
            self.data = self.data.merge(woe_df, how='left')
            self.data[f'woe_{feature}'] = self.data[f'woe_{feature}'].fillna(0)

    def prep_wrapper(self, agg_fun, window: int = 7, cat_trans: str = None, feature_list: list = None, thr: Union[int, list] = 10, correction: float = 0.01):
        start = time.time()
        print(80 * '=')
        print("Start preparing data:")
        self._delivery_null_feature()
        print("\t 1.) Adding dummy for missing delivery date: Done.")
        self._delivery_1994_feature()
        print("\t 2.) Adding dummy for delivery date in 1994: Done.")
        self._time_features()
        print("\t 3.) Creating time features like 'age' or 'delivery_duration': Done.")
        self._rel_purchase_freq()
        print("\t 4.) Feature on purchase frequency per customer: Done.")
        self._mep_in_basket()
        print("\t 5.) Create dummy for most expensive product in a basket: Done.")
        self._solo_item_purchase()
        print("\t 6.) Dummy indicating if a product was bought alone or not: Done.")
        self._cat_dummies()
        print("\t 7.) Dummies for certain categories (e.g. unsized dummy, seasonal effect, etc.): Done.")
        for agg in agg_fun:
            self._ref_price(agg_fun=agg)
            self._rolling_ref_price(agg_fun=agg, window=window)
        print("\t 8.) Ref. prices: Done.")
        if cat_trans == 'WOE':
            self._truncator(feature_list=feature_list, thr=thr)
            self._compute_woe(feature_list=feature_list, correction=correction)
            print("\t 9.) WOE calculations: Done.")
        elif cat_trans == 'CE':
            self._truncator(feature_list=feature_list, thr=thr)
            self._compute_ce(feature_list=feature_list)
            print("\t 9.) CE calculations: Done.")
        else:
            pass
            #self._truncator(feature_list=feature_list, thr=thr)
        # Save  the preprocessed data:
        if self.save_path is not None:
            if cat_trans is None:
                self.data.to_parquet(self.save_path + '/data_cleaned.parquet')
                self.data.drop(labels=['order_date', 'delivery_date', 'user_dob', 'user_title', 'user_reg_date'], axis=1, inplace=True)
                for col in self.data.columns:
                    if self.data[col].dtype == 'float64':
                        self.data[col] = self.data[col].astype("float32")
                self.data.loc[self.data['known'] == 1, self.data.columns != 'known'].to_parquet(self.save_path + '/known_cleaned.parquet')
                self.data.loc[self.data['known'] == 0, self.data.columns != 'known'].to_parquet(self.save_path + '/unknown_cleaned.parquet')
            elif cat_trans == 'WOE':
                self.data.to_parquet(self.save_path + '/data_cleaned_woe.parquet')
                self.data.drop(labels=['order_date', 'delivery_date', 'user_dob', 'user_title', 'user_reg_date', 'item_id', 'item_size', 'item_color', 'user_id', 'brand_id', 'user_state'], axis=1, inplace=True)
                for col in self.data.columns:
                    if self.data[col].dtype == 'float64':
                        self.data[col] = self.data[col].astype("float32")
                self.data.loc[self.data['known'] == 1, self.data.columns != 'known'].to_parquet(self.save_path + '/known_cleaned_woe.parquet')
                self.data.loc[self.data['known'] == 0, self.data.columns != 'known'].to_parquet(self.save_path + '/unknown_cleaned_woe.parquet')
            elif cat_trans == 'CE':
                self.data.to_parquet(self.save_path + '/data_cleaned_ce.parquet')
                self.data.drop(labels=['order_date', 'delivery_date', 'user_dob', 'user_title', 'user_reg_date', 'item_id', 'item_size', 'item_color', 'user_id', 'brand_id', 'user_state'], axis=1, inplace=True)
                for col in self.data.columns:
                    if self.data[col].dtype == 'float64':
                        self.data[col] = self.data[col].astype("float32")
                self.data.loc[self.data['known'] == 1, self.data.columns != 'known'].to_parquet(self.save_path + '/known_cleaned_ce.parquet')
                self.data.loc[self.data['known'] == 0, self.data.columns != 'known'].to_parquet(self.save_path + '/unknown_cleaned_ce.parquet')
            else:
                raise ValueError("The argument 'cat_trans' must be either 'WOE', 'CE' or None (default -> no transformation).")
        else:
            self.data.drop(labels=['order_date', 'delivery_date', 'user_dob', 'user_title', 'user_reg_date'], axis=1, inplace=True)
        end = time.time()
        print(f"\nFeature creation finished in {(end-start)/60:.2f} min.\n")
        print(80 * "=")
