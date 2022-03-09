import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', None)

data_path = 'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/1.) WS_1819/BADS2/data/'

known = pd.read_csv(data_path + 'BADS_WS1819_known.csv')
known['known'] = 1
known['brand_id'] = known['brand_id'] - 100

unknown = pd.read_csv(data_path + 'BADS_WS1819_unknown.csv')
unknown['return'] = np.nan
unknown['known'] = 0

data = pd.concat([known, unknown], axis=0).reset_index(drop=True)


tmp = data.groupby(['item_size'])['order_item_id'].count().reset_index().rename(columns={'order_item_id': 'count'})
tmp.sort_values(by='count', ascending=False, inplace=True)

# 1. Create dummy for data points that don't have a delivery date (100% return probability)
data['dd_null'] = data['delivery_date'].apply(lambda x: 1 if pd.isna(x) else 0)

# 2. Date Columns:
## 2.1. Age
tmp = pd.to_datetime(data['order_date']) - pd.to_datetime(data['user_dob'])
data['age'] = np.round(tmp.dt.days / 365)

## 2.2. Delivery Duration/Time
tmp = pd.to_datetime(data['delivery_date']) - pd.to_datetime(data['order_date'])
data['delivery_time'] = tmp.dt.days

## 2.3. Customer lifetime
tmp = pd.to_datetime(data['order_date']) - pd.to_datetime(data['user_reg_date'])
data['clt'] = tmp.dt.days + 1

# 3. User IDs
tmp = data.loc[data['dd_null'] == 0, :].groupby('user_id').agg({'return': ['count', 'mean']}).reset_index()#.rename(columns={'order_item_id':'num_trx'}).sort_values(by='num_trx', ascending=False)
tmp.columns = ['user_id', 'num_trx', 'ret_rate']
tmp.sort_values(by='num_trx', ascending=False)

tmp.groupby('num_trx')['ret_rate'].mean().reset_index()




tmp = data.loc[data['user_id'] == 2649, :]
tmp = tmp.sort_values(by='order_date')

known = data.loc[data['known'] == 1, :]
known['return'].mean()
tmp = known.groupby('user_id')['order_item_id'].count().reset_index()
tmp = tmp.loc[tmp['order_item_id'] == 1, 'user_id']

known.loc[known['user_id'].isin(tmp), 'return'].mean()


tmp = known.groupby(['user_id', 'order_date'])['order_item_id'].count().reset_index().rename(columns={
    'order_item_id': 'num_items'
})
tmp2 = tmp.groupby('user_id')['order_date'].count().reset_index().rename(columns={
    'order_date': 'num_orders'
})
tmp = tmp.merge(tmp2, how='left')
tmp = tmp.loc[(tmp['num_orders'] > 1) & (tmp['num_items'] == 1), ['user_id', 'order_date']].reset_index(drop=True)

tmp_list = []
for i in tqdm(range(tmp.shape[0])):
    cond1 = known['user_id'] == tmp.loc[i, 'user_id']
    cond2 = known['order_date'] == tmp.loc[i, 'order_date']

    tmp_list.append(known.loc[cond1 & cond2, 'return'])

np.mean(tmp_list)

tmp = tmp['order_item_id']

tmp.sort_values(by='num_trx', ascending=False).iloc[0, :]
len(np.unique(data.loc[data['user_id'] == 2649, 'order_date']))


## 3.1. Purchase frequency in days (i.e. registry time per user id / number of purchases per user id)
data = data.sort_values(by='order_date').reset_index()

pfreq_list = []
count_dict = {}

for i in tqdm(range(data.shape[0])):
    uid = data.loc[i, 'user_id']
    if uid in list(count_dict.keys()):
        count_dict[uid] += 1
        pfreq = data.loc[i, 'reg_time'] / count_dict[uid]
        pfreq_list.append(pfreq)
    else:
        count_dict[uid] = 1
        pfreq = data.loc[i, 'reg_time'] / count_dict[uid]
        pfreq_list.append(pfreq)

data['pfreq'] = pfreq_list

## 3.2. Days since last purchase (per user id)


# 4. Price
## 4.1. Rel. difference to the mean price of a product
item_df = data.groupby(['item_id']).agg({
    'item_price': 'mean'
}).reset_index().rename(columns={
    'item_price': 'mean_price'
})

data = data.merge(item_df, how='left')
data['mean_ref_price'] = data['item_price'] / data['mean_price']

## 4.2. Rel. difference to the median price of a product
item_df = data.groupby(['item_id']).agg({
    'item_price': 'median'
}).reset_index().rename(columns={
    'item_price': 'median_price'
})

data = data.merge(item_df, how='left')
data['median_ref_price'] = data['item_price'] / data['median_price']

## 4.3. Rel. difference to the max price of a product
item_df = data.groupby(['item_id']).agg({
    'item_price': 'max'
}).reset_index().rename(columns={
    'item_price': 'max_price'
})

data = data.merge(item_df, how='left')
data['median_ref_price'] = data['item_price'] / data['max_price']

## 4.4. 7-Day-Rolling-Median-Reference Price
data['7d_median_ref_price'] = np.nan

item_ids = np.unique(data['item_id'])
dates = np.unique(data['order_date'])

for i in tqdm(item_ids):
    df_price = pd.DataFrame({'order_date':dates})
    df_price['item_id'] = i

    cond = data['item_id'].values == i
    df_mp = data.loc[cond,:].groupby(['order_date']).agg({
        'item_price':'median'
    }).reset_index().rename(columns={'item_price':'median_price'})

    df_price = df_price.merge(df_mp,how='left')

    df_price['moving_med'] = df_price['median_price'].rolling(7,min_periods=1).median()
    df_price['moving_med'].fillna(method='ffill',inplace=True)
    df_price['moving_med'].fillna(method='bfill',inplace=True)
    df_price.drop(labels='median_price',axis=1,inplace=True)

    data = data.merge(df_price,how='left')
    data['7d_median_ref_price'].fillna(data['moving_med'],inplace=True)
    data.drop(labels='moving_med',axis=1,inplace=True)

## 4.5. 7-Day-Rolling-Max-Reference Price
data['7d_max_ref_price'] = np.nan

item_ids = np.unique(data['item_id'])
dates = np.unique(data['order_date'])

for i in tqdm(item_ids):
    df_price = pd.DataFrame({'order_date':dates})
    df_price['item_id'] = i

    cond = data['item_id'].values == i
    df_mp = data.loc[cond,:].groupby(['order_date']).agg({
        'item_price':'max'
    }).reset_index().rename(columns={'item_price':'max_price'})

    df_price = df_price.merge(df_mp,how='left')

    df_price['moving_max'] = df_price['max_price'].rolling(7,min_periods=1).max()
    df_price['moving_max'].fillna(method='ffill',inplace=True)
    df_price['moving_max'].fillna(method='bfill',inplace=True)
    df_price.drop(labels='max_price',axis=1,inplace=True)

    data = data.merge(df_price,how='left')
    data['7d_max_ref_price'].fillna(data['moving_max'],inplace=True)
    data.drop(labels='moving_max',axis=1,inplace=True)

# 5. Brand ID (dummify)

# 6 User state (dummify the variable)


# 7. User title (dummify the variable)


# Still missing: 'item_id', 'item_size', 'item_color', 'brand_id'