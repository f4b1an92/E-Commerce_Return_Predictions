import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime
import utils


def time_series_plot(data, vdates=None, fsize=(20, 15)):
    # Detect if target variable is in te
    if "return" in data.columns:
        upper_lim = 3
    else:
        data = data.copy()
        data['return'] = 0  # return column won't be used by plots but needs to be present for preparing the input df.
        upper_lim = 2

    # Calculate standardized item prices (using sklearn MinMaxScaler)
    fragment_list = []
    for g, df in data.groupby('item_id'):
        scaler = MinMaxScaler()
        scaler.fit(df['item_price'].values.reshape(-1, 1))

        df['item_price_std'] = scaler.transform(df['item_price'].values.reshape(-1, 1))
        fragment_list.append(df)

    data = pd.concat(fragment_list, axis=0).sort_values(by='order_item_id').reset_index(drop=True)

    # Prepare input df for plots
    input_df = data.groupby('order_date').agg({
        'order_item_id': 'count',
        'item_price_std': 'mean',
        'return': 'mean'}).reset_index().reset_index()
    input_df = input_df.rename(columns={'index': 'order_day', 'order_item_id': 'count', 'item_price_std': 'avg_price'})
    input_df['order_date'] = pd.to_datetime(input_df['order_date'])
    input_df['mcount'] = input_df['count'].rolling(7).mean()
    input_df['mavg_price'] = input_df['avg_price'].rolling(7).mean()
    input_df['mreturn'] = input_df['return'].rolling(7).mean()

    # Create the actual plot
    fig, ax = plt.subplots(upper_lim, 1, figsize=fsize)

    var_type = ['count', 'avg_price', 'return']
    var_color = ['r', 'b', 'g']
    mavg_color = ['black', 'orange', 'red']
    vline_color = ['skyblue', 'green', 'purple']
    y_label_list = ["Number of Item Purchases", "Avg. Item Price \n (Min-Max-Scaled)", 'Return Rate']

    for idx in range(upper_lim):
        sns.lineplot(data=input_df, x='order_date', y=var_type[idx], ax=ax[idx], color=var_color[idx])
        sns.lineplot(data=input_df, x='order_date', y=f'm{var_type[idx]}', ax=ax[idx], color=mavg_color[idx])
        if vdates is not None:
            for vd in vdates:
                ax[idx].axvline(pd.to_datetime(vd), color=vline_color[idx], linestyle='--')
        # Set plot title
        if (idx == 0) & (upper_lim == 3):
            ax[idx].set_title(
                "Price and Demand Development over Time in Relation to the Return Rate for the Train Set",
                size=upper_lim * 8
            )
        elif (idx == 0) & (upper_lim == 2):
            ax[idx].set_title("Price and Demand Development over Time for the Test Set", size=upper_lim * 8)
        else:
            pass
        # Set x-axis label
        if idx == (upper_lim-1):
            ax[idx].set_xlabel("Order Date")
        ax[idx].set_ylabel(y_label_list[idx], size=upper_lim * 6)
    plt.tight_layout()
    plt.show()


def hist_plot_comparison(train, test, var, var_label_title, fsize=(12, 8)):
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=fsize)
    sns.set_style('white')

    sns.histplot(data=train, x=var, kde=True, ax=ax, element='step', fill=True, stat='density', label='train set')
    sns.histplot(
        data=test, x=var, kde=True, ax=ax, color='red', element='step', fill=True, stat='density', label='test set'
    )
    ax.set_title(f"Distribution of {var_label_title} across 'known' and 'unknown' Data Sets", size=16)
    ax.set_xlabel(var_label_title, size=12)
    ax.set_ylabel("Density", size=12)
    plt.legend(prop={'size': 16})
    plt.show()


def violin_plot_comparison(train, test, dimx, dimy, dimx_label, dimy_label, fsize=(16, 8)):
    train['type'] = 'train'
    test['type'] = 'test'

    data = pd.concat([train, test], axis=0).reset_index(drop=True)
    tmp = data.loc[data[dimx].isin([1, 2, 3, 4, 5]), :]

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=fsize)
    sns.set_style('white')

    ax = sns.violinplot(x=dimx, y=dimy, hue="type", data=tmp, palette="muted")
    ax.set_title(f"Distribution of {dimy_label} across both Data Sets for the first 5 {dimx_label}s", size=16)
    ax.set_xlabel(dimx_label, size=12)
    ax.set_ylabel(dimy_label, size=12)
    plt.legend(prop={'size': 16})
    plt.show()


def freq_thr_plot(col_list: list, path: str, xmax: int = 30):
    df_list = []

    for idx in range(1, xmax+1):
        data = pd.read_csv(path + 'BADS_WS1819_known.csv')
        utils.truncator(data=data, feature_list=col_list, thr=idx)
        iv_dict = utils.compute_iv(data=data, woe_vars=col_list, correction=0.00001)
        df_list.append(pd.DataFrame({key: [val] for key, val in iv_dict.items()}))

    df = pd.concat(df_list, axis=0).reset_index(drop=True).reset_index()
    df['index'] = df['index'] + 1

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    sns.lineplot(
        data=df.melt(id_vars=['index'], var_name='cat_var', value_name='iv')
        , x='index'
        , y='iv'
        , hue='cat_var'
        , ax=ax
    )
    ax.hlines(y=0.5, xmin=1, xmax=xmax, linewidth=2, color='black', linestyles='dashdot')
    ax.set_title("Development of Information Value over different Frequency Thresholds for merging Categories")
    ax.set_xlabel("Frequency Thresholds")
    ax.set_ylabel("IV")
    plt.show()
