import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

SHORT_WINDOW = 2
LONG_WINDOW = 5

def scraper(crypto_name):
    return web.get_data_yahoo(crypto_name, start="2021-01-01", end="2021-12-30")

def plot_graph(dataset):
    plt.title('BTC-USD Adj Close Price', fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Adjusted Price")

    ax = dataset['adj_close'].plot(lw=3, figsize=(14, 7), label='Original observations')
    dataset['ema_short'].plot(ax=ax, lw=3, label='EMA (window ' + str(SHORT_WINDOW) + ')')
    dataset['ema_long'].plot(ax=ax, lw=3, label='EMA (window ' + str(LONG_WINDOW) + ')')

    plt.tick_params(labelsize=12)
    plt.legend(loc='upper left', fontsize=12)

    plt.plot(dataset.loc[dataset.positions == 1.0].index,
             dataset.ema_short[dataset.positions == 1.0],
             '^', markersize=10, color='r', label='buy')

    plt.plot(dataset.loc[dataset.positions == -1.0].index,
             dataset.ema_long[dataset.positions == -1.0],
             'v', markersize=10, color='k', label='sell')

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    crypto = "BTC-USD"
    scraped_df = scraper(crypto)

    # we evaluate the exponential moving average of the short and long windows
    ema_short = scraped_df['Adj Close'].ewm(span=SHORT_WINDOW).mean()
    ema_long = scraped_df['Adj Close'].ewm(span=LONG_WINDOW).mean()

    dataset = pd.DataFrame(index=scraped_df['Adj Close'].index)
    dataset['open'] = scraped_df['Open']
    dataset['high'] = scraped_df['High']
    dataset['low'] = scraped_df['Low']
    dataset['adj_close'] = scraped_df['Adj Close']
    dataset['ema_short'] = ema_short
    dataset['ema_long'] = ema_long
    dataset['diff_ema'] = 0.0
    dataset['trend'] = 0.0

    dataset['trend'][SHORT_WINDOW:] = np.where(dataset['ema_short'][SHORT_WINDOW:] > dataset['ema_long'][SHORT_WINDOW:], 1, 0)
    dataset['positions'] = dataset['trend'].shift(-1).diff()
    dataset['diff_ema'] = dataset['ema_short'] - dataset['ema_long']
    dataset['volume'] = scraped_df['Volume']

    dataset = dataset.fillna(0)
    dataset['trend'] = dataset['trend'].replace(1, 'up')
    dataset['trend'] = dataset['trend'].replace(0, 'down')

    dataset['trend'] = dataset['trend'].shift(-1)

    os.makedirs('data', exist_ok=True)
    dataset.to_csv('data/BTC_crypto_data.csv')

    print(dataset)
    #plot_graph(dataset)

    predictors = ['open',
                  'high',
                  'low',
                  'adj_close',
                  'ema_short',
                  'ema_long',
                  'volume'
                  ]

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()


    x_train, x_test, y_train, y_test = train_test_split(dataset[predictors],
                                              dataset[['trend']], test_size=.333,
                                              shuffle=False, random_state=0)

    print('Size of train set: ', x_train)
    print('Size of test set: ', x_test)
    print('Size of train set: ', y_train)
    print('Size of test set: ', y_test)





