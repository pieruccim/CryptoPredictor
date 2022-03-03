import pandas_datareader.data as web
import datetime

btc = web.get_data_yahoo('BTC-USD', start=datetime.datetime(2017, 1, 1), end=datetime.datetime(2022, 3, 1))
btc.to_csv('tabella.csv')