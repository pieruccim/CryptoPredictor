from datetime import datetime, timedelta
from utilities.utils import Utils

import pandas
import pandas_datareader as web
from persistence.MongoConnector import MongoConnector


class Updater:
    connection = MongoConnector()
    last_check_timestamp = None
    #last_date_mongo = None

    '''
    def __init__(self):
        self.last_date_mongo = self.check_last_date('bitcoin')
        self.last_check_timestamp = datetime.now()
    '''

    def check_last_date(self, collection_name: str):

        collection = self.connection.get_collection(collection_name)
        document = collection.find().sort("Date", -1).limit(1)
        last_date = document.next()['Date']
        return datetime.strptime(last_date, '%Y-%m-%d')

    def get_missing_tuples(self, crypto_name):

        most_recent_date = self.check_last_date(crypto_name)
        days = timedelta(int(Utils.load_config("LONG_WINDOW")))
        most_recent_shifted = most_recent_date - days

        print('Most recent date for ' + crypto_name + ': ' + str(most_recent_date))

        one = timedelta(1)
        return web.get_data_yahoo(crypto_name+'-USD',
                                  start=most_recent_shifted.strftime('%Y-%m-%d'),
                                  end=(datetime.today() - one).strftime('%Y-%m-%d'))

    def compute_statistical_values(self, original_df: pandas.DataFrame):

        # we evaluate the exponential moving average of the short and long windows
        ema_short = original_df['Adj Close'].ewm(span=int(Utils.load_config('SHORT_WINDOW'))).mean()
        ema_long = original_df['Adj Close'].ewm(span=int(Utils.load_config('LONG_WINDOW'))).mean()

        # we create a new dataframe to easily manage the column names
        final_dataset = pandas.DataFrame(index=original_df['Adj Close'].index)

        # trend is note present because these tuples are only used for prediction and not to train a model
        final_dataset['open'] = original_df['Open']
        final_dataset['high'] = original_df['High']
        final_dataset['low'] = original_df['Low']
        final_dataset['adj_close'] = original_df['Adj Close']
        final_dataset['ema_short'] = ema_short
        final_dataset['ema_long'] = ema_long
        final_dataset['diff_ema'] = final_dataset['ema_short'] - final_dataset['ema_long']
        final_dataset['volume'] = original_df['Volume']

        return final_dataset

    def store_missing_tuples(self, collection_name: str, ema_df: pandas.DataFrame):

        # we reset the index in order to add the Date as a field of the MongoDB Document
        ema_df = ema_df.reset_index(level=0)
        # we change the Date field from datetime to string
        ema_df['Date'] = ema_df['Date'].astype('str')

        # we need to drop some tuples already present in mongo, but necessary to evaluate the statistical values
        ema_df = ema_df.drop([0, 1, 2, 3, 4])

        dict_df = ema_df.to_dict('records')
        if dict_df:
            result = self.connection.get_collection(collection_name).insert_many(dict_df)
            return result
        else:
            return False

    def update_currencies_collections(self):

        crypto_currencies = Utils.load_config("CRYPTO_CURRENCIES").split(", ")
        for crypto in crypto_currencies:
            dataframe = self.get_missing_tuples(crypto)
            final_df = self.compute_statistical_values(original_df=dataframe)
            self.store_missing_tuples(crypto, final_df)
        return


