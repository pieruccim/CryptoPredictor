from datetime import datetime
from tornado import web


class Updater:
    last_check_timestamp = None
    last_date_mongo = None

    def __init__(self):
        self.last_date_mongo = #query to retrieve the day of last date to mongo
        self.last_check_timestamp = datetime.now()

    def scrape_from_yahoo(self, crypto_name, to_date=datetime.today().strftime('%Y-%m-%d')):
        return web.get_data_yahoo(crypto_name, start=self.last_date_mongo, end=to_date)

    def evaluate_statistical_indicator(self):
        return

    def update_database(self):
        return
