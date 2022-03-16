from persistence.MongoConnector import MongoConnector
from updater import Updater

#result = MongoConnector.get_collection('bitcoin').find()
#print(result.next())

up = Updater()
#dataframe = up.get_missing_tuples('BTC')

#final_df = up.compute_statistical_values(original_df=dataframe)

#up.store_missing_tuples('BTC', final_df)

up.update_currencies_collections()