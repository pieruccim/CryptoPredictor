
import joblib
import pandas as pd
from tornado import web

from main import create_preprocessor
from sklearn.pipeline import Pipeline

# LOAD AGAIN THE MODEL

filename = 'model/crypto_predictor.pkl'
clf = joblib.load(filename)

predictors = ['open',
              'high',
              'low',
              'adj_close',
              'ema_short',
              'ema_long',
              'volume'
              ]
crypto_name = 'BTC-USB'

df = web.get_data_yahoo(crypto_name, start="2020-01-01", end="2022-03-10")

record_to_predict = df.loc[df['Date'] == '2022-01-02'][predictors]

#preprocessor = create_preprocessor(predictors)

#preprocessed_tuple_to_predict = preprocessor.transform(tuple_to_predict)
print(record_to_predict)
result = clf.predict(record_to_predict)

print("Prediction for 2022-02-25: " + str(result))