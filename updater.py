
import joblib
import pandas as pd
from main import create_preprocessor
from sklearn.pipeline import Pipeline

# LOAD AGAIN THE MODEL

filename = 'model/crypto_predictor.pkl'
clf = joblib.load(filename)

dataset = pd.read_csv('data/BTC_crypto_data.csv')

predictors = ['open',
              'high',
              'low',
              'adj_close',
              'ema_short',
              'ema_long',
              'volume'
              ]

#tuple_to_predict = dataset.iloc[len(dataset.index)-1][predictors].to_frame()

#dataset.drop(columns=['Date', 'diff_ema', 'trend', 'positions'], inplace=True)
record_to_predict = dataset.loc[dataset['Date'] == '2021-10-12'][predictors]

#preprocessor = create_preprocessor(predictors)

#preprocessed_tuple_to_predict = preprocessor.transform(tuple_to_predict)
print(record_to_predict)
result = clf.predict(record_to_predict)

print("Prediction for 2022-02-25: " + str(result))