import joblib
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import Counter
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

SHORT_WINDOW = 2
LONG_WINDOW = 5
FLAT_THRESHOLD = 0.01

N_SPLITS = 9 #28
TEST_SIZE = 30 #60


def scraper(crypto_name):
    return web.get_data_yahoo(crypto_name, start="2020-01-01", end="2021-12-31")


def plot_graph(dataset):
    plt.title('BTC-USD Adj Close Price', fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Adjusted Price")

    ax = dataset['adj_close'].plot(lw=3, figsize=(14, 7), label='Original observations')
    dataset['ema_short'].plot(ax=ax, lw=3, label='EMA (window ' + str(SHORT_WINDOW) + ')')
    dataset['ema_long'].plot(ax=ax, lw=3, label='EMA (window ' + str(LONG_WINDOW) + ')')

    plt.tick_params(labelsize=12)
    plt.legend(loc='upper left', fontsize=12)

    plt.plot(dataset.loc[dataset.trend == 1.0].index,
             dataset.adj_close[dataset.trend == 1.0],
             '^', markersize=8, color='g', label='up')

    plt.plot(dataset.loc[dataset.trend == 0.0].index,
             dataset.adj_close[dataset.trend == 0.0],
             '^', markersize=8, color='y', label='flat')

    plt.plot(dataset.loc[dataset.trend == -1.0].index,
             dataset.adj_close[dataset.trend == -1.0],
             '^', markersize=8, color='r', label='down')

    '''plt.plot(dataset.loc[dataset.positions == 1.0].index,
             dataset.ema_short[dataset.positions == 1.0],
             '^', markersize=10, color='r', label='buy')

    plt.plot(dataset.loc[dataset.positions == -1.0].index,
             dataset.ema_long[dataset.positions == -1.0],
             'v', markersize=10, color='k', label='sell')'''

    plt.show()


def create_preprocessor(predictors):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
        ("pca", PCA())
    ])

    numeric_features = predictors

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features)
        ])


def cross_validation(dataset, predictors, classifier):
    # CROSS VALIDATION

    # n_splits is the number of subsets created
    # test_size is the number of records for each test sets
    tscv = TimeSeriesSplit(gap=0, n_splits= N_SPLITS, test_size=TEST_SIZE)

    accuracy_scores = []
    f1_scores_down = []
    f1_scores_flat = []
    f1_scores_up = []
    for train_index, test_index in tscv.split(dataset):
        print("--------------------------------------------------------------------------")
        #print("TRAIN:", train_index, "\n TEST:", test_index)

        train_dataset, test_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
        x_train, x_test = train_dataset[predictors], test_dataset[predictors]
        y_train, y_test = train_dataset[['trend']], test_dataset[['trend']]

        print("Class label for training set : ", Counter(y_train['trend']))
        print("Class label for test set : ", Counter(y_test['trend']))

        preprocessor = create_preprocessor(predictors)

        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor)
            , ('classifier', classifier)
        ])

        # Train the model
        pipe.fit(x_train, y_train.values.ravel())

        # Use model to make predictions
        y_pred = pipe.predict(x_test)

        # Evaluate the performance
        print("\nTraining ", classifier)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print("Accuracy on test set: ", accuracy)
        print("Metrics per class on test set:")
        print("Confusion matrix:")
        metrics.confusion_matrix(y_test, y_pred)
        print(metrics.classification_report(y_test, y_pred))

        f1 = f1_score(y_test, y_pred, average=None)

        try:
            f1_scores_down.append(f1[0])
        except Exception as e:
            print("Exception: " + str(e))
        try:
            f1_scores_flat.append(f1[1])
        except Exception as e:
            print("Exception: " + str(e))
        try:
            f1_scores_up.append(f1[2])
        except Exception as e:
            print("Exception: " + str(e))

    print("ACCURACY OVERALL MEAN: " + str(np.mean(accuracy_scores)) + " FOR CLASSIFIER: " + str(classifier))
    print("F1 OVERALL MEAN FOR LABEL -1: " + str(np.mean(f1_scores_down)) + " / FOR LABEL 0: " + str(
        np.mean(f1_scores_flat)) + " / FOR LABEL 1: " + str(
        np.mean(f1_scores_up)) + " FOR CLASSIFIER: " + str(classifier))
    return np.mean(accuracy_scores), np.mean(f1_scores_down), np.mean(f1_scores_flat), np.mean(f1_scores_up)


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

    dataset['diff_ema'] = dataset['ema_short'] - dataset['ema_long']

    # we set the labels for each day: -1 DOWN, 0 FLAT and 1 UP

    for index, row in dataset.iterrows():
        if row.diff_ema > 0:
            if abs(row.diff_ema) > (row.adj_close * FLAT_THRESHOLD):
                dataset.loc[index, 'trend'] = 1
            else:
                dataset.loc[index, 'trend'] = 0
        elif row.diff_ema <= 0:
            if abs(row.diff_ema) > (row.adj_close * FLAT_THRESHOLD):
                dataset.loc[index, 'trend'] = -1
            else:
                dataset.loc[index, 'trend'] = 0


    #dataset['positions'] = dataset['trend'].shift(-1).diff()
    dataset['volume'] = scraped_df['Volume']

    os.makedirs('data', exist_ok=True)
    dataset.to_csv('data/BTC_crypto_data.csv')

    dataset['trend'] = dataset['trend'].shift(-1)
    dataset = dataset.fillna(0)

    #plot_graph(dataset)

    predictors = ['open',
                  'high',
                  'low',
                  'adj_close',
                  'ema_short',
                  'ema_long',
                  'volume'
                  ]

    # CLASSIFIERS TRAINING
    classifiers = [
        RandomForestClassifier(),
        AdaBoostClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(),
        GaussianNB()
    ]

    overall_accuracy_avg_means = {}
    overall_f1 = {}
    for classifier in classifiers:
        avg_accuracy, f1_avg_down, f1_avg_flat, f1_avg_up = cross_validation(dataset, predictors, classifier)
        overall_accuracy_avg_means[str(classifier)] = avg_accuracy
        overall_f1[str(classifier)] = str(f1_avg_down) + "\t" + str(f1_avg_flat) + "\t" + str(f1_avg_up)

    print("\n")
    print("CLASSIFIER NAME \t\t\t\t\t ACCURACY MEAN \t\t F1 SCORE MEAN FOR -1 \t F1 SCORE MEAN FOR 0 \t F1 SCORE MEAN FOR 1")
    for classifier in classifiers:
        print(str(classifier) + "\t\t\t" + str(overall_accuracy_avg_means[str(classifier)]) + "\t" + str(overall_f1[str(classifier)]))

    print("\nThe classifier that performs better in terms of Accuracy and F1-score is the " + str(classifiers[0]))

    # TRAINING WITH THE FINAL MODEL

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    x_train, x_test, y_train, y_test = train_test_split(dataset[predictors],
                                                        dataset[['trend']], test_size=.30,
                                                        shuffle=False, random_state=0)

    preprocessor = create_preprocessor(predictors)

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor)
        , ('classifier', classifiers[0])
    ])

    # Train the model
    pipe.fit(x_train, y_train.values.ravel())

    # Use model to make predictions
    y_pred = pipe.predict(x_test)

    # Evaluate the performance
    print("\nTraining ", str(classifiers[0]))
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy on test set: ", accuracy)
    print("Metrics per class on test set:")
    print("Confusion matrix:")

    metrics.confusion_matrix(y_test, y_pred)
    print(metrics.classification_report(y_test, y_pred))

    # WE TRAIN THE FINAL MODEL ON THE WHOLE DATASET

    x_final_train = dataset[predictors]
    y_final_train = dataset[['trend']]

    # SAVE THE MODEL

    filename = 'model/crypto_predictor.pkl'
    final_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor)
        , ('classifier', classifiers[0])
    ]).fit(x_final_train, y_final_train.values.ravel())
    joblib.dump(final_pipe, filename)


