import joblib
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import os
import warnings


from collections import Counter
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from cross_validation import cross_validation, create_preprocessor
from graph_plots import plot_graph, plot_accuracy_cross_validation, plot_fscore_cross_validation, \
    plot_correlation_matrix, plot_pca_scatter
from utilities.utils import Utils

warnings.filterwarnings("ignore")

SHORT_WINDOW = int(Utils.load_config('SHORT_WINDOW'))
LONG_WINDOW = int(Utils.load_config('LONG_WINDOW'))
FLAT_THRESHOLD = 0.01

EXPORT_DATAFRAME_ON_CSV = False
EXPORT_CLASSIFIER = False
PLOT_GRAPH = False

CRYPTO_CURRENCY = "BTC-USD"
BEST_CLASSIFIER = 0  # RandomForest


def scraper(crypto_name):
    return web.get_data_yahoo(crypto_name, start="2020-01-01", end="2021-12-31")


def ranking_attributes_contribution(dataset):
    X = dataset[predictors]  # independent columns
    y = dataset[['trend']]  # target column

    best_features = SelectKBest(score_func=mutual_info_regression, k='all')
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([df_columns, df_scores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(8, 'Score'))


if __name__ == '__main__':
    scraped_df = scraper(CRYPTO_CURRENCY)

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

    # dataset['positions'] = dataset['trend'].shift(-1).diff()
    dataset['volume'] = scraped_df['Volume']

    # export the dataframe to csv
    if (EXPORT_DATAFRAME_ON_CSV == True):
        os.makedirs('data', exist_ok=True)
        dataset.to_csv('data/' + CRYPTO_CURRENCY + '_data.csv')

    dataset['trend'] = dataset['trend'].shift(-1)
    dataset = dataset.fillna(0)
    # we delete the last tuple that after shifting the dataframe lost its trend attribute value
    dataset = dataset.iloc[:-1, :]

    predictors = ['open',
                  'high',
                  'low',
                  'adj_close',
                  'ema_short',
                  'ema_long',
                  'volume'
                  ]

    if PLOT_GRAPH:
        plot_graph(dataset)
        # we can only perform one: show() or savefig()
        plt.show()
        plt.savefig('plots/labelled-graph.png')
        # plot_correlation_matrix(dataset)
        # plot_pca_scatter(dataset, predictors, CRYPTO_CURRENCY.split('-')[0])

    # CLASSIFIERS TRAINING
    classifiers = [
        RandomForestClassifier(criterion="entropy"),
        AdaBoostClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(),
        GaussianNB()
    ]

    final_table = {}

    overall_accuracy = {}
    overall_down = {}
    overall_flat = {}
    overall_up = {}

    for classifier in classifiers:
        avg_accuracy, f1_avg_down, f1_avg_flat, f1_avg_up, precision_avg_down, precision_avg_flat,  \
        precision_avg_up, recall_avg_down, recall_avg_flat, recall_avg_up, accuracy_scores,         \
        f1_down, f1_flat, f1_up = cross_validation(dataset, predictors, classifier)

        # we plot the result of the cross validation iterations for each model
        plot_accuracy_cross_validation(classifier, accuracy_scores)
        plot_fscore_cross_validation(classifier, f1_down, f1_flat, f1_up)

        overall_accuracy[str(classifier)] = "{:.3f}".format(avg_accuracy)
        overall_down[str(classifier)] = "{:.2f}".format(precision_avg_down) + "\t"*2 + "{:.2f}".format(recall_avg_down) + "\t"*2 + "{:.2f}".format(f1_avg_down)
        overall_flat[str(classifier)] = "{:.2f}".format(precision_avg_flat) + "\t"*2 + "{:.2f}".format(recall_avg_flat) + "\t"*2 + "{:.2f}".format(f1_avg_flat)
        overall_up[str(classifier)] = "{:.2f}".format(precision_avg_up) + "\t"*2 + "{:.2f}".format(recall_avg_up) + "\t"*2 + "{:.2f}".format(f1_avg_up)

    print("\n")
    print("Classifier name" + "\t"*3 + "class" + "\t" + "precision" + "\t" + "recall" + "\t"*2 + "f1-score" + "\t" + "accuracy")
    for classifier in classifiers:
        print(str(classifier).split('(')[0])

        print("\t" * 6 + '-1.0' + '\t' * 1 + overall_down[str(classifier)] + '\t'*2 + overall_accuracy[str(classifier)])
        print("\t" * 6 + '0.0' + '\t' * 2 + overall_flat[str(classifier)])
        print("\t" * 6 + '1.0 ' + '\t' * 1 + overall_up[str(classifier)])
        print()

    print("\nThe classifier that performs better in terms of Accuracy and F1-score is the " + str(
        classifiers[BEST_CLASSIFIER]))

    # we train the selected model

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    x_train, x_test, y_train, y_test = train_test_split(dataset[predictors],
                                                        dataset[['trend']], test_size=.30,
                                                        shuffle=False, random_state=0)

    print("Class label for training set : ", Counter(y_train['trend']))
    print("Class label for test set : ", Counter(y_test['trend']))

    preprocessor = create_preprocessor(predictors, None)

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor)
        , ('classifier', classifiers[BEST_CLASSIFIER])
    ])

    # train the model
    pipe.fit(x_train, y_train.values.ravel())

    # use model to make predictions
    y_pred = pipe.predict(x_test)

    # evaluate the performance
    print("\nTraining ", str(classifiers[BEST_CLASSIFIER]))
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy on test set: ", accuracy)
    print("Metrics per class on test set:")
    print("Confusion matrix:")

    metrics.confusion_matrix(y_test, y_pred)
    print(metrics.classification_report(y_test, y_pred))

    # we train the final model on the entire dataset

    x_final_train = dataset[predictors]
    y_final_train = dataset[['trend']]

    # save the model
    if EXPORT_CLASSIFIER:
        filename = 'model/' + CRYPTO_CURRENCY + '_classifier.pkl'
        final_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor)
            , ('classifier', classifiers[BEST_CLASSIFIER])
        ]).fit(x_final_train, y_final_train.values.ravel())
        joblib.dump(final_pipe, filename)

    #ranking_attributes_contribution(dataset.iloc[0:500])

