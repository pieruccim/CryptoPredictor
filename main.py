import joblib
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sn

from collections import Counter
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

SHORT_WINDOW = 3
LONG_WINDOW = 6
FLAT_THRESHOLD = 0.01

N_SPLITS = 9  # 28
TEST_SIZE = 30  # 60

EXPORT_DATAFRAME_ON_CSV = False
EXPORT_CLASSIFIER = False
PLOT_GRAPH = False

CRYPTO_CURRENCY = "BTC-USD"
BEST_CLASSIFIER = 0  # RandomForest


def scraper(crypto_name):
    return web.get_data_yahoo(crypto_name, start="2020-01-01", end="2021-12-31")


def plot_correlation_matrix(dataset):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.
        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot
    """

    print(dataset.head())
    corrMatrix = dataset.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


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
             '^', markersize=6, color='g', label='up')

    plt.plot(dataset.loc[dataset.trend == 0.0].index,
             dataset.adj_close[dataset.trend == 0.0],
             'o', markersize=6, color='y', label='flat')

    plt.plot(dataset.loc[dataset.trend == -1.0].index,
             dataset.adj_close[dataset.trend == -1.0],
             'v', markersize=6, color='r', label='down')


def plot_pca_scatter(df, pred, crypto_name):
    x = df[predictors]
    y = df[['trend']]
    classes = ['down', 'flat', 'up']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pre = create_preprocessor(pred, 3)
    x_new_3d = pre.fit_transform(x)

    scat = ax.scatter(x_new_3d[:, 0], x_new_3d[:, 1], x_new_3d[:, 2], c=y)
    ax.set_title(crypto_name, fontsize=16)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(handles=scat.legend_elements()[0], labels=classes, loc='upper left', fontsize=12)

    plt.savefig('plots/scatter_plots/pca_scatter_plot_3d_' + crypto_name + '.png')

    y = df['trend']
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)

    pre2 = create_preprocessor(pred, 2)
    x_new_2d = pre2.fit_transform(x)

    scat2 = ax.scatter(x_new_2d[:, 0], x_new_2d[:, 1], c=y)
    ax.set_title(crypto_name, fontsize=16)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(handles=scat2.legend_elements()[0], labels=classes, loc='upper left', fontsize=12)

    plt.savefig('plots/scatter_plots/pca_scatter_plot_2d_' + crypto_name + '.png')


def plot_accuracy_cross_validation(clf, scores):
    df_accuracy = pd.DataFrame(scores, columns=['accuracy'])
    df_accuracy.plot()
    plt.title(str(clf).split('(')[0] + ' accuracy in cross-validation', fontsize=16)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.savefig('plots/plot-accuracy-cross-validation/' + str(clf).split('(')[0] + '.png')


def plot_fscore_cross_validation(clf, f1_down, f1_flat, f1_up):
    # print(str(f1_down)+","+str(f1_flat)+","+str(f1_up))
    try:
        df_f1score = pd.DataFrame({
            "f1_down": f1_down,
            "f1_flat": f1_flat,
            "f1_up": f1_up
        })
        df_f1score.fillna(0)
        df_f1score.plot()
    except:
        pass

    plt.title(str(clf).split('(')[0] + ' f score in cross-validation', fontsize=16)
    plt.xlabel("Iteration")
    plt.ylabel("F-Score")
    plt.ylim([0, 1])
    plt.savefig('plots/plot-fscore-cross-validation/' + str(clf).split('(')[0] + '.png')


def create_preprocessor(pred, comp):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
        ("pca", PCA(n_components=comp))
    ])

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, pred)
        ])


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


def cross_validation(dataset, predictors, classifier):
    # CROSS VALIDATION

    # n_splits is the number of subsets created
    # test_size is the number of records for each test sets
    tscv = TimeSeriesSplit(gap=0, n_splits=N_SPLITS, test_size=TEST_SIZE)

    accuracy_scores = []
    f1_scores_down = []
    f1_scores_flat = []
    f1_scores_up = []

    precision_scores_down = []
    precision_scores_flat = []
    precision_scores_up = []

    recall_scores_down = []
    recall_scores_flat = []
    recall_scores_up = []
    for train_index, test_index in tscv.split(dataset):
        print("--------------------------------------------------------------------------")
        # print("TRAIN:", train_index, "\n TEST:", test_index)

        train_dataset, test_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
        x_train, x_test = train_dataset[predictors], test_dataset[predictors]
        y_train, y_test = train_dataset[['trend']], test_dataset[['trend']]

        print("Class label for training set : ", Counter(y_train['trend']))
        print("Class label for test set : ", Counter(y_test['trend']))

        preprocessor = create_preprocessor(predictors, None)

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
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)

        try:
            f1_scores_down.append(f1[0])
            precision_scores_down.append(precision[0])
            recall_scores_down.append(recall[0])
        except Exception as e:
            print("Exception: " + str(e))
        try:
            f1_scores_flat.append(f1[1])
            precision_scores_flat.append(precision[1])
            recall_scores_flat.append(recall[1])
        except Exception as e:
            print("Exception: " + str(e))
        try:
            f1_scores_up.append(f1[2])
            precision_scores_up.append(precision[2])
            recall_scores_up.append(recall[2])
        except Exception as e:
            print("Exception: " + str(e))

    print("ACCURACY OVERALL MEAN: " + str(np.mean(accuracy_scores)) + " FOR CLASSIFIER: " + str(classifier))
    print("F1 OVERALL MEAN FOR LABEL -1: " + str(np.mean(f1_scores_down)) + " / FOR LABEL 0: " + str(
        np.mean(f1_scores_flat)) + " / FOR LABEL 1: " + str(
        np.mean(f1_scores_up)) + " FOR CLASSIFIER: " + str(classifier))
    return np.mean(accuracy_scores), np.mean(f1_scores_down), np.mean(f1_scores_flat), np.mean(f1_scores_up),   \
           np.mean(precision_scores_down), np.mean(precision_scores_flat), np.mean(precision_scores_up),        \
           np.mean(recall_scores_down),  np.mean(recall_scores_flat), np.mean(recall_scores_up),                \
           accuracy_scores, f1_scores_down, f1_scores_flat, f1_scores_up


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
                  'ema_long'
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

    # TRAINING WITH THE FINAL MODEL

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

    # Train the model
    pipe.fit(x_train, y_train.values.ravel())

    # Use model to make predictions
    y_pred = pipe.predict(x_test)

    # Evaluate the performance
    print("\nTraining ", str(classifiers[BEST_CLASSIFIER]))
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
    if EXPORT_CLASSIFIER:
        filename = 'model/' + CRYPTO_CURRENCY + '_classifier.pkl'
        final_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor)
            , ('classifier', classifiers[BEST_CLASSIFIER])
        ]).fit(x_final_train, y_final_train.values.ravel())
        joblib.dump(final_pipe, filename)

    ranking_attributes_contribution(dataset)
