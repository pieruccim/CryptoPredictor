import statistics

import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from cross_validation import create_preprocessor
from utilities.utils import Utils

SHORT_WINDOW = int(Utils.load_config('SHORT_WINDOW'))
LONG_WINDOW = int(Utils.load_config('LONG_WINDOW'))


def plot_correlation_matrix(dataset):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.
        Input:
            df: pandas DataFrame
    """
    corrMatrix = dataset.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.title('Correlation matrix', fontsize=16)
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
    x = df[pred]
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
    #print(str(f1_down)+","+str(f1_flat)+","+str(f1_up))

    f1_avg = []
    for i in range(len(f1_up)):
        list_float = [f1_down[i], f1_flat[i], f1_up[i]]
        f1_avg.append(statistics.mean(list_float))

    df_f1score = pd.DataFrame(f1_avg, columns=['f_scores'])
    df_f1score.plot()

    plt.title(str(clf).split('(')[0] + ' f score in cross-validation', fontsize=16)
    plt.xlabel("Iteration")
    plt.ylabel("F-Score")
    plt.ylim([0, 1])
    plt.savefig('plots/plot-fscore-cross-validation/' + str(clf).split('(')[0] + '.png')
