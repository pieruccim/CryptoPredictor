import matplotlib
import pandas
from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
import dash
from dash import html, dcc
import plotly.graph_objects as go


from persistence.MongoConnector import MongoConnector
from utilities.utils import Utils

# coll_name = Utils.load_config("COLLECTION_NAME")
# result = MongoConnector.get_collection(coll_name).find()
# print(result.next())

app = dash.Dash()

def plot_graph(dataset):
    plt1 = go.Figure()
    plt1.title('BTC-USD Adj Close Price', fontsize=16)
    plt1.xlabel("Date")
    plt1.ylabel("Adjusted Price")

    ax = dataset['adj_close'].plot(lw=3, figsize=(14, 7), label='Original observations')
    dataset['ema_short'].plot(ax=ax, lw=3, label='EMA (window ' + str(2) + ')')
    dataset['ema_long'].plot(ax=ax, lw=3, label='EMA (window ' + str(5) + ')')

    plt1.tick_params(labelsize=12)
    plt1.legend(loc='upper left', fontsize=12)

    plt1.plot(dataset.loc[dataset.trend == 1.0].index,
             dataset.adj_close[dataset.trend == 1.0],
             '^', markersize=8, color='g', label='up')

    plt1.plot(dataset.loc[dataset.trend == 0.0].index,
             dataset.adj_close[dataset.trend == 0.0],
             '^', markersize=8, color='y', label='flat')

    plt1.plot(dataset.loc[dataset.trend == -1.0].index,
             dataset.adj_close[dataset.trend == -1.0],
             '^', markersize=8, color='r', label='down')

    plotly_fig = mpl_to_plotly(plt1)
    graph = dcc.Graph(id='myGraph', fig=plotly_fig)
    return graph


if __name__ == '__main__':
    dataframe = pandas.read_csv("data/BTC_data.csv")

    graph = plot_graph(dataframe)

    app.layout = html.Div([
        dcc.Graph(id='matplotlib-graph', figure=graph)

    ])

    app.run_server(debug=True, port=8010, host='localhost')
