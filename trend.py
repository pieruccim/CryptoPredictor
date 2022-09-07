import urllib

import joblib
import pymongo
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from app import app
from persistence.MongoConnector import MongoConnector

FROM_YEAR = 2020
TO_YEAR = 2022

layout = html.Div([
    dcc.Location(id='url-trend', refresh=True),
    html.Div(id='page-content-trend')
])


@app.callback(Output('page-content-trend', 'children'),
              Input('url-trend', 'search'))
def display_page(params):
    # e.g. params = '?firstname=John&lastname=Smith&birthyear=1990'
    parsed = urllib.parse.urlparse(params)
    parsed_dict = urllib.parse.parse_qs(parsed.query)
    # e.g. parsed_dict = {'firstname': ['John'], 'lastname': ['Smith'], 'birthyear': ['1990']}

    COLLECTION_NAME = ""
    CURRENCY = ""
    try:
        if "bitcoin" in parsed_dict["currency"]:
            COLLECTION_NAME = "BTC"
            CURRENCY = "Bitcoin"
        elif "ethereum" in parsed_dict["currency"]:
            COLLECTION_NAME = "ETH"
            CURRENCY = "Ethereum"
        elif "binance" in parsed_dict["currency"]:
            COLLECTION_NAME = "BNB"
            CURRENCY = "Binance"
    except:
        COLLECTION_NAME = "BTC"
        CURRENCY = "Bitcoin"

    # get the collection data from mongo
    collection = MongoConnector.get_collection(COLLECTION_NAME)
    document = collection.find({}, {'Date': 1, 'adj_close': 1}).sort("Date", pymongo.DESCENDING)
    dff = pd.DataFrame.from_dict(document)
    # drop the _id column
    df = dff.iloc[:, 1:]
    df['Year'] = pd.DatetimeIndex(df['Date']).year

    layout = html.Div([
        html.Div([
            dcc.Link('Back', href='/', className='button col-1',
                     style={'textAlign': "center", "margin-top": "auto", "margin-bottom": "auto", "font-size": 16}),
            html.Div([html.H1(CURRENCY + " trend prediction")], style={'textAlign': "center", }, className='col-11')
            ], className="row", style={"margin-top": 20}
        ),
        html.Div([
            html.Div([
                html.Div([dcc.Graph(id="my-graph-bitcoin")], className="row",
                         style={"margin": "auto", "margin-top": 20}),
                html.Div([
                    html.Div([html.Div(dcc.RangeSlider(id="year selection", updatemode='drag',
                                                       marks={i: '{}'.format(i) for i in df.Year.unique().tolist()},
                                                       min=df.Year.min(), max=df.Year.max(),
                                                       value=[FROM_YEAR, TO_YEAR]),
                                       className="row",
                                       style={"padding-bottom": 30, "padding-top": 30, "width": "60%",
                                              "margin": "auto"}),
                              html.Span("Moving Average selection", className="row",
                                        style={"padding-top": 30}),
                              html.Div(dcc.Slider(id="select-range1", updatemode='drag',
                                                  marks={i * 10: str(i * 10) for i in range(0, 21)},
                                                  min=0, max=50, value=12), className="row", style={"padding": 10}),
                              html.Div(dcc.Slider(id="select-range2", updatemode='drag',
                                                  marks={i * 10: str(i * 10) for i in range(0, 21)},
                                                  min=0, max=100, value=50), className="row", style={"padding": 10})

                              ], className="col"),

                    # Prediction Button
                    html.Div(
                        html.Button('PREDICT', id='my-button', className="button"),
                        style={"margin-top": "auto", "margin-bottom": "auto", "display": "flex",
                               "justifyContent": "center",
                               "font-size": "2em"},
                        className='col-3'
                    ),

                    # Result of the prediction
                    html.Div(id='response-bitcoin', children='Click to predict trend',
                             style={"display": "flex", "justifyContent": "center",
                                    "margin-top": "auto", "margin-bottom": "auto"},
                             className='col-3'
                             )
                ], className='row')
            ], className="twelve columns", style={"margin-right": 0, "padding": 0}),

        ], className="row"),

    ], className="container", style={"width": "100%"})

    return layout


@app.callback(Output('response-bitcoin', 'children'), [Input('my-button', 'n_clicks'), Input('url-trend', 'search')])
def on_click(button_click, params):
    parsed = urllib.parse.urlparse(params)
    parsed_dict = urllib.parse.parse_qs(parsed.query)

    COLLECTION_NAME = ""

    try:
        if "bitcoin" in parsed_dict["currency"]:
            COLLECTION_NAME = "BTC"
            FILENAME = 'model/BTC-USD_classifier.pkl'

        elif "ethereum" in parsed_dict["currency"]:
            COLLECTION_NAME = "ETH"
            FILENAME = 'model/ETH-USD_classifier.pkl'

        elif "binance" in parsed_dict["currency"]:
            COLLECTION_NAME = "BNB"
            FILENAME = 'model/BNB-USD_classifier.pkl'
    except:
        COLLECTION_NAME = "BTC"
        FILENAME = 'model/BTC-USD_classifier.pkl'

    # get the collection data from mongo
    collection = MongoConnector.get_collection(COLLECTION_NAME)

    # load classifier
    clf = joblib.load(FILENAME)

    predictors = ['open',
                  'high',
                  'low',
                  'adj_close',
                  'ema_short',
                  'ema_long',
                  'volume'
                  ]

    # query to mongoDB for getting the tuple of today and convert it from dict to dataframe to better elaborate it
    last_tuple = pd.DataFrame(list(collection.find().limit(1).sort("Date", pymongo.DESCENDING)))
    # project only the predictors attributes
    tuple_to_predict = last_tuple[predictors]
    # use classifier to predict tuple class [-1, 0, 1] and send it back
    result = clf.predict(tuple_to_predict)

    # in case the button has never been pressed yet
    if (button_click == None):
        return None
    # if we press the prediction button, or if we have already pressed it, show class prediction
    else:
        if (result == -1):
            return html.Img(src="assets/down-trend.png", style={"height": 150, "width": 150})
        elif (result == 0):
            return html.Img(src="assets/flat-trend.png", style={"height": 150, "width": 150})
        elif (result == 1):
            return html.Img(src="assets/up-trend.png", style={"height": 150, "width": 150})


@app.callback(
    Output("my-graph-bitcoin", 'figure'),
    [Input("year selection", 'value'),
     Input("select-range1", 'value'),
     Input("select-range2", 'value')],
    Input('url-trend', 'search'))
def update_figure(year, range1, range2, params):
    parsed = urllib.parse.urlparse(params)
    parsed_dict = urllib.parse.parse_qs(parsed.query)

    COLLECTION_NAME = ""

    try:
        if "bitcoin" in parsed_dict["currency"]:
            COLLECTION_NAME = "BTC"

        elif "ethereum" in parsed_dict["currency"]:
            COLLECTION_NAME = "ETH"

        elif "binance" in parsed_dict["currency"]:
            COLLECTION_NAME = "BNB"
    except:
        COLLECTION_NAME = "BTC"

    # get the collection data from mongo
    collection = MongoConnector.get_collection(COLLECTION_NAME)
    document = collection.find({}, {'Date': 1, 'adj_close': 1}).sort("Date", pymongo.ASCENDING)
    dff = pd.DataFrame.from_dict(document)
    # drop the _id column
    df = dff.iloc[:, 1:]
    df['Year'] = pd.DatetimeIndex(df['Date']).year

    dff_apl = df[(df["Year"] >= year[0]) & (df["Year"] <= year[1])]

    ema_short = dff_apl['adj_close'].ewm(span=range1).mean()
    ema_long = dff_apl['adj_close'].ewm(span=range2).mean()

    trace1 = go.Scatter(x=dff_apl['Date'], y=dff_apl['adj_close'], mode='lines', name='Crypto')
    trace_a = go.Scatter(x=dff_apl['Date'], y=ema_short, mode='lines', yaxis='y', name=f'Window {range1}')
    trace_b = go.Scatter(x=dff_apl['Date'], y=ema_long, mode='lines', yaxis='y', name=f'Window {range2}')

    layout1 = go.Layout({'title': 'Crypto Trend With Moving Average',
                         "legend": {"orientation": "h", "xanchor": "left"},
                         "xaxis": {
                             "rangeselector": {
                                 "buttons": [
                                     {"count": 6, "label": "6M", "step": "month",
                                      "stepmode": "backward"},
                                     {"count": 1, "label": "1Y", "step": "year",
                                      "stepmode": "backward"},
                                     {"count": 1, "label": "YTD", "step": "year",
                                      "stepmode": "todate"},
                                     {"label": "2Y", "step": "all",
                                      "stepmode": "backward"}
                                 ]
                             }
                         },
                         "font": {"family": "Arial"},
                         "height": 500
                         })

    figure = {'data': [trace1],
              'layout': layout1
              }
    figure['data'].append(trace_a)
    figure['data'].append(trace_b)
    return figure
