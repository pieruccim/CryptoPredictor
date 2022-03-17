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

# get the collection data from mongo
collection = MongoConnector.get_collection("BTC")
document = collection.find({}, {'Date': 1, 'adj_close': 1}).sort("Date", pymongo.DESCENDING)
dff = pd.DataFrame.from_dict(document)

# drop the _id column
df = dff.iloc[:, 1:]

df = dff[2:].reset_index()

df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month

layout = html.Div([
    dcc.Link('Back to Crypto Selection', href='/'),
    html.Br(),
    html.Div([html.H1("Bitcoin Trend Prediction")], style={'textAlign': "center"}),
    html.Div([
        html.Div([
            html.Div([dcc.Graph(id="my-graph-bitcoin")], className="row", style={"margin": "auto"}),
            html.Div([html.Div(dcc.RangeSlider(id="year selection", updatemode='drag',
                                               marks={i: '{}'.format(i) for i in df.Year.unique().tolist()},
                                               min=df.Year.min(), max=df.Year.max(),
                                               value=[FROM_YEAR, TO_YEAR]),
                               className="row",
                               style={"padding-bottom": 30, "padding-top": 30, "width": "60%", "margin": "auto"}),
                      html.Span("Moving Average :Select Window Interval", className="row", style={"padding-top": 30}),
                      html.Div(dcc.Slider(id="select-range1", updatemode='drag',
                                          marks={i * 10: str(i * 10) for i in range(0, 21)},
                                          min=0, max=50, value=12), className="row", style={"padding": 10}),
                      html.Div(dcc.Slider(id="select-range2", updatemode='drag',
                                          marks={i * 10: str(i * 10) for i in range(0, 21)},
                                          min=0, max=100, value=50), className="row", style={"padding": 10})

                      ], className="row")
        ], className="twelve columns", style={"margin-right": 0, "padding": 0}),

    ], className="row"),

    # PREDICTION BUTTON
    html.Div(
        html.Button('PREDICT', id='my-button', className="button"),
        style={"margin-top": 150, "margin-bottom": 150, "display": "flex", "justifyContent": "center", "font-size": "2em"}
    ),
    html.Br(),
    html.Div(id='response-bitcoin', children='Click to predict trend',
             style={"margin-top": 10, "margin-bottom": 100, "display": "flex", "justifyContent": "center"})


], className="container", style={"width": "100%"})


@app.callback(Output('response-bitcoin', 'children'), [Input('my-button', 'n_clicks')])
def on_click(button_click):
    filename = '../model/BTC-USD_classifier.pkl'
    clf = joblib.load(filename)

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
    if(button_click==None):
        return None
    # if we press the prediction button, or if we have already pressed it, show class prediction
    else:
        if(result==-1):
            return html.Img(src="assets/down-trend.png")
        elif(result==0):
            return html.Img(src="assets/flat-trend.png")
        elif(result==1):
            return html.Img(src="assets/up-trend.png")


@app.callback(
    Output("my-graph-bitcoin", 'figure'),
    [Input("year selection", 'value'),
     Input("select-range1", 'value'),
     Input("select-range2", 'value')])
def update_figure(year, range1, range2):
    dff_apl = df[(df["Year"] >= year[0]) & (df["Year"] <= year[1])]

    ema_short = dff_apl['adj_close'].ewm(span=range1).mean()
    ema_long = dff_apl['adj_close'].ewm(span=range2).mean()

    trace1 = go.Scatter(x=dff_apl['Date'], y=dff_apl['adj_close'], mode='lines', name='Crypto')
    trace_a = go.Scatter(x=dff_apl['Date'], y=ema_short, mode='lines', yaxis='y', name=f'Window {range1}')
    trace_b = go.Scatter(x=dff_apl['Date'], y=ema_long, mode='lines', yaxis='y', name=f'Window {range2}')

    layout1 = go.Layout({'title': 'Crypto Trend With Moving Average',
                         "legend": {"orientation": "h", "xanchor": "right"},
                         "xaxis": {
                             "rangeselector": {
                                 "buttons": [
                                     {"count": 6, "label": "6M", "step": "month",
                                      "stepmode": "backward"},
                                     {"count": 1, "label": "1Y", "step": "year",
                                      "stepmode": "backward"},
                                     {"count": 1, "label": "YTD", "step": "year",
                                      "stepmode": "todate"},
                                     {"label": "5Y", "step": "all",
                                      "stepmode": "backward"}
                                 ]
                             }
                         }})

    figure = {'data': [trace1],
              'layout': layout1
              }
    figure['data'].append(trace_a)
    figure['data'].append(trace_b)
    return figure
