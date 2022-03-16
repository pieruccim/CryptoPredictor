from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from app import app
from persistence.MongoConnector import MongoConnector

# get the collection data from mongo
collection = MongoConnector.get_collection("BNB")
document = collection.find({}, {'Date': 1, 'adj_close': 1})
dff = pd.DataFrame.from_dict(document)

# drop the _id column
df = dff.iloc[:, 1:]

df = dff[2:].reset_index()

df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month

'''
calendar=[]
for y in df.Year.unique().tolist():
    for m in df.Month.unique().tolist():
        calendar.append(str(y) +"-"+ str("{:02d}".format(m)))
'''

layout = html.Div([
    html.Div([html.H1("Technical Analysis : Moving Average and Returns ")], style={'textAlign': "center"}),
    html.Div([
        html.Div([
            html.Div([dcc.Graph(id="my-graph")], className="row", style={"margin": "auto"}),
            html.Div([html.Div(dcc.RangeSlider(id="year selection", updatemode='drag',
                                               marks={i: '{}'.format(i) for i in df.Year.unique().tolist()},
                                               min=df.Year.min(), max=df.Year.max(),
                                               value=[2020, 2022]),
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

    ], className="row")
], className="container", style={"width": "100%"})


@app.callback(
    Output("my-graph", 'figure'),
    [Input("year selection", 'value'),
     Input("select-range1", 'value'),
     Input("select-range2", 'value')])
def update_figure(year, range1, range2):
    dff_apl = df[(df["Year"] >= year[0]) & (df["Year"] <= year[1])]

    ema_short = dff_apl['adj_close'].ewm(span=range1).mean()
    ema_long = dff_apl['adj_close'].ewm(span=range2).mean()

    trace1 = go.Scatter(x=dff_apl['Date'], y=dff_apl['adj_close'],
                        mode='lines', name='Crypto')
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
                             }}})

    figure = {'data': [trace1],
              'layout': layout1
              }
    figure['data'].append(trace_a)
    figure['data'].append(trace_b)
    return figure
