import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback

CURRENCIES = ["BITCOIN", "ETHEREUM", "RIPPLE", "LITECOIN", "BINANCE", "MONERO", "DASH", "ZCASH", "SOLANA"]

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])

card_list = []
for currency in CURRENCIES:
    card = dbc.Card(
        [
            dbc.CardImg(src="assets/" + str(currency) + ".png", top=True),
            dbc.CardBody(
                [
                    html.H4(currency, className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(html.Button("PREDICTION"), href="/"+str(currency).lower(), refresh=True),
                    # dbc.Button("Go somewhere", color="primary"),
                ]
            ),
        ],
        style={"width": "auto"},
    )
    card_list.append(card)

row_1 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_list[0], color="primary", outline=True)),
        dbc.Col(dbc.Card(card_list[1], color="secondary", outline=True)),
        dbc.Col(dbc.Card(card_list[2], color="info", outline=True)),
    ],
    className="mb-4",
)

row_2 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_list[3], color="success", outline=True)),
        dbc.Col(dbc.Card(card_list[4], color="warning", outline=True)),
        dbc.Col(dbc.Card(card_list[5], color="danger", outline=True)),
    ],
    className="mb-4",
)

row_3 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_list[6], color="success", outline=True)),
        dbc.Col(dbc.Card(card_list[7], color="warning", outline=True)),
        dbc.Col(dbc.Card(card_list[8], color="danger", outline=True)),
    ],
    className="mb-4",
)

cards = html.Div([row_1, row_2, row_3])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

'''
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
'''
index_page = html.Div([
    row_1, row_2, row_3
], style={'margin':200})

bitcoin = html.Div([
    html.H1('bitcoin'),
    dcc.Dropdown(['2020', '2021', '2022'], 'LA', id='page-1-dropdown'),
    html.Div(id='page-1-content'),
    #html.Br(),
    #dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])


@callback(Output('page-1-content', 'children'),
          [Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return f'You have selected {value}'


binance = html.Div([
    html.H1('binance'),
    dcc.RadioItems(['Orange', 'Blue', 'Red'], 'Orange', id='page-2-radios'),
    #dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])


@callback(Output('page-2-content', 'children'),
          [Input('page-2-radios', 'value')])
def page_2_radios(value):
    return f'You have selected {value}'

dash = html.Div([
    html.H1('dash'),
    dcc.Link('Go back to home', href='/'),
    #html.Div(id='page-2-content'),
    html.Br()
])

ethereum = html.Div([
    html.H1('ethereum'),
    dcc.Link('Go back to home', href='/'),
    #html.Div(id='page-2-content'),
    html.Br()
])

litecoin = html.Div([
    html.H1('litecoin'),
    dcc.Link('Go back to home', href='/'),
    #html.Div(id='page-2-content'),
    html.Br()
])

monero = html.Div([
    html.H1('monero'),
    dcc.Link('Go back to home', href='/'),
    #html.Div(id='page-2-content'),
    html.Br()
])

# Update the index
@callback(Output('page-content', 'children'),
          [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/bitcoin':
        return bitcoin
    elif pathname == '/binance':
        return binance
    elif pathname == '/dash':
        return dash
    elif pathname == '/ethereum':
        return ethereum
    elif pathname == '/litecoin':
        return litecoin
    elif pathname == '/monero':
        return monero
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


# Then we incorporate the snippet into our layout.
# This example keeps it simple and just wraps it in a Container
# app.layout = dbc.Container(cards, fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
