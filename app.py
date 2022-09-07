import dash_bootstrap_components as dbc
import importlib
from dash import Dash
from dash import dcc, html, Input, Output, callback
import trend

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
# app.run_server()

# debug mode is False because otherwise the updater() runs continously and overwrites mongo data
app.title = 'Crypto Predictor'

CURRENCIES = ["BITCOIN", "ETHEREUM", "BINANCE"]

DESCRIPTIONS = {"BITCOIN": "Bitcoin is the world’s most traded cryptocurrency, representing a huge slice of the "
                           "crypto market pie.",
                "ETHEREUM": "Ethereum is a platform for "
                            "creating decentralized applications based on blockchain and smart contract technology.",
                "BINANCE": "Binance Coin is a cryptocurrency that can be used to trade and pay fees on the "
                           "Binance cryptocurrency exchange. "
                }

card_list = []
for currency in CURRENCIES:
    card = dbc.Card(
        [
            dbc.CardImg(src="assets/" + str(currency) + ".png", top=True, className="p-5"),
            dbc.CardBody(
                [
                    html.H4(currency, className="card-title text-center"),
                    html.P(
                        DESCRIPTIONS[str(currency)],
                        className="card-text",
                        style={"height": "12vh"}
                    ),
                    dcc.Link(html.Button("Prediction", className="button"),
                             href="/trend?currency=" + str(currency).lower(),
                             refresh=True, className="text-decoration-none")
                ]
            ),
        ],
        style={"width": "auto"}
    )
    card_list.append(card)

row_1 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_list[0], className="container", outline=True)),
        dbc.Col(dbc.Card(card_list[1], className="container", outline=True)),
        dbc.Col(dbc.Card(card_list[2], className="container", outline=True)),
    ],
    className="mb-4",
)

cards = html.Div([row_1])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1("Crypto Predictor", style={'text-align': "center", 'marginBottom': 50, 'font-size': 70}),
    row_1
], style={'margin': 80})


# Update the index
@callback(Output('page-content', 'children'),
          [Input('url', 'pathname')])
def display_page(pathname):
    if pathname.startswith("/trend"):
        return trend.layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here
