import dash_bootstrap_components as dbc
import bitcoin, ethereum, binance
from dash import dcc, html, Input, Output, callback
from app import app

CURRENCIES = ["BITCOIN", "ETHEREUM", "BINANCE"]

card_list = []
for currency in CURRENCIES:
    card = dbc.Card(
        [
            dbc.CardImg(src="assets/" + str(currency) + ".png", top=True),
            dbc.CardBody(
                [
                    html.H4(currency, className="card-title text-center"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(html.Button("Prediction", className="button"),
                             href="/" + str(currency).lower(),
                             refresh=True, className="text-decoration-none")
                ]
            ),
        ],
        style={"width": "auto"},
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
    if pathname == '/bitcoin':
        return bitcoin.layout
    elif pathname == '/ethereum':
        return ethereum.layout
    elif pathname == '/binance':
        return binance.layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True)
