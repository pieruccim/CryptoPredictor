import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)

CURRENCIES = ["BITCOIN", "ETHEREUM", "RIPPLE", "LITECOIN", "BINANCE", "MONERO", "DASH", "ZCASH", "SOLANA"]

# 2. Create a Dash app instance
app = Dash(external_stylesheets=[dbc.themes.DARKLY])

# 3. Add the cards to the app's layout
'''
card_content = [
    dbc.CardHeader("Card header"),
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]
'''

card_list = []
for currency in CURRENCIES:
    card = dbc.Card(
        [
            dbc.CardImg(src="assets/"+ str(currency) +".png", top=True),
            dbc.CardBody(
                [
                    html.H4(currency, className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", color="primary"),
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

# Then we incorporate the snippet into our layout.
# This example keeps it simple and just wraps it in a Container
app.layout = dbc.Container(cards, fluid=True)

# 5. Start the Dash server
if __name__ == "__main__":
    app.run_server()
