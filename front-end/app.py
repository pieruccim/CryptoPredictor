import dash_bootstrap_components as dbc
from dash import Dash

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])
