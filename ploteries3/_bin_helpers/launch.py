from .main import main, path_arg
from pglib.profiling import time_and_print

from collections import namedtuple, OrderedDict
#
import climax as clx
from ploteries3._cli_interface import PloteriesLaunchInterface
from ploteries3.data_store import DataStore
# from ploteries2._ploteries2_helper import get_train_args
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from multiprocessing import cpu_count
from pglib.gunicorn import GunicornServer

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#

global DATA_INTERFACE, DATA_STORE, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH, APP
HEIGHT, WIDTH = None, None
global GRAPH_KWARGS
GRAPH_KWARGS = {}  # {'config': {'displayModeBar': True}}
global FIGURE_LAYOUT


DEFAULT_WIDTH = 550
DEFAULT_HEIGHT_TO_WIDTH = 2/3
CONTROL_WIDGET_STYLE = {'float': 'left', 'margin': '0em 1em 0em 1em'}

#
# suppress_callback_exceptions=True)
APP = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

# Layout creation


def create_layout(update_interval):
    global DATA_INTERFACE

    # Get figure handlers, posns, tabs, groups.
    empty_figs = DATA_INTERFACE.render_empty_figures()
    # Ordered, unique tabs
    tabs = list(OrderedDict.fromkeys([_fig.posn.tab for _fig in empty_figs]))
    # Ordered, unique groups
    groups = list(OrderedDict.fromkeys([_fig.posn.group for _fig in empty_figs]))

    # Layout
    layout = html.Div(
        children=[
            html.Div([
                html.H2(children='Ploteries', style={
                        'float': 'right', 'margin-right': '1em', 'font-family': 'Courier New'}),
                # html.Div([
                html.Div([
                    daq.ToggleSwitch(id='auto-update-switch', size=20,
                                     value=True,
                                     label=' ',
                                     style=CONTROL_WIDGET_STYLE),  # 'display': 'inline-block'
                    html.Label(['Global step:', dcc.Dropdown(
                        id='global-index-dropdown')], style=dict(**{'min-width': '20em'}, **CONTROL_WIDGET_STYLE))], )
            ], style={'content': "", 'clear': 'both', 'display': 'table', 'width': '100%'}),
            dcc.Tabs([
                dcc.Tab(
                    label=tab,
                    children=[html.Details([
                        html.Summary(group),
                        html.Div(
                            [_fig.html
                             for _fig in empty_figs
                             if _fig.posn.group == group and _fig.posn.tab == tab])], open=True)
                        for group in groups])
                for tab in tabs]),
            dcc.Interval(
                id='interval-component',
                interval=update_interval*1000,  # in milliseconds
                n_intervals=0
            )]
    )
    return layout


def create_toolbar_callbacks():
    # Auto update switch
    def auto_update_switch(enabled):
        enabled = bool(enabled)
        return not enabled, 'Auto update '+['disabled', 'enabled'][enabled]
    _g = globals()
    _g['auto_update_switch'] = APP.callback(
        [Output('interval-component', 'disabled'),
         Output('auto-update-switch', 'label')],
        [Input('auto-update-switch', 'value')]
    )(auto_update_switch)


@main.command(parents=[path_arg])
@clx.option('--interval', type=float, help='Seconds between automatic update of all plots.', default=300)
@clx.option('--debug', action='store_true', help='Enables javascript debug console', default=False)
@clx.option('--host', help='Host name.', default='0.0.0.0')
@clx.option('--port', help='Port number.', default='8000')
@clx.option('--workers', help='Number of workers (ignored in debug mode).', default=(cpu_count() * 2) + 1)
@clx.option(
    '--height', help=f'Figure height (default={DEFAULT_WIDTH*DEFAULT_HEIGHT_TO_WIDTH})', type=int,
    default=DEFAULT_WIDTH * DEFAULT_HEIGHT_TO_WIDTH)
@clx.option(
    '--width', help=f'Figure width (default={DEFAULT_WIDTH}),', type=int,
    default=DEFAULT_WIDTH)
def launch(path, debug, host, interval, height, width, port, workers):
    """
    Launch a ploteries visualization server.
    """
    #
    global DATA_INTERFACE, DATA_STORE, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH
    HEIGHT = height
    WIDTH = width
    LARGE_HEIGHT = 2*HEIGHT
    LARGE_WIDTH = 2*WIDTH

    DATA_STORE = DataStore(path, read_only=True)
    DATA_INTERFACE = PloteriesLaunchInterface(
        DATA_STORE,
        figure_layout_kwargs={
            'height': HEIGHT,
            'width': WIDTH,
            'margin': go.layout.Margin(**dict(zip('lrbt', [0, 30, 0, 0]), pad=4)),
            'legend': {
                'orientation': "h",
                'yanchor': "bottom",
                'y': 1.02,
                'xanchor': "right",
                'x': 1},
            'modebar': {
                'orientation': 'v'}})

    #
    DATA_INTERFACE.create_callbacks(
        APP,
        Input('interval-component', 'n_intervals'),
        Input('global-index-dropdown', 'value'),
        Output("global-index-dropdown", "options"),
    )

    APP.layout = lambda: create_layout(update_interval=interval)

    if debug:
        APP.run_server(debug=debug, host=host, port=port)
    else:
        options = {
            'bind': '%s:%s' % (host, port),
            'workers': workers,
            'timeout': 180,
        }
        GunicornServer(APP.server, options).run()
