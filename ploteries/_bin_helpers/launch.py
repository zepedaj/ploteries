from .main import main, path_arg
import glob
from pglib.profiling import time_and_print
#
import climax as clx
from ploteries.cli_interface import PloteriesLaunchInterface, FigureHandlerHook, TableHandlerHook
from ploteries.data_store import DataStore
# from ploteries2._ploteries2_helper import get_train_args
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from multiprocessing import cpu_count
from pglib.gunicorn import GunicornServer
import logging
from collections import OrderedDict

LOGGER = logging.getLogger(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#

global DATA_INTERFACES, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH, APP
HEIGHT, WIDTH = None, None
global GRAPH_KWARGS
GRAPH_KWARGS = {}  # {'config': {'displayModeBar': True}}
global FIGURE_LAYOUT


DEFAULT_WIDTH = 550
DEFAULT_HEIGHT_TO_WIDTH = 2/3
CONTROL_WIDGET_STYLE = {'float': 'left', 'margin': '0em 1em 0em 1em'}

#
# suppress_callback_exceptions=True)
APP = dash.Dash(__name__,  external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

# Layout creation


def create_layout(update_interval):
    global DATA_INTERFACES

    # Layout
    layout = html.Div(
        children=[
            html.Div([
                html.H2(children='Ploteries', style={
                        'float': 'right', 'margin-right': '1em', 'font-family': 'Courier New'}),
                # html.Div([
                html.Div([
                    html.Label(
                        ['Data store:',
                         dcc.Dropdown(
                             id='data-store-dropdown',
                             persistence=True,
                             options=[{'label': _x, 'value': _x}
                                      for _x in DATA_INTERFACES.keys()],
                             value=(next(iter(DATA_INTERFACES.keys())) if DATA_INTERFACES else None))],
                        style=dict(**{'min-width': '40em'}, **CONTROL_WIDGET_STYLE)),
                    html.Label(
                        ['Global step:',
                         html.Span(
                             id='global-index-dropdown-container',
                             children=dcc.Dropdown(id='global-index-dropdown'))],
                        style=dict(**{'min-width': '20em'}, **CONTROL_WIDGET_STYLE)),
                    daq.ToggleSwitch(id='auto-update-switch', size=20,
                                     value=True,
                                     persistence=True,
                                     label=' ',
                                     style=CONTROL_WIDGET_STYLE),  # 'display': 'inline-block'

                ], )
            ], style={
                'content': "",
                'clear': 'both',
                'display': 'table',
                'width': '100%',
                'position': 'sticky',
                'top': 0,
                'backgroundColor': 'white',
                'padding': '0.5em',
                'marginBottom': '0.5em',
                'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                'zIndex': 100000}),
            dcc.Tabs(id='figure-tabs', persistence=True),
            dcc.Interval(
                id='interval-component',
                interval=update_interval*1000,  # in milliseconds
                n_intervals=0
            )]
    )
    return layout


@APP.callback(
    Output('global-index-dropdown-container', 'children'),
    Input('data-store-dropdown', 'value')
)
def create_global_index_dropdown_with_persistence(data_store_name):
    if data_store_name is None:
        raise PreventUpdate
    return dcc.Dropdown(
        id='global-index-dropdown',
        persistence=data_store_name)


@APP.callback(
    Output('figure-tabs', 'children'),
    Input('data-store-dropdown', 'value'),
)
@time_and_print()
def update_figure_tabs(data_store_name):

    if data_store_name is None:
        raise PreventUpdate

    # Get figure handlers, posns, tabs, groups.
    empty_figs = DATA_INTERFACES[data_store_name].render_empty_figures()
    # Ordered, unique tabs
    tabs = list(OrderedDict.fromkeys([_fig.posn.tab for _fig in empty_figs]))
    # Ordered, unique groups
    groups = {_tab: list(
        OrderedDict.fromkeys([
            _fig.posn.group for _fig in empty_figs if _fig.posn.tab == _tab]))
        for _tab in tabs}

    out_tabs = [
        dcc.Tab(
            label=tab,
            children=[html.Details([
                html.Summary(group),
                html.Div(
                    [_fig.html
                     for _fig in empty_figs
                     if _fig.posn.group == group and _fig.posn.tab == tab])], open=True)
                for group in groups[tab]])
        for tab in tabs]

    return out_tabs


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


@main.command()
@clx.option('glob_path',
            help='Data store path. Will be interpreted with a call to recursive glob (see https://docs.python.org/3/library/glob.html).')
@clx.option('--interval',
            help='Seconds between automatic update of all plots.',
            type=float,
            default=300)
@clx.option('--debug', action='store_true',
            help='Enables javascript debug console', default=False)
@clx.option('--host',
            help='Host name.', default='0.0.0.0')
@clx.option('--port',
            help='Port number.', default='8000')
@clx.option('--workers',
            help='Number of workers (ignored in debug mode).', default=(cpu_count() * 2) + 1)
@clx.option(
    '--height', help=f'Figure height (default={DEFAULT_WIDTH*DEFAULT_HEIGHT_TO_WIDTH})', type=int,
    default=DEFAULT_WIDTH * DEFAULT_HEIGHT_TO_WIDTH)
@clx.option(
    '--width', help=f'Figure width (default={DEFAULT_WIDTH}),', type=int,
    default=DEFAULT_WIDTH)
def launch(glob_path, debug, host, interval, height, width, port, workers):
    """
    Launch a ploteries visualization server.
    """
    #
    global DATA_INTERFACES, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH
    HEIGHT = height
    WIDTH = width
    LARGE_HEIGHT = 2*HEIGHT
    LARGE_WIDTH = 2*WIDTH

    DATA_INTERFACES = OrderedDict()
    for _path in sorted(glob.glob(glob_path, recursive=True)):
        try:
            data_store = DataStore(_path, read_only=True)
            print(f'Loaded {_path}')
        except Exception:
            LOGGER.error('Error loading {_path}.')
        else:
            DATA_INTERFACES[_path] = PloteriesLaunchInterface(
                data_store,
                hooks=[
                    FigureHandlerHook(
                        data_store,
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
                                'orientation': 'v'}})])

    # Create callbacks
    for data_interface_type in set(type(_x) for _x in DATA_INTERFACES.values()):
        data_interface_type.create_callbacks(
            APP,
            lambda data_store_name: DATA_INTERFACES[data_store_name],
            callback_args=dict(
                interface_name_state=State('data-store-dropdown', 'value'),
                n_interval_input=Input('interval-component', 'n_intervals'),
                global_index_input_value=Input('global-index-dropdown', 'value'),
                global_index_dropdown_options=Output("global-index-dropdown", "options"),)
        )
    create_toolbar_callbacks()

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
