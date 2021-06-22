from .main import main, path_arg

from collections import namedtuple, OrderedDict
#
import climax as clx
from ploteries3.figure_handlers import FigureHandler
from ploteries3.data_store import DataStore
from ploteries2._ploteries2_helper import get_train_args
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
global DB_PATH, READER, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH, APP, SLIDER_GROUP, REGISTRY
HEIGHT, WIDTH = None, None
REGISTRY = set()
global GRAPH_KWARGS
GRAPH_KWARGS = {}  # {'config': {'displayModeBar': True}}
global FIGURE_LAYOUT


#####
_wd = 550


def update_figure_layout():
    global FIGURE_LAYOUT
    FIGURE_LAYOUT = dict(
        height=HEIGHT, width=WIDTH,
        margin=go.layout.Margin(**dict(zip('lrbt', [0, 30, 0, 0]), pad=4)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1),
        modebar=dict(
            orientation='v'))


update_figure_layout()

#
APP = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Helper functions.

CONTROL_WIDGET_STYLE = {'float': 'left', 'margin': '0em 1em 0em 1em'}


# Layout creation
PosnTuple = namedtuple('PosnTuple', ['tab', 'group', 'abs_name', 'rel_name'])


def get_fig_handler_posn(handler, default='Others'):
    """
    Specifies where figures will be positions in the layout.
    """

    fig_name = handler.name

    hierarchy = fig_name.split('/')
    if len(hierarchy) == 1:
        tab, group, rel_name = (default, default, fig_name)
    elif len(hierarchy) == 2:
        tab, group, rel_name = (hierarchy[0], default, hierarchy[1])
    elif len(hierarchy) > 2:
        tab, group, rel_name = hierarchy[0], hierarchy[1],  '/'.join(hierarchy[2:])
    else:
        raise Exception('Unexpected case!')
    return PosnTuple(tab, group, abs_name=fig_name, rel_name=rel_name)


def create_layout(update_interval):
    global APP, DATA_STORE

    # Get figure handlers, posns, tabs, groups.
    fig_handlers = DATA_STORE.get_figure_handlers()
    print(fig_handlers)
    fig_posns = [get_fig_handler_posn(_fh) for _fh in fig_handlers]
    tabs = list(OrderedDict.fromkeys([_fig_posn.tab for _fig_posn in fig_posns]))
    groups = list(OrderedDict.fromkeys([_fig_posn.group for _fig_posn in fig_posns]))

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
                            [_fig_handler.build_html(empty=True)
                             for _fig_handler, _fig_posn in zip(fig_handlers, fig_posns)
                             if _fig_posn.group == group and _fig_posn.tab == tab])])
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
@clx.option('--height', help='Figure height,', type=int, default=_wd*(2/3))
@clx.option('--width', help='Figure width,', type=int, default=_wd)
def launch(path, debug, host, interval, height, width, port, workers):
    """
    Launch a ploteries visualization server.
    """
    #
    global DB_PATH, DATA_STORE, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH, SLIDER_GROUP
    DB_PATH = path
    HEIGHT = height
    WIDTH = width
    LARGE_HEIGHT = 2*HEIGHT
    LARGE_WIDTH = 2*WIDTH
    update_figure_layout()
    DATA_STORE = DataStore(DB_PATH, read_only=True)
    #
    lambda: FigureHandler.create_dash_callbacks(
        APP, DATA_STORE,
        Input('interval-component', 'n_intervals'),
        Input('global-index-dropdown', 'value'),
        Output("global-index-dropdown", "options"),
        REGISTRY
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
