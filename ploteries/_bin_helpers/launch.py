from .main import main, path_arg
import os.path as osp
from threading import Lock
from dash.dash import no_update
import glob
from jztools.profiling import time_and_print

#
import climax as clx
from ploteries.cli_interface import (
    PloteriesLaunchInterface,
    FigureHandlerHook,
    TableHandlerHook,
)
from ploteries.data_store import DataStore

# from ploteries2._ploteries2_helper import get_train_args
import dash
from dash import dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from multiprocessing import cpu_count
from jztools.gunicorn import GunicornServer
import logging
from collections import OrderedDict

LOGGER = logging.getLogger(__name__)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
#

global DATA_INTERFACES, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH, APP
HEIGHT, WIDTH = None, None
global GRAPH_KWARGS
GRAPH_KWARGS = {}  # {'config': {'displayModeBar': True}}
global FIGURE_LAYOUT

DEFAULT_WIDTH = 550
DEFAULT_HEIGHT_TO_WIDTH = 2 / 3
CONTROL_WIDGET_STYLE = {"float": "left", "margin": "0em 1em 0em 1em"}

#
# suppress_callback_exceptions=True)
APP = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
app_callback = APP.callback


# Search all paths for ploteries databases and open them.
class DataInterfaces:
    def __init__(self, glob_path):
        self.lock = Lock()
        self.orig_glob_path = glob_path
        self.glob_path = (
            osp.join(glob_path, "**/*.pltr") if osp.isdir(glob_path) else glob_path
        )
        self.data_interfaces = OrderedDict()
        self.failed_paths = set()

        self.keys = self.data_interfaces.keys
        self.values = self.data_interfaces.values

    def options(self):
        return [{"label": _x, "value": _x} for _x in DATA_INTERFACES.keys()]

    def default_value(self):
        try:
            return next(iter(self.keys()))
        except StopIteration:
            return None

    def __getitem__(self, idx):
        try:
            return self.data_interfaces[idx]
        except KeyError:
            # The update might have happened in a different process,
            # attempt updating in the current process.
            self.update(verbose=False)
            return self.data_interfaces[idx]

    def __bool__(self):
        return bool(self.data_interfaces)

    def update(self, verbose=True):
        """
        Add any new paths that were created.
        """
        with self.lock:
            for _path in sorted(glob.glob(self.glob_path, recursive=True)):
                if _path not in self.data_interfaces and _path not in self.failed_paths:
                    try:
                        data_store = DataStore(_path, read_only=True)
                    except Exception:
                        self.failed_paths.add(_path)
                        if verbose:
                            LOGGER.error("Error loading {_path}.")
                    else:
                        if verbose:
                            print(f"Loaded {_path}")
                        self.data_interfaces[_path] = PloteriesLaunchInterface(
                            data_store,
                            hooks=[
                                FigureHandlerHook(
                                    data_store,
                                    figure_layout_kwargs={
                                        "height": HEIGHT,
                                        "width": WIDTH,
                                        "margin": go.layout.Margin(
                                            **dict(zip("lrbt", [0, 30, 0, 0]), pad=4)
                                        ),
                                        "legend": {
                                            "orientation": "h",
                                            "yanchor": "bottom",
                                            "y": 1.02,
                                            "xanchor": "right",
                                            "x": 1,
                                        },
                                        "modebar": {"orientation": "v"},
                                    },
                                ),
                                TableHandlerHook(data_store),
                            ],
                        )


# Layout creation


def create_layout(update_interval):
    global DATA_INTERFACES

    # Layout
    layout = html.Div(
        children=[
            html.Div(
                [
                    html.H2(
                        children=[
                            "Ploteries",
                            html.Img(
                                src="assets/ploteries2.png",
                                style={
                                    "height": "1.5em",
                                    "margin-bottom": "-0.4em",
                                    "margin-top": 0,
                                },
                            ),
                        ],
                        style={
                            "float": "right",
                            "margin-right": "1em",
                            "font-family": "Courier New",
                        },
                    ),
                    # html.Div([
                    html.Div(
                        [
                            html.Label(
                                [
                                    "Data store:",
                                    dcc.Dropdown(
                                        id="data-store-dropdown",
                                        persistence=osp.abspath(
                                            DATA_INTERFACES.glob_path
                                        ),
                                        persistence_type="session",
                                        clearable=False,
                                        options=DATA_INTERFACES.options(),
                                        value=DATA_INTERFACES.default_value(),
                                    ),
                                ],
                                style=dict(
                                    **{"min-width": "40em"}, **CONTROL_WIDGET_STYLE
                                ),
                            ),
                            html.Label(
                                [
                                    "Global step:",
                                    html.Span(
                                        id="global-index-dropdown-container",
                                        children=dcc.Dropdown(
                                            id="global-index-dropdown"
                                        ),
                                    ),
                                ],
                                style=dict(
                                    **{"min-width": "20em"}, **CONTROL_WIDGET_STYLE
                                ),
                            ),
                            daq.ToggleSwitch(
                                id="auto-update-switch",
                                size=20,
                                value=True,
                                persistence=True,
                                label=" ",
                                style=CONTROL_WIDGET_STYLE,
                            ),  # 'display': 'inline-block'
                        ],
                    ),
                ],
                style={
                    "content": "",
                    "clear": "both",
                    "display": "table",
                    "width": "100%",
                    "position": "sticky",
                    "top": 0,
                    "backgroundColor": "white",
                    "padding": "0.5em",
                    "marginBottom": "0.5em",
                    "boxShadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
                    "zIndex": 100000,
                },
            ),
            dcc.Tabs(id="figure-tabs", persistence=True),
            dcc.Interval(
                id="interval-component",
                interval=update_interval * 1000,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )
    return layout


@app_callback(
    Output("global-index-dropdown-container", "children"),
    Input("data-store-dropdown", "value"),
)
def create_global_index_dropdown_with_persistence(data_store_name):
    if data_store_name is None:
        raise PreventUpdate
    return dcc.Dropdown(
        id="global-index-dropdown", persistence=osp.abspath(data_store_name)
    )


def create_discover_callback():
    @app_callback(
        Output("data-store-dropdown", "options"),
        Input("interval-component", "n_intervals"),
        State("data-store-dropdown", "options"),
    )
    def update_data_store_dropdown_options(n_intervals, data_store_options):
        DATA_INTERFACES.update(verbose=False)
        new_paths = DATA_INTERFACES.keys()

        if set(new_paths) == set(_x["value"] for _x in data_store_options):
            options = no_update
        else:
            options = DATA_INTERFACES.options()

        return options


@app_callback(Output("figure-tabs", "children"), Input("data-store-dropdown", "value"))
def update_figure_tabs(data_store_name):
    if data_store_name is None:
        raise PreventUpdate

    # Get figure handlers, posns, tabs, groups.
    empty_figs = DATA_INTERFACES[data_store_name].render_empty_figures()
    # Ordered, unique tabs
    tabs = list(OrderedDict.fromkeys([_fig.posn.tab for _fig in empty_figs]))
    # Ordered, unique groups
    groups = {
        _tab: list(
            OrderedDict.fromkeys(
                [_fig.posn.group for _fig in empty_figs if _fig.posn.tab == _tab]
            )
        )
        for _tab in tabs
    }

    out_tabs = [
        dcc.Tab(
            label=tab,
            children=[
                html.Details(
                    [
                        html.Summary(group),
                        html.Div(
                            [
                                _fig.html
                                for _fig in empty_figs
                                if _fig.posn.group == group and _fig.posn.tab == tab
                            ]
                        ),
                    ],
                    open=True,
                )
                for group in groups[tab]
            ],
        )
        for tab in tabs
    ]

    return out_tabs


def create_toolbar_callbacks():
    # Auto update switch
    def auto_update_switch(enabled):
        enabled = bool(enabled)
        return not enabled, "Auto update " + ["disabled", "enabled"][enabled]

    _g = globals()
    _g["auto_update_switch"] = app_callback(
        [
            Output("interval-component", "disabled"),
            Output("auto-update-switch", "label"),
        ],
        [Input("auto-update-switch", "value")],
    )(auto_update_switch)


@main.command()
@clx.option(
    "--path",
    dest="glob_path",
    default=(default_path := "**/*.pltr"),
    help=f"Data store path. Will be interpreted with a call to recursive glob (see "
    f"https://docs.python.org/3/library/glob.html). The default is `{default_path}`.",
)
@clx.option(
    "--interval",
    help="Seconds between automatic update of all plots.",
    type=float,
    default=300,
)
@clx.option(
    "--debug",
    action="store_true",
    help="Enables javascript debug console",
    default=False,
)
@clx.option("--host", help="Host name.", default="0.0.0.0")
@clx.option("--port", help="Port number.", default="7000")
@clx.option("--workers", help="Number of workers (ignored in debug mode).", default=3)
@clx.option(
    "--height",
    help=f"Figure height (default={DEFAULT_WIDTH*DEFAULT_HEIGHT_TO_WIDTH})",
    type=int,
    default=DEFAULT_WIDTH * DEFAULT_HEIGHT_TO_WIDTH,
)
@clx.option(
    "--width",
    help=f"Figure width (default={DEFAULT_WIDTH}),",
    type=int,
    default=DEFAULT_WIDTH,
)
@clx.option(
    "--discover",
    help="Search for new files matching glob pattern every interval seconds.",
    choices=["on", "off"],
    default="on",
)
def launch(glob_path, debug, host, interval, height, width, port, workers, discover):
    """
    Launch a ploteries visualization server.
    """
    #
    global DATA_INTERFACES, GLOB_PATH, HEIGHT, WIDTH, LARGE_HEIGHT, LARGE_WIDTH
    GLOB_PATH = glob_path
    HEIGHT = height
    WIDTH = width
    LARGE_HEIGHT = 2 * HEIGHT
    LARGE_WIDTH = 2 * WIDTH

    DATA_INTERFACES = DataInterfaces(glob_path)
    PloteriesLaunchInterface.create_callbacks(
        app_callback,
        lambda data_store_name: DATA_INTERFACES[data_store_name],
        callback_args=dict(
            interface_name_state=State("data-store-dropdown", "value"),
            n_interval_input=Input("interval-component", "n_intervals"),
            global_index_input_value=Input("global-index-dropdown", "value"),
            global_index_dropdown_options=Output("global-index-dropdown", "options"),
        ),
    )
    create_toolbar_callbacks()
    if discover == "on":
        create_discover_callback()

    DATA_INTERFACES.update(verbose=False)
    APP.layout = lambda: create_layout(update_interval=interval)

    if debug:
        APP.run_server(debug=debug, host=host, port=port)
    else:
        options = {
            "bind": "%s:%s" % (host, port),
            "workers": workers,
            "timeout": 180,
        }
        GunicornServer(APP.server, options).run()
