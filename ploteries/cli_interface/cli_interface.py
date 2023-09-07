"""
Interface to `ploteries launch` CLI.
"""

from typing import Callable, Dict, Union
from dash import Dash
from dash.dependencies import Input, Output, State
from collections import namedtuple

from ploteries.base_handlers import DataHandler

from .figure_handler_hook import FigureHandlerHook
from .table_handler_hook import TableHandlerHook


PosnTuple = namedtuple("PosnTuple", ("tab", "group", "abs_name", "rel_name"))
"""
The name of the tab and group where the figure will be placed.
"""

RenderedFigure = namedtuple("RenderedFigure", ("name", "posn", "html"))
"""
name:str, Specifies the figures absolute name.
posn:PosnTuple, Specifies the figure's hierarchical position.
html:<Dash html obj>  HTML object for the figure.
"""


class PloteriesLaunchInterface:
    """
    Implements the interface (methods :meth:`render_empty_figures` and :meth:`create_callbacks`)) required by the 'ploteries launch' CLI to access the data and figures in a data store.
    """

    hook_classes = [FigureHandlerHook, TableHandlerHook]

    # Default element kwargs
    def __init__(self, data_store, hooks=None):
        hooks = hooks or [cls(data_store) for cls in self.hook_classes]
        self.data_store = data_store
        if not len(set(map(type, hooks))) == len(hooks):
            raise Exception(
                "Can pass at most one hook of each type to avoid defining callbacks multiple times."
            )
        for hook_type in map(type, hooks):
            if hook_type not in self.hook_classes:
                raise ValueError(f"Unsupported hook type {hook_type}.")
        self.hooks = {type(_hook).handler_class: _hook for _hook in hooks}

    # CALLBACKS
    _slider_output_keys = ("marks", "min", "max", "value", "disabled")

    def render_empty_figures(self):
        """
        Generates empty place holders for all figures.
        """
        return [
            RenderedFigure(
                name=_fig_handler.name,
                posn=self._name_to_posn(_fig_handler.name),
                html=self.get_hook(type(_fig_handler)).build_empty_html(_fig_handler),
            )
            for _fig_handler in self.data_store.get_figure_handlers()
        ]

    def get_hook(self, handler_type: DataHandler):
        """Tries to find a matching hook based on class inheritance structure."""
        try:
            return self.hooks[handler_type]
        except KeyError:
            available_ancestors = set(self.hooks.keys())
            if not (
                closest_ancestor := [
                    x for x in handler_type.mro() if x in available_ancestors
                ]
            ):
                raise
            return self.hooks[closest_ancestor[0]]

    def _name_to_posn(self, fig_name):
        default = None
        hierarchy = [_x or default for _x in fig_name.split("/", maxsplit=2)] + [
            default
        ] * 3
        tab, group, rel_name = hierarchy[:3]
        return PosnTuple(tab=tab, group=group, rel_name=rel_name, abs_name=fig_name)

    @classmethod
    def create_callbacks(
        cls,
        app_callback: Callable,
        get_interface: Callable[[str], "PloteriesLaunchInterface"],
        callback_args: Dict[str, Union[State, Input, Output]],
    ):
        """
        This method creates the callbacks required to support web visualizations. It should only be called once to avoid multiply defining callbacks.

        :class:`PloteriesLaunchInterface` supports Dash apps where the data store (i.e., the instance of the :class:`PloteriesLaunchInterface`) is changed by the user from the web interface. Each callback will thus first retrieve the relevant :class:`PloteriesLaunchInterface` object by calling the input callable :attr:`get_interface`, which takes an interface name that is in turn received by the callback from the :class:`Input` :attr:`interface_name`.

        : param app_callback: Method :meth:`dash.Dash.callback` used to decorate callbacks, from the global :attr:`APP` object.
        : param get_interface: Callable that returns an instance of this class. Will be used within hook callbacks to process requests.
        : param callback_args: Dictionary of callback States, Inputs and Outputs required by all dependent hooks. Besides the arguments required by dependent hooks, it must also contain

           * interface_name_state (:class:`State`):  The attribute that provides the interface name, e.g., ``State('data-store-dropdown', 'value')``.
        """

        for hook in cls.hook_classes:
            hook.create_callbacks(
                app_callback,
                lambda path, hook=hook: get_interface(path).get_hook(
                    hook.handler_class
                ),
                callback_args,
            )
