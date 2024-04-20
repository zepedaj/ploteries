import numpy as np
from dash.dash_table import DataTable
from typing import Callable, Union, Dict
import itertools as it
from jztools.py import class_name
from jztools.validation import checked_get_single
from ploteries.data_store import Col_

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH
from .abstract_hook import AbstractInterfaceHook
from ploteries.figure_handler import TableHandler


class TableHandlerHook(AbstractInterfaceHook):
    """
    Implements the interface (methods :meth:`render_empty_figures` and :meth:`create_callbacks`)) required by the 'ploteries launch' CLI to access the tables in a data store.
    """

    handler_class = TableHandler

    def __init__(self, data_store, data_table_kwargs={}):
        self.data_store = data_store
        self.data_table_kwargs = data_table_kwargs

    # Dictionary ids.
    @classmethod
    def _get_table_id(cls, figure_name):
        return cls._get_id(figure_name, element="data_table")

    @classmethod
    def _get_input_id(cls, figure_name):
        return cls._get_id(figure_name, element="input")

    ##
    def build_empty_html(self, figure_handler):
        """
        Builds empty figure with accompanying html.
        """

        # Empty figure
        table = DataTable()
        self._apply_data_table_kwargs(table)

        return html.Div(
            [
                html.Div(
                    [
                        figure_handler.name,
                        html.Span(
                            dcc.Input(
                                id=self._get_input_id(figure_handler.name),
                                type="text",
                                placeholder="",
                                debounce=True,
                                value=":-20:-1",
                                persistence=True,
                                size=14,
                                style={"marginLeft": "3em", "textAlign": "center"},
                            )
                        ),
                    ]
                ),
                html.Div(table, id=self._get_table_id(figure_handler.name)),
            ],
            style={"display": "inline-block", "margin": "1em"},
        )

    def _apply_data_table_kwargs(self, table):
        [setattr(table, key, val) for key, val in self.data_table_kwargs.items()]

    def _build_formatted_figure_from_name(self, name, slice_obj):
        table = checked_get_single(
            self.data_store.get_figure_handlers(Col_("name") == name)
        ).build_table(slice_obj=slice_obj)
        self._apply_data_table_kwargs(table)
        return table

    @classmethod
    def encoded_class_name(cls):
        return class_name(cls).replace(".", "|")

    @classmethod
    def create_callbacks(
        cls,
        app_callback: Callable,
        get_hook: Callable[[str], "TableHandlerHook"],
        callback_args: Dict[str, Union[State, Input, Output]],
    ):
        """
        Creates the pattern-matching callbacks below (arrow):

            * :attr:`n_interval_input` -> each table figure

        This callbacks expect the following contents in the :attr:`callback_args` dictionary (besides :attr:`interface_name_state`, see :meth:`~ploteries.cli_interface.cli_interface.PloteriesLaunchInterface.create_callbacks`):

            * :attr:`n_interval_input` (:class:`Input`):  The ``Interval.n_intervals`` atttribute that that will trigger the auto-updates, e.g., ``Input('interval-component', 'n_intervals')``.

        (See :meth:`.cli_interface.PloteriesLaunchInterface.create_callbacks` and :meth:`.abstract_hook.AbstractInterfaceHook.create_callbacks`.)

        """

        #
        interface_name_state = callback_args["interface_name_state"]
        n_interval_input = callback_args["n_interval_input"]

        # Figure update on interval tick
        @app_callback(
            Output(cls._get_table_id(figure_name=MATCH), "children"),
            n_interval_input,
            Input(cls._get_input_id(figure_name=MATCH), "value"),
            State(cls._get_table_id(figure_name=MATCH), "id"),
            interface_name_state,
        )
        def update_table(n_interval, input_slice, elem_id, interface_name):
            slice_obj = slice(
                *list(map(lambda _x: int(_x) if _x else None, input_slice.split(":")))
            )
            return get_hook(interface_name)._build_formatted_figure_from_name(
                elem_id["name"], slice_obj
            )
