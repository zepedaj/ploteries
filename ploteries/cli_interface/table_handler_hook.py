import numpy as np
from dash_table import DataTable
from typing import Callable, Union, Dict
import functools
import itertools as it
from pglib.profiling import time_and_print
from pglib.py import class_name
from pglib.validation import checked_get_single
from ploteries.data_store import Col_
import plotly.graph_objects as go
# from sqlalchemy.sql import select
from sqlalchemy import select, func
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from .abstract_hook import AbstractInterfaceHook
from ploteries.figure_handler import TableHandler


class TableHandlerHook(AbstractInterfaceHook):
    """
    Implements the interface (methods :meth:`render_empty_figures` and :meth:`create_callbacks`)) required by the 'ploteries launch' CLI to access the tables in a data store.
    """

    handler_class = TableHandler

    def __init__(self,
                 data_store,
                 data_table_kwargs={}):
        self.data_store = data_store
        self.data_table_kwargs = data_table_kwargs

    # Dictionary ids.
    @classmethod
    def _get_table_id(cls, figure_name):
        return cls._get_id(figure_name, element='data_table')

    ##
    def build_empty_html(self, figure_handler):
        """
        Builds empty figure with accompanying html.
        """

        # Empty figure
        table = DataTable()
        self._apply_data_table_kwargs(table)

        return html.Div(table, id=self._get_table_id(figure_handler.name))

    def _apply_data_table_kwargs(self, table):
        [setattr(table, key, val) for key, val in self.data_table_kwargs.items()]

    def _build_formatted_figure_from_name(self, name):
        table = checked_get_single(
            self.data_store.get_figure_handlers(Col_('name') == name)).build_table()
        self._apply_data_table_kwargs(table)
        return table

    @classmethod
    def encoded_class_name(cls):
        return class_name(cls).replace('.', '|')

    @classmethod
    def create_callbacks(
            cls,
            app: Dash,
            get_hook: Callable[[str], 'TableHandlerHook'],
            callback_args: Dict[str, Union[State, Input, Output]]):
        """
        Creates the pattern-matching callbacks below (arrow):

            * :attr:`n_interval_input` -> each table figure

        This callbacks expect the following contents in the :attr:`callback_args` dictionary (besides :attr:`interface_name_state`, see :meth:`~ploteries.cli_interface.cli_interface.PloteriesLaunchInterface.create_callbacks`):

            * :attr:`n_interval_input` (:class:`Input`):  The ``Interval.n_intervals`` atttribute that that will trigger the auto-updates, e.g., ``Input('interval-component', 'n_intervals')``.

        (See :meth:`.cli_interface.PloteriesLaunchInterface.create_callbacks` and :meth:`.abstract_hook.AbstractInterfaceHook.create_callbacks`.)

        """

        #
        interface_name_state = callback_args['interface_name_state']
        n_interval_input = callback_args['n_interval_input']

        # Figure update on interval tick
        @app.callback(
            Output(
                cls._get_table_id(figure_name=MATCH),
                'children'),
            n_interval_input,
            State(
                cls._get_table_id(figure_name=MATCH),
                'id'),
            interface_name_state
        )
        @time_and_print()
        def update_table(n_interval, elem_id, interface_name):
            return get_hook(interface_name)._build_formatted_figure_from_name(elem_id['name'])
