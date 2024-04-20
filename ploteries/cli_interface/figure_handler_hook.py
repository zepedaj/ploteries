import numpy as np
from typing import Callable, Union, Dict
import functools
import itertools as it
from jztools.py import class_name
from jztools.validation import checked_get_single
from ploteries.data_store import Col_
import plotly.graph_objects as go

# from sqlalchemy.sql import select
from sqlalchemy import select, func
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from .abstract_hook import AbstractInterfaceHook
from ploteries.figure_handler import FigureHandler


class FigureHandlerHook(AbstractInterfaceHook):
    """
    Implements the interface (methods :meth:`render_empty_figures` and :meth:`create_callbacks`)) required by the 'ploteries launch' CLI to access the
    figures in a data store.
    """

    # Default element kwargs
    _default_slider_kwargs = {
        "tooltip": {"always_visible": False, "placement": "top"},
        "updatemode": "mouseup",
        "step": None,
    }

    handler_class = FigureHandler

    def __init__(
        self, data_store, figure_layout_kwargs={}, graph_kwargs={}, slider_kwargs={}
    ):
        self.data_store = data_store
        self.figure_layout_kwargs = {**figure_layout_kwargs}
        self.graph_kwargs = {**graph_kwargs}
        self.slider_kwargs = {**self._default_slider_kwargs, **slider_kwargs}

    # CALLBACKS
    _slider_output_keys = ("marks", "min", "max", "value", "disabled")

    def _int(self, val):
        return f"{val:,d}"

    # Dictionary ids.

    @classmethod
    def _get_figure_id(cls, figure_name, has_slider):
        return cls._get_id(figure_name, element="graph", has_slider=has_slider)

    @classmethod
    def _get_slider_id(cls, figure_name):
        return cls._get_id(figure_name, element="slider")

    ##
    def build_empty_html(self, figure_handler):
        """
        Builds figure without any data and accompanying html.
        """

        #
        has_slider = figure_handler.is_indexed

        # Empty figure
        figure = go.Figure()
        figure.update_layout(**self.figure_layout_kwargs)

        #
        graph = dcc.Graph(
            figure=figure,
            id=self._get_figure_id(
                figure_name=figure_handler.name, has_slider=has_slider
            ),
            **self.graph_kwargs,
        )

        if has_slider:
            slider = dcc.Slider(
                id=self._get_slider_id(figure_name=figure_handler.name),
                **self.slider_kwargs,
            )

        out = html.Div(
            [html.Div([html.Div(figure_handler.name), graph])]
            + ([html.Div([slider])] if has_slider else []),
            style={"display": "inline-block", "margin": "1em"},
        )

        return out

    def _build_formatted_figure_from_name(self, name, index=None):
        figure = checked_get_single(
            self.data_store.get_figure_handlers(Col_("name") == name)
        ).build_figure(index=index)
        figure.update_layout(**self.figure_layout_kwargs)
        return figure

    @classmethod
    def encoded_class_name(cls):
        return class_name(cls).replace(".", "|")

    @classmethod
    def create_callbacks(
        cls,
        app_callback: Callable,
        get_hook: Callable[[str], "FigureHandlerHook"],
        callback_args: Dict[str, Union[State, Input, Output]],
    ):
        """
        Creates three pattern-matching callbacks (corresponding to the arrows below):

          * n_interval_input -> each slider-less figures
          * n_interval_input -> each slider -> each with-slider figures

        These callbacks expect the following contents in the :attr:`callback_args` dictionary (besides :attr:`interface_name_state`, see :meth:`~ploteries.cli_interface.cli_interface.PloteriesLaunchInterface.create_callbacks`):

          * :attr:`n_interval_input` (:class:`Input`):  The ``Interval.n_intervals`` atttribute that that will trigger the auto-updates, e.g., ``Input('interval-component', 'n_intervals')``.
          * :attr:`global_index_input_value` (:class:`Input`):  The global index value that will trigger on-demand figure updates, e.g., ``Input('global-index-dropdown', 'value')``
          * :attr:`global_index_dropdown_options` (:class:`Output`):  Options for global index dropdown menu, e.g., ``Output("global-step-dropdown", "options")``.

        (See :meth:`.cli_interface.PloteriesLaunchInterface.create_callbacks` and :meth:`.abstract_hook.InterfaceHook.create_callbacks`.)

        """

        #
        interface_name_state = callback_args["interface_name_state"]
        n_interval_input = callback_args["n_interval_input"]
        global_index_input_value = callback_args["global_index_input_value"]
        global_index_dropdown_options = callback_args["global_index_dropdown_options"]

        # Figure update on interval tick
        @app_callback(
            Output(cls._get_figure_id(figure_name=MATCH, has_slider=False), "figure"),
            n_interval_input,
            State(cls._get_figure_id(figure_name=MATCH, has_slider=False), "id"),
            interface_name_state,
        )
        def update_figure_with_no_slider(n_interval, elem_id, interface_name):
            return get_hook(interface_name)._build_formatted_figure_from_name(
                elem_id["name"]
            )

        # Figure update on slider change

        @app_callback(
            Output(cls._get_figure_id(figure_name=MATCH, has_slider=True), "figure"),
            Input(cls._get_slider_id(figure_name=MATCH), "value"),
            State(cls._get_slider_id(figure_name=MATCH), "id"),
            interface_name_state,
        )
        def update_figure_with_slider(slider_value, slider_id, interface_name):
            if slider_value is None:
                raise PreventUpdate
            return get_hook(interface_name)._build_formatted_figure_from_name(
                slider_id["name"], index=slider_value
            )

        # Update all sliders and global index dropdown options on interval tick

        @app_callback(
            # Outputs
            (
                [Output(cls._get_slider_id(ALL), _x) for _x in cls._slider_output_keys]
                + [global_index_dropdown_options]
            ),
            # Inputs
            [n_interval_input, global_index_input_value],
            # States
            State(cls._get_slider_id(ALL), "id"),
            interface_name_state,
        )
        def update_all_sliders_and_global_index_dropdown_options(
            n_intervals, global_index, slider_ids, interface_name
        ):
            if not slider_ids:
                raise PreventUpdate
            return get_hook(
                interface_name
            )._update_all_sliders_and_global_index_dropdown_options(
                n_intervals, global_index, slider_ids
            )

    def _get_figure_indices(self, fig_handlers):
        # Get data ids for each figure.
        fig_to_data_def_ids = {
            _fig.name: [
                _dh.decoded_data_def.id
                for _dh in self.data_store.get_data_handlers(
                    Col_("name").in_(_fig.get_data_names())
                )
            ]
            for _fig in fig_handlers
        }

        # Retrieve all the current indices
        with self.data_store.begin_connection() as connection:
            qry = (
                select(
                    self.data_store.data_records_table.c.index.asc(),
                    self.data_store.data_records_table.c.data_def_id,
                )
                .where(
                    self.data_store.data_records_table.c.data_def_id.in_(
                        list(it.chain(*fig_to_data_def_ids.values()))
                    )
                )
                .distinct()
            )
            indices_as_rows = connection.execute(qry)

            # Assign to numpy record array.
            max_possible_indices = connection.execute(
                select(func.count()).select_from(self.data_store.data_records_table)
            ).one()[0]
            indices = np.empty(
                max_possible_indices, dtype=[("index", "i8"), ("data_def_id", "i")]
            )
            _k = -1
            for _k, _row in enumerate(indices_as_rows):
                indices[_k] = tuple(_row)
            indices = indices[: _k + 1]

        # Obtain indices for each figure from intersection of data indices.
        fig_to_indices = {
            fig_name: functools.reduce(
                np.intersect1d,
                [
                    indices["index"][indices["data_def_id"] == _data_def_id]
                    for _data_def_id in data_def_ids
                ],
            )
            for fig_name, data_def_ids in fig_to_data_def_ids.items()
        }

        return fig_to_indices

    def _update_all_sliders_and_global_index_dropdown_options(
        self, n_intervals, global_index, slider_ids
    ):
        """
        Callback helper function that does the following:

        * Always updates marks for all sliders with the latest indices from the data store.
        * Always triggers upates of all figures with the specified global_index(the latest if None).

        : param data_store, cls._slider_output_keys: Bound parameters.
        : param n_intervals, global_index, slider_ids: Same as for: meth: `create_callbacks`.
        """

        # Contains the unique, sorted fig indices for each figure name.
        fig_names = [_x["name"] for _x in slider_ids]
        fig_to_indices = self._get_figure_indices(
            self.data_store.get_figure_handlers(
                self.data_store.figure_defs_table.c.name.in_(fig_names)
            )
        )

        # Build the slider state for each figure.
        new_slider_states = []
        for fig_name in [_x["name"] for _x in slider_ids]:
            #
            indices = fig_to_indices[fig_name].tolist()

            # Set the slider value.
            value = global_index if global_index is not None else indices[-1]

            # Build marks.
            marks = dict(zip(indices, [""] * len(indices)))
            if len(marks) > 0:
                for _m in [indices[0], indices[-1]]:
                    marks[_m] = self._int(int(_m))
            min_mark = min(indices) if indices else None
            max_mark = max(indices) if indices else None

            # Build slider parameters
            new_slider_states.append(
                {
                    "marks": marks,
                    "min": min_mark,
                    "max": max_mark,
                    "value": value,
                    "disabled": len(marks) == 1,
                }
            )

        # Get union of all indices for the global index dropdown.
        all_indices = functools.reduce(np.union1d, fig_to_indices.values()).tolist()
        global_index_dropdown_options = [
            {"label": self._int(int(_x)), "value": int(_x)} for _x in all_indices
        ]

        return [
            [_x[_key] for _x in new_slider_states] for _key in self._slider_output_keys
        ] + [global_index_dropdown_options]
