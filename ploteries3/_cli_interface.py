import abc
from pglib.profiling import time_and_print
from pglib.py import class_name
from pglib.validation import checked_get_single
from ploteries3.data_store import Col_
from typing import List
from .figure_handler import FigureHandler as _FigureHandler
from typing import Optional
import plotly.graph_objects as go
from sqlalchemy.sql import select
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from pglib.sqlalchemy import ClassType as _ClassType
from collections import namedtuple


PosnTuple = namedtuple('PosnTuple', ('tab', 'group', 'abs_name', 'rel_name'))
"""
The name of the tab and group where the figure will be placed.
"""

RenderedFigure = namedtuple('RenderedFigure', ('name', 'posn', 'html'))
"""
name:str, Specifies the figures absolute name.
posn:PosnTuple, Specifies the figure's hierarchical position.
html:<Dash html obj>  HTML object for the figure.
"""


class PloteriesLaunchInterface:
    """
    Implements the interface (methods :meth:`render_empty_figures` and :meth:`create_callbacks`)) required by the 'ploteries launch' CLI to access the data and figures in a data store.
    """

    # Default element kwargs
    _default_slider_kwargs = {
        'tooltip': {
            'always_visible': False,
            'placement': 'top'},
        'updatemode': 'mouseup',
        'step': None}

    def __init__(self,
                 data_store,
                 figure_layout_kwargs={},
                 graph_kwargs={},
                 slider_kwargs={}):
        self.data_store = data_store
        self.figure_layout_kwargs = {**figure_layout_kwargs}
        self.graph_kwargs = {**graph_kwargs}
        self.slider_kwargs = {**self._default_slider_kwargs, **slider_kwargs}

    # CALLBACKS
    _slider_output_keys = ('marks', 'min', 'max', 'value', 'disabled')

    def render_empty_figures(self):
        """

        """
        return [
            RenderedFigure(
                name=_fig_handler.name,
                posn=self._name_to_posn(_fig_handler.name),
                html=self._build_empty_html(_fig_handler))
            for _fig_handler in self.data_store.get_figure_handlers()
        ]

    def _int(self, val):
        return f'{val:,d}'

    def _name_to_posn(self, fig_name):
        default = None
        hierarchy = fig_name.split('/')
        if len(hierarchy) == 1:
            tab, group, rel_name = (default, default, fig_name)
        elif len(hierarchy) == 2:
            tab, group, rel_name = (hierarchy[0], default, hierarchy[1])
        elif len(hierarchy) > 2:
            tab, group, rel_name = hierarchy[0], hierarchy[1],  '/'.join(hierarchy[2:])
        else:
            raise Exception('Unexpected case!')
        return PosnTuple(tab=tab, group=group, rel_name=rel_name, abs_name=fig_name)

    # Dictionary ids.
    def _get_figure_id(self, figure_name, has_slider):
        return {
            'name': figure_name,
            'type': self.encoded_class_name(),
            'element': 'graph',
            'has_slider': has_slider,
        }

    def _get_slider_id(self, figure_name):
        return {
            'name': figure_name,
            'type': self.encoded_class_name(),
            'element': 'slider'
        }

    ##
    def _build_empty_html(self, figure_handler):
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
                figure_name=figure_handler.name,
                has_slider=has_slider),
            ** self.graph_kwargs)

        if has_slider:
            slider = dcc.Slider(
                id=self._get_slider_id(figure_name=figure_handler.name),
                **self.slider_kwargs)

        out = html.Div(
            [html.Div([html.Div(figure_handler.name), graph])] +
            ([html.Div([slider])] if has_slider else []),
            style={'display': 'inline-block', 'margin': '1em'})

        return out

    def _build_formatted_figure_from_name(self, name, index=None):
        figure = checked_get_single(
            self.data_store.get_figure_handlers(Col_('name') == name)).build_figure(index=index)
        figure.update_layout(**self.figure_layout_kwargs)
        return figure

    @classmethod
    def encoded_class_name(cls):
        return class_name(cls).replace('.', '|')

    def create_callbacks(
            self, app: Dash,
            n_interval_input: Input,
            global_index_input_value: Input,
            global_index_dropdown_options: Output):
        """
        Produces three callbacks(corresponding to arrows below):
        * n_interval_input -> slider-less figures
        * n_interval_input -> slider -> with-slider figures

        : param app: The Dash object where callbacks are added.
        : param data_store: The data store.
        : param n_interval_input: The ``Interval.n_intervals`` atttribute that that will trigger the auto-updates, e.g., ``Input('interval-component', 'n_intervals')``.
        : param global_index_input_value: The global index value that will trigger on-demand figure updates, e.g., ``Input('global-index-dropdown', 'value')``
        : param global_index_dropdown_options: Options for global index dropdown menu, e.g., ``Output("global-step-dropdown", "options")``.
        : param registry: A set where figure handlers are registered to avoid re-creating existing callbacks. An error will be raised if the callbacks for this handler have already been registered.
        """

        # Figure update on interval tick
        @app.callback(
            Output(
                self._get_figure_id(figure_name=MATCH, has_slider=False),
                'figure'),
            n_interval_input,
            State(
                self._get_figure_id(figure_name=MATCH, has_slider=False),
                'id'))
        def update_figure_with_no_slider(n_interval, elem_id):
            return self._build_formatted_figure_from_name(elem_id['name'])

        # Figure update on slider change

        @app.callback(
            Output(
                self._get_figure_id(figure_name=MATCH, has_slider=True),
                'figure'),
            Input(
                self._get_slider_id(figure_name=MATCH),
                'value'),
            State(
                self._get_slider_id(figure_name=MATCH),
                'id'))
        @time_and_print()
        def update_figure_with_slider(slider_value, slider_id):
            if slider_value is None:
                raise PreventUpdate
            return self._build_formatted_figure_from_name(slider_id['name'], index=slider_value)

        # Update all sliders and global index dropdown options on interval tick

        @app.callback(
            # Outputs
            ([Output(
                self._get_slider_id(ALL), _x)
              for _x in self._slider_output_keys] +
             [global_index_dropdown_options]),
            # Inputs
            [n_interval_input, global_index_input_value],
            # States
            [State(
                self._get_slider_id(ALL),
                'id')])
        def update_all_sliders_and_global_index_dropdown_options(
                n_intervals, global_index, slider_ids):
            return self._update_all_sliders_and_global_index_dropdown_options(
                n_intervals, global_index, slider_ids)

    def _update_all_sliders_and_global_index_dropdown_options(
            self, n_intervals, global_index, slider_ids):
        """
        Callback helper function that does the following:

        * Always updates marks for all sliders with the latest indices from the data store.
        * Always triggers upates of all figures with the specified global_index(the latest if None).

        : param data_store, cls._slider_output_keys: Bound parameters.
        : param n_intervals, global_index, slider_ids: Same as for: meth: `create_callbacks`.
        """

        # Retrieve all the current indices
        with self.data_store.begin_connection() as connection:
            indices = [_x[0] for _x in connection.execute(
                select(self.data_store.data_records_table.c.index.asc()).distinct()).fetchall()]

        # Set the slider value.
        value = global_index if global_index is not None else indices[-1]

        # Build marks.
        marks = dict(zip(indices, ['']*len(indices)))
        if len(marks) > 0:
            for _m in [indices[0], indices[-1]]:
                marks[_m] = self._int(int(_m))
        min_mark = min(indices)
        max_mark = max(indices)

        # Build slider parameters
        shared_slider_state = {'marks': marks, 'min': min_mark,
                               'max': max_mark, 'value': value, 'disabled': len(marks) == 1}

        global_index_dropdown_options = [
            {'label': self._int(int(_x)),
             'value': int(_x)} for _x in marks]

        return [
            [shared_slider_state[key]]*len(slider_ids)
            for key in self._slider_output_keys
        ] + [global_index_dropdown_options]
