import numpy as np
from typing import Callable
import functools
import itertools as it
from pglib.profiling import time_and_print
from pglib.py import class_name
from pglib.validation import checked_get_single
from ploteries3.data_store import Col_
import plotly.graph_objects as go
#from sqlalchemy.sql import select
from sqlalchemy import select, func
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
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
        hierarchy = [_x or default for _x in fig_name.split('/', maxsplit=2)] + [default]*3
        tab, group, rel_name = hierarchy[:3]
        return PosnTuple(tab=tab, group=group, rel_name=rel_name, abs_name=fig_name)

    # Dictionary ids.
    @classmethod
    def _get_figure_id(cls, figure_name, has_slider):
        return {
            'name': figure_name,
            'type': cls.encoded_class_name(),
            'element': 'graph',
            'has_slider': has_slider,
        }

    @classmethod
    def _get_slider_id(cls, figure_name):
        return {
            'name': figure_name,
            'type': cls.encoded_class_name(),
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

    @classmethod
    def create_callbacks(
            cls,
            app: Dash,
            get_interface: Callable[[str], 'PloteriesLaunchInterface'],
            interface_name_state: State,
            n_interval_input: Input,
            global_index_input_value: Input,
            global_index_dropdown_options: Output):
        """        

        This method creates the callbacks required to support web valizations. In only needs to be called once for each :class:`PloteriesLaunchInterface` class. It produces three pattern-matching callbacks (corresponding to the arrows below):
        * n_interval_input -> each slider-less figures
        * n_interval_input -> each slider -> each with-slider figures        

        :class:`PloteriesLaunchInterface` supports Dash apps where the data store (i.e., the instance of the :class:`PloteriesLaunchInterface`) is changed by the user from the web interface. Each callback will thus first retrieve the relevant :class:`PloteriesLaunchInterface` object by calling the input callable :attr:`get_interface`, which takes an interface name that is in turn received by the callback from the :class:`Input` :attr:`interface_name`.

        : param app: The Dash object where callbacks are added.
        : param get_interface: Callable that returns an instance of this class. Will be used within callbacks to process requests.
        : param interface_name: The ``Dropdown.value`` attribute that provides the interface name, e.g., ``State('data-store-dropdown', 'value')``
        : param n_interval_input: The ``Interval.n_intervals`` atttribute that that will trigger the auto-updates, e.g., ``Input('interval-component', 'n_intervals')``.
        : param global_index_input_value: The global index value that will trigger on-demand figure updates, e.g., ``Input('global-index-dropdown', 'value')``
        : param global_index_dropdown_options: Options for global index dropdown menu, e.g., ``Output("global-step-dropdown", "options")``.
        """

        # Figure update on interval tick
        @app.callback(
            Output(
                cls._get_figure_id(figure_name=MATCH, has_slider=False),
                'figure'),
            n_interval_input,
            State(
                cls._get_figure_id(figure_name=MATCH, has_slider=False),
                'id'),
            interface_name_state
        )
        @time_and_print()
        def update_figure_with_no_slider(n_interval, elem_id, interface_name):
            return get_interface(interface_name)._build_formatted_figure_from_name(elem_id['name'])

        # Figure update on slider change

        @app.callback(
            Output(
                cls._get_figure_id(figure_name=MATCH, has_slider=True),
                'figure'),
            Input(
                cls._get_slider_id(figure_name=MATCH),
                'value'),
            State(
                cls._get_slider_id(figure_name=MATCH),
                'id'),
            interface_name_state
        )
        @time_and_print()
        def update_figure_with_slider(slider_value, slider_id, interface_name):
            if slider_value is None:
                raise PreventUpdate
            return get_interface(interface_name)._build_formatted_figure_from_name(
                slider_id['name'],
                index=slider_value)

        # Update all sliders and global index dropdown options on interval tick

        @app.callback(
            # Outputs
            ([Output(
                cls._get_slider_id(ALL), _x)
              for _x in cls._slider_output_keys] +
             [global_index_dropdown_options]),
            # Inputs
            [n_interval_input, global_index_input_value],
            # States
            State(
                cls._get_slider_id(ALL),
                'id'),
            interface_name_state)
        @time_and_print()
        def update_all_sliders_and_global_index_dropdown_options(
                n_intervals, global_index, slider_ids, interface_name):
            if not slider_ids:
                raise PreventUpdate
            return \
                get_interface(interface_name)._update_all_sliders_and_global_index_dropdown_options(
                    n_intervals, global_index, slider_ids)

    def _get_figure_indices(self, fig_handlers):
        # Get data ids for each figure.
        fig_to_data_def_ids = {
            _fig.name:
            [_dh.decoded_data_def.id for _dh in self.data_store.get_data_handlers(
                Col_('name').in_(_fig.get_data_names()))]
            for _fig in fig_handlers}

        # Retrieve all the current indices
        with self.data_store.begin_connection() as connection:
            qry = select(
                [self.data_store.data_records_table.c.index.asc(),
                 self.data_store.data_records_table.c.data_def_id]
            ).where(
                self.data_store.data_records_table.c.data_def_id.in_(
                    list(it.chain(*fig_to_data_def_ids.values())))
            ).distinct()
            indices_as_rows = connection.execute(qry)

            # Assign to numpy record array.
            max_possible_indices = connection.execute(
                select([func.count()]).select_from(self.data_store.data_records_table)).one()[0]
            indices = np.empty(max_possible_indices, dtype=[('index', 'i'), ('data_def_id', 'i')])
            _k = -1
            for _k, _row in enumerate(indices_as_rows):
                indices[_k] = tuple(_row)
            indices = indices[:_k+1]

        # Obtain indices for each figure from intersection of data indices.
        fig_to_indices = {
            fig_name: functools.reduce(
                np.intersect1d,
                [indices['index'][indices['data_def_id'] == _data_def_id]
                 for _data_def_id in data_def_ids])
            for fig_name, data_def_ids in fig_to_data_def_ids.items()}

        return fig_to_indices

    def _update_all_sliders_and_global_index_dropdown_options(
            self, n_intervals, global_index, slider_ids):
        """
        Callback helper function that does the following:

        * Always updates marks for all sliders with the latest indices from the data store.
        * Always triggers upates of all figures with the specified global_index(the latest if None).

        : param data_store, cls._slider_output_keys: Bound parameters.
        : param n_intervals, global_index, slider_ids: Same as for: meth: `create_callbacks`.
        """

        # Contains the unique, sorted fig indices for each figure name.
        fig_names = [_x['name'] for _x in slider_ids]
        fig_to_indices = self._get_figure_indices(
            self.data_store.get_figure_handlers(self.data_store.figure_defs_table.c.name.in_(
                fig_names)))

        # Build the slider state for each figure.
        new_slider_states = []
        for fig_name in [_x['name'] for _x in slider_ids]:

            #
            indices = fig_to_indices[fig_name].tolist()

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
            new_slider_states.append({
                'marks': marks, 'min': min_mark,
                'max': max_mark, 'value': value,
                'disabled': len(marks) == 1})

        # Get union of all indices for the global index dropdown.
        all_indices = functools.reduce(np.union1d, fig_to_indices.values()).tolist()
        global_index_dropdown_options = [
            {'label': self._int(int(_x)),
             'value': int(_x)} for _x in all_indices]

        return [
            [_x[_key] for _x in new_slider_states]
            for _key in self._slider_output_keys] + [global_index_dropdown_options]
