import plotly.graph_objects as go
from pglib.profiling import time_and_print
from .base_handlers import Handler
from sqlalchemy.sql import column, select
from pglib.sqlalchemy import ClassType as _ClassType
from typing import List
from dash import Dash
from typing import Dict, Union, Optional
from pglib.py import SSQ
from dataclasses import dataclass
from copy import deepcopy
from ploteries3.data_store import col

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate


@dataclass
class SourceSpec:
    """ Defines a data source configuration. """

    data_name: str
    """ Name of data in data store. """

    single_record: bool = False
    """ Load a single record. """

    @classmethod
    def produce(cls, val):
        if isinstance(val, str):
            return cls(val)
        elif isinstance(val, dict):
            return SourceSpec(**val)
        elif isinstance(val, cls):
            return val
        else:
            raise TypeError(f'Invalid type {type(val)}.')

    def as_serializable(self):
        return {'data_name': self.data_name,
                'single_record': self.single_record}

    @classmethod
    def from_serializable(cls, val):
        return cls(**val)


class Mapping:
    """
    Defines how a figure entry is filled from a data source entry. Entries are defined by sequences of keys encoded internally as :class:`SliceSequence` objects.
    """
    figure_keys: SSQ
    """ The slice sequence-producible that is used to index the figure dictionary."""
    source_keys: SSQ
    """ The slice sequence-producible that is used to index the source."""

    def __init__(self, figure_keys, source_keys):
        self.figure_keys = SSQ.produce(figure_keys)
        self.source_keys = SSQ.produce(source_keys)

    def __eq__(self, val):
        return self.figure_keys == val.figure_keys and self.source_keys == val.source_keys

    @classmethod
    def produce(cls, val):
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return cls(*val)
        elif isinstance(val, dict):
            return cls(**val)
        elif isinstance(val, cls):
            return val
        else:
            raise TypeError(f'Invalid type {type(val)}.')

    def as_serializable(self):
        return {'figure_keys': self.figure_keys.as_serializable(),
                'source_keys': self.source_keys.as_serializable()}

    @classmethod
    def from_serializable(cls, serializable):
        kwargs = {
            'figure_keys': SSQ.from_serializable(serializable['figure_keys']),
            'source_keys': SSQ.from_serializable(serializable['source_keys'])}
        return cls.produce(kwargs)


def encoded_class_name(in_class):
    """
    Dash hangs if an identifier dictionary in a pattern matching callback contains a value string with '.' in it.
    """
    in_class = in_class if isinstance(in_class, type) else type(in_class)
    return _ClassType.class_name(in_class).replace('.', '|')


class FigureHandler(Handler):

    data_store = None
    decoded_data_def = None

    # Default element kwargs
    default_slider_kwargs = {
        'tooltip': {
            'always_visible': False,
            'placement': 'top'},
        'updatemode': 'mouseup',
        'step': None}
    default_figure_layout_kwargs = {}
    default_graph_kwargs = {}

    def _int(val):
        return f'{val:,d}'

    def __init__(self,
                 data_store,
                 name: str,
                 sources: Dict[str, Union[str, dict]],
                 mappings: List[Mapping],
                 figure: Union[go.Figure, dict]):
        """

        Instantiates a new figure handler. Note that this does not read or write a figure handler from the data store. To read or write a figure handler's definition form the data store, use :meth:`from_name` or :meth:`write_def`.

        :param sources:  A ``source_name`` to ``data_name`` dictionary. The data name can contain the name of a data_records table record, or optionally be a dictionary with configuration parameters (supported fields: `name:str`, `single_record:bool`). The source name will be used in the ``mappings`` parameter. Example:
        ```
        {'source_name0': 'data_name0',
         'source_name1': {'name':'data_name1', single_record:True}}
        ```
        Valid configuration parameters to be used as dictionary values are (see :class:`SourceSpec`):

        *`'data_name' (str)`: The name of the data_store data member.
        *`'single_record' (False|True)`: Whether to load a single record or all records.

        :param mappings: A list of :class:`Mapping`-producibles that describe how to fill (nested) figure fields from (nested) source slices. Example:
        ```
        [{'figure_keys':SSQ()['data'][0]['x'], 'source_keys':SSQ()['source_name0']['meta']['index']},
         {'figure_keys':SSQ()['data'][0]['y'], 'source_keys':SSQ()
                            ['source_name1']['data']['field1']},
         {'figure_keys':SSQ()['data'][0]['z'], 'source_keys':SSQ()['source_name1']['data']['field2']}]
        ```
        :class:`SSQ`s represent slice sequences and provide a more natural way to access nested fields. For example, ``SSQ()['layer0'][1]['f0']({'layer0': [None, {'layer1':np.array([(10,20), (10,20)], dtype=[('f0','i'), ('f1', 'i')])}]})`` would extract  a reference to field ``'f0'`` of the numpy array  ``np.array([(20,20)], dtype='f')``.

        Other, more concise forms that can be combined are 2-tuples or dictionaries of :class:`SSQ`-producibles:
        ```
        [(('data', 0, 'x'), ['source_name0', 'meta', 'index']),
         (SSQ()['data'][0]['y'], SSQ()['source_name1']['data']['field1']),
         {'figure_keys':('data', 0, 'z'), 'source_keys':('source_name1', 'data', 'field2')}]
        ```

        .. todo:: Add support for joining data sources.
        """
        self.data_store = data_store
        self.name = name
        self.sources = {key: SourceSpec.produce(val) for key, val in sources.items()}
        self.mappings = tuple([Mapping.produce(val) for val in mappings])
        self.figure = figure.to_dict() if isinstance(figure, go.Figure) else figure

    def write_def(self):
        """
        Adds a record to the data store's :attr:`figure_defs` table.
        """
        self._write_def()

    @ classmethod
    def from_def_record(cls, data_store, data_def_record):
        cls.decode_params(data_def_record['params'])
        return cls(data_store, data_def_record.name, **data_def_record.params)

    @ classmethod
    def get_defs_table(cls, data_store):
        """
        Returns the defs table (e..g., data_defs or figure_defs)
        """
        return data_store.figure_defs_table

    @ classmethod
    def decode_params(cls, params):
        """
        In-place decoding of the the params field of the data_defs record.
        """
        params['mappings'] = tuple(
            [Mapping.from_serializable(val) for val in params['mappings']])
        params['sources'] = {
            key: SourceSpec.from_serializable(val)
            for key, val in params['sources'].items()}

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        params = {
            'sources': {
                key: val.as_serializable()
                for key, val in self.sources.items()},
            'mappings': [
                val.as_serializable() for val in self.mappings],
            'figure': self.figure}
        return params

    def _load_figure_data(self, index=None, connection=None):
        """
        Loads the most up-to-date data from the data store.
        """

        with self.data_store.begin_connection(connection=connection) as connection:

            # Get data handlers.
            handlers = {
                name: self.data_store.get_data_handlers(
                    col('name') == spec.data_name, connection=connection)[0]
                for name, spec in self.sources.items()}

            # Load the data.
            loaded_data = {}
            for name, spec in self.sources.items():
                criterion = (
                    [column('index') == index] if (index and spec.single_record) is not None
                    else [])
                loaded_data[name] = handlers[name].load_data(
                    single_record=spec.single_record, *criterion, connection=connection)

        return loaded_data

    def build_figure(self, index=None):
        """
        Returns a Figure object with the most up-to-date data from the data store.
        """

        # Load data.
        data = self._load_figure_data(index)
        figure_dict = deepcopy(self.figure)
        for _mapping in self.mappings:
            _mapping.figure_keys.assign(figure_dict, _mapping.source_keys(data))

        # Build figure.
        figure = go.Figure(figure_dict)
        figure.update_layout(**self.default_figure_layout_kwargs)

        return figure

    def has_slider(self):
        return any((val.single_record for val in self.sources.values()))

    def build_html(self, index=None, empty=False):
        """
        :param index: Specifies which record to retrieve for single-record data mappings.
        :param empty: If true, return an empty, place-holder figure with no data in it.
        """

        # Get figure.
        if empty:
            figure = go.Figure()
            figure.update_layout(**self.default_figure_layout_kwargs)
        else:
            figure = self.build_figure(index=index)

        graph = dcc.Graph(
            figure=figure,
            id={
                'type': encoded_class_name(self),
                'element': 'graph',
                'has_slider': self.has_slider(),
                'name': self.name
            },
            **self.default_graph_kwargs)

        if self.has_slider():
            slider = dcc.Slider(
                id={
                    'type': encoded_class_name(self),
                    'element': 'slider',
                    'name': self.name
                },
                **self.default_slider_kwargs)

        out = html.Div(
            [html.Div([html.Div(self.name), graph])] +
            ([html.Div([slider])] if self.has_slider() else []),
            style={'display': 'inline-block', 'margin': '1em'})

        return out

    @ classmethod
    def create_dash_callbacks(
            cls, app: Dash,
            data_store,
            n_interval_input,
            global_index_input_value,
            global_index_dropdown_options,
            registry: Optional[set] = set):
        """
        Produces three callbacks (corresponding to arrows below):
        * interval -> slider-less figures
        * intreval -> slider -> with-slider figures

The first one updates each single_record figure whenever a slider changes or when . The second one f

        :param app: The Dash object where callbacks are added.
        :param data_store: The data store.
        :param n_interval_input: The ``Interval.n_intervals`` atttribute that that will trigger the auto-updates, e.g., ``Input('interval-component', 'n_intervals')``.
        :param global_index_input_value: The global index value that will trigger on-demand figure updates, e.g., ``Input('global-index-dropdown', 'value')``
        :param global_index_dropdown_options: Options for global index dropdown menu, e.g., ``Output("global-step-dropdown", "options")``.
        :param registry: A set where figure handlers are registered to avoid re-creating existing callbacks. An error will be raised if the callbacks for this handler have already been registered.
        """

        # Check if callbacks previously created.
        if cls in registry:
            raise Exception(f'Callbacks already created for {cls}.')
        else:
            registry.add(cls)

        # Figure update on interval tick

        @ app.callback(
            Output(
                fig_id := {
                    'type': encoded_class_name(cls),
                    'element': 'graph',
                    'has_slider': False,
                    'name': MATCH},
                'figure'),
            n_interval_input,
            State(fig_id, 'id'))
        def update_figure_with_no_slider(n_interval, elem_id):
            fig_handler = cls.from_name(data_store, elem_id['name'])
            figure = fig_handler.build_figure()
            return figure

        # Figure update on slider change

        @ app.callback(
            Output(
                {'type': encoded_class_name(cls),
                 'element': 'graph', 'has_slider': True, 'name': MATCH},
                'figure'),
            Input(
                slider_id :=
                {'type': encoded_class_name(cls),
                 'element': 'slider', 'name': MATCH},
                'value'),
            State(slider_id, 'id'))
        def update_figure_with_slider(index, slider_value, slider_id):
            if slider_value is None:
                raise PreventUpdate
            fig_handler = cls.from_name(data_store, slider_id['name'])
            return fig_handler.build_figure(index=index)

        # Update all sliders and global index dropdown options on interval tick
        slider_output_keys = ('marks', 'min', 'max', 'value', 'disabled')

        @ app.callback(
            [Output({'type': encoded_class_name(cls),
                     'element': 'slider',
                     'name': ALL}, _x)
             for _x in slider_output_keys] + [global_index_dropdown_options],
            [n_interval_input, global_index_input_value],
            [State({'type': encoded_class_name(cls),
                    'element': 'slider',
                    'name': ALL}, 'id')])
        def update_all_sliders_and_global_index_dropdown_options(
                n_intervals, global_index, slider_ids):
            """
            Always updates marks for all sliders with the latest indices from the data store.
            Always triggers upates of all figures with the specified global_index (the latest if None).
            """

            # Retrieve all the current indices
            with data_store.begin_connection() as connection:
                indices = [_x[0] for _x in connection.execute(
                    select(data_store.data_records_table.c.index.asc()).distinct()).fetchall()]

            # Set the slider value.
            value = global_index if global_index is not None else indices[-1]

            # Build marks.
            marks = dict(zip(indices, ['']*len(indices)))
            if len(marks) > 0:
                for _m in [indices[0], indices[-1]]:
                    marks[_m] = cls._int(int(_m))
            min_mark = min(indices)
            max_mark = max(indices)

            # Build slider parameters
            shared_slider_state = {'marks': marks, 'min': min_mark,
                                   'max': max_mark, 'value': value, 'disabled': len(marks) == 1}

            global_index_dropdown_options = [
                {'label': cls._int(int(_x)),
                 'value': int(_x)} for _x in marks]

            return [
                [shared_slider_state[key]]*len(slider_ids)
                for key in slider_output_keys
            ] + [global_index_dropdown_options]
