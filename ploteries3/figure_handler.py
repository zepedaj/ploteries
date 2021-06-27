import plotly.graph_objects as go
from pglib.profiling import time_and_print
from .base_handlers import Handler
from sqlalchemy.sql import column
from typing import List
from typing import Dict, Union
from pglib.py import SSQ
from dataclasses import dataclass
from copy import deepcopy
from ploteries3.data_store import col


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


class FigureHandler(Handler):

    data_store = None
    decoded_data_def = None

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

    @classmethod
    def from_def_record(cls, data_store, data_def_record):
        cls.decode_params(data_def_record['params'])
        return cls(data_store, data_def_record.name, **data_def_record.params)

    @classmethod
    def get_defs_table(cls, data_store):
        """
        Returns the defs table (e..g., data_defs or figure_defs)
        """
        return data_store.figure_defs_table

    @classmethod
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

        return figure

    @property
    def is_indexed(self):
        """
        Specifies whether the figure depends on a single time index.
        """
        return any((val.single_record for val in self.sources.values()))
