import plotly.graph_objects as go
from .base_handlers import Handler
from sqlalchemy.sql import column
from dash import Dash
from typing import Dict, Union
from pglib.py import SSQ
from dataclasses import dataclass
from copy import deepcopy
from ploteries3.data_store import _c


@dataclass
class SourceSpec:
    """ Defines a data source specification. """

    data_name: str
    """ Name of data in data store. """

    single_record: bool = False
    """ Load a single record. """


class FigureHandler(Handler):

    data_store = None
    decoded_data_def = None

    def __init__(self,
                 data_store,
                 name: str,
                 sources: Dict[str, Union[str, dict]],
                 data_mappings: Dict[SSQ, SSQ],
                 figure: Union[go.Figure, dict]):
        """
        :param sources:  A ``source_name`` to ``data_name`` dictionary. The data name can contain the name of a data_records table record, or optionally be a dictionary with configuration parameters (supported fields: `name:str`, `single_record:bool`). The source name will be used in the ``data_mappings`` parameter. Example:
        ```
        {'source_name1': 'data_name1',
         'source_name2': {'name':'data_name2', single_record:True}}
        ```
        Valid configuration parameters to be used as dictionary values are (see :class:`SourceSpec`):

        *`'data_name' (str)`: The name of the data_store data member.
        *`'single_record' (False|True)`: Whether to load a single record or all records.

        :param data_mappings: Dictionary with :class:`pglib.py.SSQ`-producible keys indicating the figure fields to fill, and :class:`pglib.py.SSQ`-producible values indicating the data source slice to use.  Example:
        ```
        {('data', 0, 'x'): ['source_name1', 'data', 'field1'],
         ('data', 0, 'y'): ['source_name2', 'data', 'field2']}
        ```

        .. todo:: Add support for joining data sources.
        """
        self.data_store = data_store
        self.name = name
        self.sources = {key: SourceSpec(val) if isinstance(val, str) else
                        SourceSpec(**val) for key, val in sources.items()}
        self.data_mappings = {SSQ(key): SSQ(val) for key, val in data_mappings.items()}
        self.figure = figure.to_dict() if isinstance(figure, go.Figure) else figure

    @classmethod
    def from_def_record(self, data_store, data_def_record):
        ############################## TODO ###################
        pass

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
        params['data_mappings'] = {
            SSQ.deserialize(key): SSQ.deserialize(val) for key, val in
            params['data_mappings'].items()}

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        params = {
            'sources': self.sources,
            'data_mappings': {key.serialize(): val.serialize() for key, val in self.sources},
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
                    _c('name') == spec.data_name, connection=connection)[0]
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

        data = self._load_figure_data(index)
        figure_dict = deepcopy(self.figure)
        for fig_key, data_key in self.data_mappings.items():
            fig_key.assign(figure_dict, data_key(data))

        return go.Figure(figure_dict)

    @ classmethod
    def create_dash_callbacks(cls, APP: Dash):
        pass
