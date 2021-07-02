import plotly.graph_objects as go
from pglib.profiling import time_and_print
from .base_handlers import Handler
from typing import List, Optional
from typing import Dict, Union, Any
from pglib.slice_sequence import SSQ_
from dataclasses import dataclass
from copy import deepcopy
from ploteries3.data_store import Col_, Ref_
from pglib.py import get_nested_keys


class FigureHandler(Handler):

    data_store = None
    decoded_data_def = None

    def __init__(self,
                 data_store,
                 name: str,
                 figure_dict):
        """
        Instantiates a new figure handler. Note that this does not read or write a figure handler from the data store (use methos :meth:`from_name` and :meth:`write_def` for this purpose).

        :param figure_dict: Dictionary representation of a plotly figure containing placeholder values of type :class:`ploteries3.data_store.Ref_` that will be replaced by data from the store when building the figure.

        """
        self.data_store = data_store
        self.name = name
        self.figure_dict = figure_dict
        self._parse_figure()

    def _parse_figure(self):
        # Get all placeholder positions and values.
        self.figure_keys = [
            SSQ_.produce(_keys) for _keys in
            get_nested_keys(self.figure_dict, lambda x: isinstance(x, Ref_))]
        self.data_keys = [_keys(self.figure_dict) for _keys in self.figure_keys]

    def build_figure(self, index=None):
        #
        new_figure = deepcopy(self.figure_dict)

        # Set new index if specified
        if index is not None:
            data_keys = [_data_key.copy() for _data_key in self.data_keys]
            for _data_key in data_keys:
                if _data_key.query_params['index'] == 'latest':
                    _data_key.query_params['index'] = index
        else:
            data_keys = self.data_keys

        # Retrieve data from the data store.
        data = Ref_.call_multi(self.data_store, *data_keys)

        # Assign data to placeholders.
        for _key, _data in zip(self.figure_keys, data):
            _key.set(new_figure, _data)

        #
        new_figure = go.Figure(new_figure)

        return new_figure

    @classmethod
    def from_def_record(cls, data_store, data_def_record):
        return cls(data_store, data_def_record.name, **data_def_record.params)

    @classmethod
    def get_defs_table(cls, data_store):
        """
        Returns the defs table (e..g., data_defs or figure_defs)
        """
        return data_store.figure_defs_table

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        params = {
            'figure_dict': self.figure_dict}
        return params

    @property
    def is_indexed(self):
        """
        Specifies whether the figure depends on a single time index.
        """
        return any((
            _data_key.query_params['index'] == 'latest'
            for _data_key in self.data_keys))
