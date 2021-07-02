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

    # def add_figure(
    #         self,
    #         name: str,
    #         traces: List[Dict],
    #         figure: Optional[Union[Dict, go.Figure]] = None,
    #         overwrite=False):
    #     """
    #     Creates a figure with the given name and containing the specified traces.

    #     :param sources: Data source specification. Same syntax as :class:`~ploteries.figure_handler.FigureHandler`
    #     :param traces: Trace specification as dictionary. Fields in this dictionary that are :class:`Ref_` instances will be linked to the data source.

    #     Example:
    #     ```
    #     add_data('uniform', 'msft_stock', 0, np.array([0.0, 1.1, 2.2]))
    #     add_figure(
    #         traces = [
    #             {'type': 'scatter', 'name': 'plot 1',
    #              'x': Ref_()['msft_stock']['data'][::2], 'y': Ref_()[1, ::2]},
    #             {'type': 'scatter', 'name': 'plot 1',
    #              'x': Ref_()[{'series':'msft_stock', 'index':'latest'}]['data'][::2], 'y': Ref_()[{'series':'msft_stock', 'index':'latest'}],1, ::2]}]
    #     )
    #     ```
    #     """

    #     # Build data mappings, remove Ref_ objects from traces.
    #     mappings = []

    #     for k, _trace in enumerate(traces):
    #         mappings.extend([
    #             {'figure_keys': ('data', k, key), 'data_keys': ssq}
    #             for key, ssq in _trace.items() if isinstance(ssq, Ref_)])
    #         traces[k] = {
    #             key: val for key, val in _trace.items() if not isinstance(val, Ref_)}

    #     # Build figure.
    #     if not figure:
    #         figure = go.Figure(layout_template=None)
    #     elif not isinstance(figure, go.Figure):
    #         raise TypeError('Arg figure needs to be a go.Figure object.')

    #     # Add traces, ensure type checking.
    #     for trace in traces:
    #         figure.add_trace(getattr(go, trace.pop('type').capitalize())(**trace))

    #     # Add the figure definition to the data store.
    #     fh = FigureHandler(self.data_store, name, sources, mappings, figure)
    #     fh.write_def()
