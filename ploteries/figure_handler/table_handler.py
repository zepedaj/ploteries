from .figure_handler import FigureHandler as _FigureHandler
from pglib.py import strict_zip
from numbers import Number
from pglib import validation as pgval
import itertools as it
from ploteries.data_store import Ref_
from typing import List, Optional
import plotly.graph_objects as go
from copy import deepcopy


class TableHandler(_FigureHandler):
    """
    This object creates a table from dictionaries, where the dictionary at time step k fills the k-th row (or column) in the table. It supports a subset of functionality of `Plotly tables <https://plotly.com/python/table/>`, including all header kwargs shared across all header (same for cells) - see the same link for the syntax. Use :class:`~ploteries.figure_handler.figure_handler.FigureHandler` for tables that support all functionality but have fixed, pre-determined size, since Plotly figure objects can render tables.
    """

    time_step_key = 'Time step'

    def __init__(self,
                 data_store,
                 name: str,
                 source_data_name: str,
                 transposed: bool = False,
                 header_kwargs={'align': 'right'},
                 cells_kwargs={'align': 'right'},
                 decoded_data_def=None,
                 figure_dict=None,
                 keys: Optional[List[str]] = None,
                 sorting='descending'):
        """
        :param data_store: Source data store.
        :param name: Name of the FigureHandler instance that will be used in the figure defs table.
        :param source_data_name: Data name of relevant data records from the data store.
        :param transposed: Whether to display the table in transposed form (where each record will appear as a column for the k-th time step).
        :param keys: If specified, will only display these keys as rows or columns. Otherwise, this value will be set to the union of all keys for all records.

        TableHandler(data_store, 'my_table', 'data_source_table' ))
        """
        self.data_store = data_store
        self.name = name
        self.source_data_name = source_data_name
        self.figure_dict = figure_dict or go.Figure(layout_template=None).to_dict()
        self.decoded_data_def = decoded_data_def
        self.transposed = transposed
        self.header_kwargs = header_kwargs
        self.cells_kwargs = cells_kwargs
        self.keys = keys
        self.sorting = pgval.check_option('sorting', sorting, ['ascending', 'descending'])

    # Invalid methods.
    _parse_figure = None
    from_traces = None

    def build_table(self, *args, **kwargs): return self.build_figure(*args, **kwargs)

    def get_key(self, rec, key):
        if key not in rec:
            return ''
        else:
            val = rec[key]
            return val if isinstance(val, Number) else str(val)

    def build_figure(self, index=None):
        #
        if index is not None:
            raise Exception('Only index=None is supported.')
        #
        new_figure = deepcopy(self.figure_dict)

        # Retrieve data from the data store.
        raw_data = Ref_(self.source_data_name)(self.data_store)
        indices = raw_data['meta']['index']
        records = raw_data['data']

        if self.sorting == 'descending':
            indices = indices[::-1]
            records = records[::-1]

        # Get all keys
        # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order#:~:text=450-,Edit%202020,-As%20of%20CPython
        keys = list(self.keys or dict.fromkeys(it.chain(*(_x.keys() for _x in records))))

        # Build the columns of the table
        if self.transposed:
            # Each record is a column.
            columns = [
                {'name': _name, 'id': _name} for _name in
                it.chain(['Field'], indices)]
            # TODO: The first column is not styled.
            data = [{
                **{'Field': _key},
                **{_idx: _rec[_key] for _idx in indices}
                for _idx, _rec in strict_zip(indices, records)}]

            cells = {**self.cells_kwargs, 'values': columns}
            header = None
        else:
            columns = [{'name': name, 'id': id} for name in [self.time_step_key] + keys]
            # Var data will be a list of dicts containing one row.
            data = [{
                **{self.time_step_key: _idx},
                **{_key: self.get_key(_rec, _key) for _key in keys}} for
                _idx, _rec in strict_zip(indices, records)]

        # Build the trace object and add it to the figure.
        trace = go.Table(header=header, cells=cells)

        #
        new_figure = go.Figure(new_figure)
        new_figure.add_trace(trace)

        return new_figure

    def get_data_names(self):
        return self.source_data_name

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        params = {
            'source_data_name': self.source_data_name,
            'figure_dict': self.figure_dict,
            'transposed': self.transposed,
            'header_kwargs': self.header_kwargs,
            'cells_kwargs': self.cells_kwargs,
            'keys': self.keys,
            'sorting': self.sorting}
        return params

    @ property
    def is_indexed(self):
        return False
