from copy import deepcopy
import plotly.graph_objects as go
from typing import List, Optional, Union, Dict, Any
from ploteries.data_store import Ref_
import itertools as it
from pglib import validation as pgval
from numbers import Number
from pglib.py import strict_zip
from .figure_handler import FigureHandler as _FigureHandler
from dash_table import DataTable


class TableHandler(_FigureHandler):
    """
    This object creates a table from dictionaries, where the dictionary at time step k fills the k-th row (or column) in the table. It is based on the `Dash DataTable <https://dash.plotly.com/datatable/reference>`_ class :class:`dash_table.DataTable`. *Note:* use class :class:`~ploteries.figure_handler.figure_handler.FigureHandler` to build `Plotly Tables <https://plotly.com/python/table/>`_ of type :class:`plotly.graphic_objects.Table`.
    """

    time_step_key = 'Time step'
    _state_keys = ['source_data_name', 'data_table_dict',
                   'transposed', 'keys', 'columns_kwargs', 'sorting']

    def __init__(self,
                 data_store,
                 name: str,
                 source_data_name: str,
                 transposed: bool = False,
                 data_table_dict=None,
                 keys: Optional[Union[List[str], Dict[str, Any]]] = None,
                 columns_kwargs: Dict[str, dict] = {},
                 sorting='descending',
                 decoded_data_def=None):
        """
        :param data_store: Source data store.
        :param name: Name of the FigureHandler instance that will be used in the figure defs table.
        :param source_data_name: Data name of relevant data records from the data store.
        :param transposed: Whether to display the table in transposed form (where each record will appear as a column for the k-th time step).
        :param keys: If specified, will only display these columns as rows or columns.
        :param column_kwargs: Dictionary mapping some or all record keys to extra kwargs to add to the :attr:`columns` argument to :class:`dash_table.DataTable` (see `DataTable reference <https://dash.plotly.com/datatable/reference>`_).
        :param data_table_dict: A dictionary specifying a :class:`dash.dcc.DataTable` object that will be used as the template to build the data table.
        """
        self.data_store = data_store
        self.name = name
        self.source_data_name = source_data_name
        self.data_table_dict = (
            (data_table_dict.to_plotly_json()
             if isinstance(data_table_dict, DataTable)
             else data_table_dict)
            or DataTable().to_plotly_json())
        self.decoded_data_def = decoded_data_def
        self.transposed = transposed
        self.keys = keys
        self.columns_kwargs = columns_kwargs
        self.sorting = pgval.check_option('sorting', sorting, ['ascending', 'descending'])

    # Invalid methods.
    _parse_figure = None
    from_traces = None

    def build_table(self, *args, **kwargs): return self.build_figure(*args, **kwargs)

    def _get_key(self, rec, key):
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
            indices = indices.tolist()
            columns = [
                {'name': _name, 'id': _name} for _name in
                it.chain(['Field'], indices)]
            data = (
                [{**{'Field': self.time_step_key},
                  **{_idx: _idx for _idx in indices}}] +
                [{**{'Field': _key},
                  **{_idx: self._get_key(_rec, _key) for _idx, _rec in strict_zip(
                      indices, records)}} for _key in keys])

        else:
            columns = [{'name': name, 'id': name} for name in [self.time_step_key] + keys]
            # Var data will be a list of dicts containing one row.
            data = [{
                **{self.time_step_key: _idx},
                **{_key: self._get_key(_rec, _key) for _key in keys}} for
                _idx, _rec in strict_zip(indices, records)]

        # Apply column styling.
        [_col.update(self.column_kwargs.get(_col['id'], {})) for _col in self.columns_kwargs]

        # Build the DataTable object.
        data_table = DataTable(**{
            **self.data_table_dict['props'],
            **{'columns': columns, 'data': data}
        })

        return data_table

    def get_data_names(self):
        return self.source_data_name

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        params = {key: getattr(self, key) for key in self._state_keys}
        return params

    @ property
    def is_indexed(self):
        return False
