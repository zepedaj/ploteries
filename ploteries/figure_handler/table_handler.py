from copy import deepcopy
from jztools.validation import checked_get_single
from jztools.slice_sequence import SSQ_
import plotly.graph_objects as go
from typing import List, Optional, Union, Dict, Any, Tuple
from ploteries.data_store import Ref_
import itertools as it
from collections import OrderedDict
from jztools import validation as pgval
from numbers import Number
from jztools.py import strict_zip
from .figure_handler import FigureHandler as _FigureHandler
from dash.dash_table import DataTable
import re


def markdown_escape(val, level=0):
    if isinstance(val, str):
        return re.sub(r"([\[\]\(\)\\`\*_\{\}<>#+-\.!|])", r"\\\1", val)
    elif isinstance(val, dict):
        tab = "\t"
        return "\n" * (level > 0) + "\n".join(
            [
                f"{tab*level}* **{markdown_escape(_key, level+1)}**: {markdown_escape(_value,level+1)}"
                for _key, _value in val.items()
            ]
        )
    else:
        return val


class TableHandler(_FigureHandler):
    """
    This object creates a table from one or more data series, with one row (or column when ``transposed=True``) per time step, and the fields specified by data dictionaries derived from the data series. The simplest use case is to display data from a :class:`ploteries.serializable_data_handler.SerializableDataHandler` data series.

    The display uses the `Dash DataTable <https://dash.plotly.com/datatable/reference>`_ class :class:`dash_table.DataTable`. *Note:* use class :class:`~ploteries.figure_handler.figure_handler.FigureHandler` to build `Plotly Tables <https://plotly.com/python/table/>`_ of type :class:`plotly.graphic_objects.Table`.

    .. ipython::

        In [304]: print('Hello world')
        Hello world

        In [305]:
    """

    time_step_key = "Time step"
    _state_keys = [
        "data_mappings",
        "data_table_template",
        "transposed",
        "columns_kwargs",
        "sorting",
    ]

    def __init__(
        self,
        data_store,
        name: str,
        data_mappings: Union[Tuple[str, SSQ_], Dict[str, Tuple[str, SSQ_]]],
        transposed: bool = False,
        data_table_template=None,
        columns_kwargs: Dict[str, dict] = {},
        sorting="ascending",
        decoded_data_def=None,
    ):
        """
        :param data_store: Source data store.
        :param name: Name of the FigureHandler instance that will be used in the figure defs table.
        :param data_mappings: A data series name /  :class:`jztools.slice_sequence.SSQ_`-producible pair, or a dictionary of such pairs. In the first case, the :class:`SSQ_` object is expected to produce a dictionary. In both cases, the :class:`SSQ_` objects are applied to each record retrieved from the corresponding data series (i.e., to each entry of ``data_store['data_series_name']['data']``), and the dictionary keys will be the :attr:`id` for the :attr:`columns` argument of the :class:`dash_table.DataStore` object (as well as the :attr:`name` by default, unless overwritten with argument :attr:`columns_kwargs`).
        :param data_table_template: A :class:`dash.dcc.DataTable` or derived dictionary specifying the template used to build the data table.
        :param transposed: Whether to display the table in transposed form (where each record will appear as a column for the k-th time step).
        :param sorting: Whether to display the data fields in ascending or descending order.
        :param columns_kwargs: Dictionary mapping some or all :attr:`data` keys to extra kwargs to add to the corresponding :class:`dash_table.DataTable` :attr:`columns` argument (see `DataTable reference <https://dash.plotly.com/datatable/reference>`_).
        """
        self.data_store = data_store
        self.name = name
        try:
            if isinstance(data_mappings, dict):
                self.data_mappings = {
                    col_id: (_ds_name, SSQ_.produce(_ssq))
                    for col_id, (_ds_name, _ssq) in data_mappings
                }
            else:
                _ds_name, _ssq = data_mappings
                self.data_mappings = (_ds_name, SSQ_.produce(_ssq))
        except Exception:
            print("**************", data_mappings)
            raise Exception("Invalid format for arg data_mappings.")

        self.data_table_template = (
            data_table_template.to_plotly_json()
            if isinstance(data_table_template, DataTable)
            else data_table_template
        ) or DataTable().to_plotly_json()
        self.decoded_data_def = decoded_data_def
        self.transposed = transposed
        self.columns_kwargs = columns_kwargs
        self.sorting = pgval.check_option(
            "sorting", sorting, ["ascending", "descending"]
        )

    # Invalid methods.
    _parse_figure = None
    from_traces = None

    def _get_key(self, rec, key):
        if key not in rec:
            return ""
        else:
            val = rec[key]

            return val  # if isinstance(val, Number) else markdown_escape(str(val))

    def build_table(self, slice_obj=slice(None, None, None)):

        # Retrieve a join of all data series.
        data_series_names = (
            list(set(_x[0] for _x in self.data_mappings.values()))
            if isinstance(self.data_mappings, dict)
            else [self.data_mappings[0]]
        )
        raw_data = self.data_store[data_series_names]

        # Extract and sort data
        indices = raw_data["meta"]["index"][slice_obj]
        data_series = {
            _ds_name: _ds_contents["data"][slice_obj]
            for _ds_name, _ds_contents in raw_data["series"].items()
        }
        if self.sorting == "descending":
            indices = indices[::-1]
            data_series = {_key: _data[::-1] for _key, _data in data_series.items()}

        # Get all keys
        if isinstance(self.data_mappings, dict):
            keys = list(self.data_mappings.keys())
            records = [
                {
                    _key: _ref(data_series[_ds_name][_k])
                    for _key, (_ds_name, _ref) in self.data_mappings.items()
                }
                for _k in range(len(indices))
            ]
        else:
            data_series_name, ref = self.data_mappings
            records = [
                ref(_rec) for _rec in checked_get_single(data_series, data_series_name)
            ]
            # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order#:~:text=450-,Edit%202020,-As%20of%20CPython
            keys = list(
                OrderedDict.fromkeys(it.chain(*(_rec.keys() for _rec in records)))
            )

        # Build the columns of the table
        if self.transposed:
            # Each record is a column.
            indices = indices.tolist()
            columns = [
                {"name": _name, "id": _name, "presentation": "markdown"}
                for _name in it.chain(["Field"], indices)
            ]
            data = [
                {
                    **{"Field": f"**{markdown_escape(self.time_step_key)}**"},
                    **{_idx: _idx for _idx in indices},
                }
            ] + [
                {
                    **{"Field": f"**{markdown_escape(_key)}**"},
                    **{
                        _idx: markdown_escape(self._get_key(_rec, _key))
                        for _idx, _rec in strict_zip(indices, records)
                    },
                }
                for _key in keys
            ]

        else:
            columns = [
                {"name": name, "id": name, "presentation": "markdown"}
                for name in [self.time_step_key] + keys
            ]
            # Var data will be a list of dicts containing one row.
            data = [
                {
                    **{self.time_step_key: _idx},
                    **{
                        _key: markdown_escape(self._get_key(_rec, _key))
                        for _key in keys
                    },
                }
                for _idx, _rec in strict_zip(indices, records)
            ]

        # Apply column styling.
        [
            _col.update(self.column_kwargs.get(_col["id"], {}))
            for _col in self.columns_kwargs
        ]

        # Build the DataTable object.
        data_table = DataTable(
            **{
                **self.data_table_template["props"],
                **{"columns": columns, "data": data},
            }
        )

        return data_table

    def get_data_names(self):
        return self.source_data_name

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        params = {key: getattr(self, key) for key in self._state_keys}
        return params

    @property
    def is_indexed(self):
        return False
