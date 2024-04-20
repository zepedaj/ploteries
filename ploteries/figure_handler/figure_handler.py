import plotly.graph_objects as go
import itertools as it
from sqlalchemy import exc
from jztools.profiling import time_and_print
from ploteries.base_handlers import Handler
from typing import List, Optional
from typing import Dict, Union, Any
from jztools.slice_sequence import SSQ_
from dataclasses import dataclass
from copy import deepcopy
from ploteries.data_store import Col_, Ref_
from jztools.py import get_nested_keys


class FigureHandler(Handler):
    data_store = None
    decoded_data_def = None

    def __init__(self, data_store, name: str, figure_dict, decoded_data_def=None):
        """
        Instantiates a new figure handler. Note that this does not read or write a figure handler from the data store (use methos :meth:`from_name` and :meth:`write_def` for this purpose).

        :param figure_dict: Dictionary representation of a plotly figure containing placeholder values of type :class:`ploteries.data_store.Ref_` that will be replaced by data from the store when building the figure.

        .. todo:: Example.

        """
        self.data_store = data_store
        self.name = name
        self.figure_dict = figure_dict
        self._parse_figure()
        self.decoded_data_def = decoded_data_def

    def _parse_figure(self):
        # Get all placeholder positions and values.
        self.figure_keys = [
            SSQ_.produce(_keys)
            for _keys in get_nested_keys(
                self.figure_dict, lambda x: isinstance(x, Ref_)
            )
        ]
        self.data_keys = [_keys(self.figure_dict) for _keys in self.figure_keys]

    def build_figure(self, index=None):
        #
        new_figure = deepcopy(self.figure_dict)

        # Set new index if specified
        if index is not None:
            data_keys = [_data_key.copy() for _data_key in self.data_keys]
            for _data_key in data_keys:
                if _data_key.query_params["index"] == "latest":
                    _data_key.query_params["index"] = index
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

    def get_data_names(self):
        """
        Returns a list of all unique data names that this figure is dependent on.
        """
        all_data_names = list(
            set(it.chain((_dk.slice_sequence[0]["data"] for _dk in self.data_keys)))
        )
        return all_data_names

    @classmethod
    def from_def_record(cls, data_store, data_def_record):
        # cls.decode_params(data_def_record)
        return cls(
            data_store,
            data_def_record.name,
            decoded_data_def=data_def_record,
            **data_def_record.params
        )

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
        params = {"figure_dict": self.figure_dict}
        return params

    @property
    def is_indexed(self):
        """
        Specifies whether the figure depends on a single time index.
        """
        return any(
            (
                _data_key.query_params["index"] == "latest"
                for _data_key in self.data_keys
            )
        )

    # Constructors

    @classmethod
    def from_traces(
        cls,
        data_store,
        name: str,
        traces: List[Dict],
        default_trace_kwargs={},
        layout_kwargs={},
        connection=None,
    ):
        """
        :param traces: List of traces as dictionaries, potentially containing :class:`~ploteries.data_store.Ref_` references.
        :param default_trace_kwargs: Default trace kwargs applied to input param :attr:`traces`.
        :param layout_kwargs: Keyword args passed to :meth:`go.Figure.update_layout`.
        """

        # Build default trace args.
        traces = [{**default_trace_kwargs, **_trace} for _trace in traces]

        # Create figure and append traces
        fig = go.Figure(layout_template=None)
        fig.update_layout(**layout_kwargs)
        fig_dict = fig.to_dict()
        fig_dict["data"].extend(traces)

        # Save figure.
        fig_handler = cls(data_store, name=name, figure_dict=fig_dict)

        return fig_handler
