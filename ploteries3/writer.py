"""
This module aims to simulate the tensorboard API so that ploteries can easily used in place of tensorboard in existing or new projects.
"""
import numpy as np
from numbers import Number
from sqlalchemy import exc
from .data_store import DataStore, Ref_
from .ndarray_data_handlers import UniformNDArrayDataHandler
from .serializable_data_handler import SerializableDataHandler
from .figure_handler import FigureHandler
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from typing import Optional, List, Dict
from pglib.slice_sequence import SSQ_


class Writer:

    _table_names = {
        'add_scalars': '__add_scalars__.{figure_name}',
        'add_plots': '__add_plots__.{figure_name}'}
    default_trace_kwargs = {'type': 'scatter', 'mode': 'lines'}

    def __init__(self, path):
        self.data_store = path if isinstance(path, DataStore) else DataStore(path)

    @classmethod
    def _get_table_name(cls, func, **kwargs):
        return cls._table_names[func].format(**kwargs)

    def add_scalars(self, figure_name: str, values: ArrayLike, global_step: int,
                    traces_kwargs: Optional[List[Dict]] = None, data_name=None, layout_kwargs={}):
        """
        :param figure_name: The figure name. If specified in format '<tab>/<group>/...' , the tab and group entries will determine the position of the figure in the page.
        :param values: The values for each scalar trace as an array-like.
        :param names: The legend name to use for each trace.
        :param traces: None or list of dictionaries of length equal to that of values containing keyword arguments for the trace. The default value for each trace is ``{'type':'scatter', 'mode':'lines'}`` and will be updated with the specified values.
        :param data_name: The name of the data series when stored in the data table. If ``None``, the name ``__add_scalars__.<figure_name>`` will be used.

        Example:

        ```
        writer.add_scalar(
            'three_plots', [0.1, 0.4, 0.5],
            10,
            [{'type': 'scatter', 'name': 'trace 0'},
             {'name': 'trace 1'},
             {'type': 'bar', 'name': 'trace 2'}])
        ```
        """

        # Get data name.
        data_name = data_name or self._get_table_name(
            'add_scalars', figure_name=figure_name)

        # Check values input
        values = np.require(values)
        if values.ndim != 1 or not isinstance(values[0], Number):
            raise ValueError('Expected a 1-dim array-like object.')

        # Check traces_kwargs input
        if traces_kwargs and len(traces_kwargs) != len(values):
            raise ValueError(
                f'Param traces_kwargs has {len(traces_kwargs)} values, '
                f'but expected 0 or {len(values)}.')
        traces_kwargs = traces_kwargs or [{}]*len(values)

        # Build traces with data store references.
        traces = [
            {'x': Ref_(data_name)['meta']['index'],
             'y': Ref_(data_name)['data'][:, k],
             **_traces_kwargs}
            for k, _traces_kwargs in enumerate(traces_kwargs)]

        # Create figure handler.
        fig_handler = FigureHandler.from_traces(
            self.data_store,
            name=figure_name,
            traces=traces,
            default_trace_kwargs=self.default_trace_kwargs,
            layout_kwargs=layout_kwargs)

        with self.data_store.begin_connection() as connection:
            # Write figure (if it does not exist)
            fig_handler.write_def(connection=connection)

            # Write data
            data_handler = UniformNDArrayDataHandler(self.data_store, name=data_name)
            data_handler.add_data(global_step, values, connection=connection)

    def add_plots(
            self, figure_name: str,
            values: ArrayLike,
            global_step: int,
            traces_kwargs: Optional[List[Dict]] = None,
            data_name: Optional[List[str]] = None,
            layout_kwargs={}):
        """
        :param figure_name: (See :meth:`add_scalar`).
        :param values: The values for each scalar trace as a dictionary, e.g., ``[{'x': [0,2,4], 'y': [0,2,4]}, {'x': [0,2,4], 'y': [0,4,16]}]``. Dictionaries can contain lists, strings numpy ndarrays and generally anything compatible with :class:`~pglib.serializer.Serializer`.
        :param data_name: (See :meth:`add_scalar`).
        :param traces_kwargs: (See :meth:`add_scalar`).
        :param layout_kwargs: (See :meth:`add_scalar`).

        Example:

        ```
        writer.add_plots(
            'three_plots', [{'x': [0,2,4], 'y': [0,2,4]}, {'x': [0,2,4], 'y': [0,4,16]}],
            10,
            [{'type': 'scatter', 'name': 'trace 0'},
             {'name': 'trace 1'},
             {'type': 'bar', 'name': 'trace 2'}])
        ```
        """

        # Get data name.
        data_name = data_name or self._get_table_name(
            'add_plots', figure_name=figure_name)

        # Check traces_kwargs input
        if traces_kwargs and len(traces_kwargs) != len(values):
            raise ValueError(
                f'Param traces_kwargs has {len(traces_kwargs)} values, '
                f'but expected 0 or {len(values)}.')
        traces_kwargs = traces_kwargs or [{}]*len(values)

        # Build traces with data store references.
        traces = [
            {**{_key: Ref_(data_name, index='latest')['data'][0][k][_key] for _key in values[k]},
             **_traces_kwargs}
            for k, _traces_kwargs in enumerate(traces_kwargs)]

        # Create figure handler.
        fig_handler = FigureHandler.from_traces(
            self.data_store,
            name=figure_name,
            traces=traces,
            default_trace_kwargs=self.default_trace_kwargs,
            layout_kwargs=layout_kwargs)

        with self.data_store.begin_connection() as connection:
            # Write figure (if it does not exist)
            fig_handler.write_def(connection=connection)

            # Write data
            data_handler = SerializableDataHandler(
                self.data_store, name=data_name, connection=connection)
            data_handler.add_data(global_step, values, connection=connection)
