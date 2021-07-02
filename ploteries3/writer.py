"""
This module aims to simulate the tensorboard API so that ploteries can easily used in place of tensorboard in existing or new projects.
"""
import numpy as np
from numbers import Number
from sqlalchemy import exc
from .data_store import DataStore, Ref_
from .ndarray_data_handlers import UniformNDArrayDataHandler
from .figure_handler import FigureHandler
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from typing import Optional, List, Dict
from pglib.slice_sequence import SSQ_


class Writer:

    _table_names = {
        'add_scalars': '__add_scalars__.{figure_name}'}

    def __init__(self, path):
        self.data_store = path if isinstance(path, DataStore) else DataStore(path)

    @classmethod
    def _get_table_name(cls, func, **kwargs):
        return cls._table_names[func].format(**kwargs)

    def add_scalars(self, figure_name: str, values: ArrayLike, global_step: int,
                    trace_args: Optional[List[Dict]] = None, data_name=None):
        """
        :param figure_name: The figure name. If specified in format '<tab>/<group>/...' , the tab and group entries will determine the position of the figure in the page.
        :param values: The values for each scalar trace as an array-like.
        :param names: The legend name to use for each trace.
        :param traces: None or list of dictionaries of length equal to that of values containing keyword arguments for the trace. The default value for each trace is ``{'type':'scatter', 'mode':'lines'}`` and will be updated with the specified values.

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
        default_trace_args = {'type': 'scatter', 'mode': 'lines'}

        # Get data name.
        data_name = data_name or self._get_table_name(
            'add_scalars', figure_name=figure_name)

        # Check values input
        values = np.require(values)
        if values.ndim != 1 or not isinstance(values[0], Number):
            raise ValueError('Expected a 1-dim array-like object.')

        # Check trace_args input
        if trace_args and len(trace_args) != len(values):
            raise ValueError(
                f'Param trace_args has {len(trace_args)} values, '
                f'but expected 0 or {len(values)}.')

        # Build default trace args.
        traces = [
            {'x': Ref_(data_name)['meta']['index'],
             'y': Ref_(data_name)['data'][:, k],
             **{**default_trace_args, **(_trace_args or {})}
             }
            for k, _trace_args in enumerate(
                (trace_args or [None]*len(values)))]

        # Create figure handler.
        fig_handler = FigureHandler.from_traces(
            self.data_store,
            name=figure_name,
            traces=traces)

        with self.data_store.begin_connection() as connection:
            # Write figure (if it does not exist)
            fig_handler.write_def(connection=connection)

            # Write data
            data_handler = UniformNDArrayDataHandler(self.data_store, name=data_name)
            data_handler.add_data(global_step, values, connection=connection)

    def add_plots(self, figure_name: str, values: ArrayLike, global_step: int, names=None,
                  overwrite=False):

        values = [
            {'trace': {'type': 'scatter', 'name': 'plot 1', 'x': SSQ_()[0, ::2], 'y': SSQ_()[1, ::2]},
             'data': np.array([[10, 20, 30, 40], [20, 40, 60, 80]])},
            {'trace': {'type': 'bar', 'name': 'plot 2', 'x': SSQ_()['f0'], 'y': SSQ_()['f1']},
             'data': np.array([(0, 0.0), (1, 1.1)], dtype=[('f0', 'i'), ('f1', 'f')])}],
        data = np.array([(0, 0.0), (1, 1.1)], dtype=[('f0', 'i'), ('f1', 'f')])
