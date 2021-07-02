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
        'add_scalars': '__add_scalars__.{tag}'}

    def __init__(self, path):
        self.data_store = path if isinstance(path, DataStore) else DataStore(path)

    @classmethod
    def _get_table_name(cls, func, **kwargs):
        return cls._table_names[func].format(**kwargs)

    def add_scalars(self, tag: str, values: ArrayLike, global_step: int,
                    trace_args: Optional[List[Dict]] = None):
        """
        :param tag: The figure name. If specified in format '<tab>/<group>/...' , the tab and group entries will determine the position of the figure in the page.
        :param values: The values for each scalar trace as an array-like.
        :param names: The legend name to use for each trace.
        :param traces: List of length equal to that of values containing keyword arguments for the trace. The default value for unspecified valuew (None) is ``{'type':'scatter', 'mode':'lines'}``.

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

        with self.data_store.begin_connection() as connection:

            #
            data_series_name = self._get_table_name('add_scalars', tag=tag)

            # Check values input
            values = np.require(values)
            if values.ndim != 1 or not isinstance(values[0], Number):
                raise ValueError('Expected a 1-dim array-like object.')

            # Check if the figure exists.
            try:
                fig_handler = FigureHandler.from_name(self.data_store, tag)
                figure_exists = True
            except exc.NoResultFound:
                figure_exists = False

            # Write the figure
            if not figure_exists:

                # Check trace_args input
                if trace_args and len(trace_args) != len(values):
                    raise ValueError(
                        f'Param trace_args has {len(trace_args)} values, '
                        f'but expected 0 or {len(values)}.')

                # Build default trace args.
                traces = [
                    {'x': Ref_(data_series_name)['meta']['index'],
                     'y': Ref_(data_series_name)['data'][:, k],
                     **(_trace_args or {'type': 'scatter', 'mode': 'lines'})}
                    for k, _trace_args in enumerate(
                        (trace_args or [None]*len(values)))]

                # Create figure and append traces
                figure = go.Figure(layout_template=None).to_dict()
                figure['data'].extend(traces)

                # Save figure.
                fig_handler = FigureHandler(
                    self.data_store,
                    name=tag,
                    figure_dict=figure)
                fig_handler.write_def(connection=connection)

            # Write the data data.
            data_handler = UniformNDArrayDataHandler(self.data_store, name=data_series_name)
            data_handler.add_data(global_step, values, connection=connection)

    def add_plots(self, tag: str, values: ArrayLike, global_step: int, names=None,
                  overwrite=False):

        values = [
            {'trace': {'type': 'scatter', 'name': 'plot 1', 'x': SSQ_()[0, ::2], 'y': SSQ_()[1, ::2]},
             'data': np.array([[10, 20, 30, 40], [20, 40, 60, 80]])},
            {'trace': {'type': 'bar', 'name': 'plot 2', 'x': SSQ_()['f0'], 'y': SSQ_()['f1']},
             'data': np.array([(0, 0.0), (1, 1.1)], dtype=[('f0', 'i'), ('f1', 'f')])}],
        data = np.array([(0, 0.0), (1, 1.1)], dtype=[('f0', 'i'), ('f1', 'f')])
