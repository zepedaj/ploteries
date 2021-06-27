"""
This module aims to simulate the tensorboard API so that ploteries can easily used in place of tensorboard in existing or new projects.
"""
import numpy as np
from numbers import Number
from sqlalchemy import exc
from .data_store import DataStore
from .ndarray_data_handlers import UniformNDArrayDataHandler
from .figure_handler import FigureHandler
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from typing import Optional, List, Dict
from pglib.py import SSQ


class Writer:

    def __init__(self, path):
        self.data_store = path if isinstance(path, DataStore) else DataStore(path)

    def add_scalars(self, tag: str, values: ArrayLike, global_step: int,
                    trace_args: Optional[List[Dict]] = None):
        """
        :param tag: The figure name. If specified in format '<tab>/<group>/...' , the tab and group entries will determine the position of the figure in the page.
        :param values: The values for each scalar trace as an array-like.
        :param names: The legend name to use for each trace.
        :param traces: List of length equal to that of values containing 1) the type of trace to use, and 2) the keyword arguments for the trace. The default value for each entry of the list is ``{'type':'scatter', 'mode':'lines'}``, and this default value will be updated with the corresponding input dictionary.

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
            data_name = f'_add_scalars.{tag}'

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
                trace_args = [
                    {'type': 'scatter', 'mode': 'lines', **_curr_trace_args}
                    for _curr_trace_args in (trace_args or [{}]*len(values))]

                # Create figure template
                figure = go.Figure()
                for _curr_trace_args in trace_args:
                    new_trace = getattr(go, _curr_trace_args.pop('type').capitalize())(
                        **_curr_trace_args)
                    figure.add_trace(new_trace)

                # Build data mappings
                mappings = []
                for k in range(len(values)):
                    mappings.extend([
                        # ('data', k, 'x'): (tag, 'meta', 'index'),
                        {'figure_keys': SSQ()['data'][k]['x'],
                         'source_keys': SSQ()[data_name]['meta']['index']},
                        # ('data', k, 'y'): (tag, 'data', (slice(None), k))
                        {'figure_keys': SSQ()['data'][k]['y'],
                         'source_keys': SSQ()[data_name]['data'][:, k]}
                    ])

                # Save figure.
                fig_handler = FigureHandler(
                    self.data_store,
                    name=tag,
                    sources={data_name: data_name},
                    mappings=mappings,
                    figure=figure)
                fig_handler.write_def(connection=connection)

            # Write the data data.
            data_name = f'_add_scalars.{tag}'
            data_handler = UniformNDArrayDataHandler(self.data_store, name=data_name)
            data_handler.add_data(global_step, values, connection=connection)

    def add_plots(self, tag: str, values: ArrayLike, global_step: int, names=None,
                  overwrite=False):

        values = [
            {'trace': {'type': 'scatter', 'name': 'plot 1', 'x': SSQ()[0, ::2], 'y': SSQ()[1, ::2]},
             'data': np.array([[10, 20, 30, 40], [20, 40, 60, 80]])},
            {'trace': {'type': 'bar', 'name': 'plot 2', 'x': SSQ()['f0'], 'y': SSQ()['f1']},
             'data': np.array([(0, 0.0), (1, 1.1)], dtype=[('f0', 'i'), ('f1', 'f')])}],
        data = np.array([(0, 0.0), (1, 1.1)], dtype=[('f0', 'i'), ('f1', 'f')])

    def add_figure(
            self, name: str, sources, traces: List[Dict],
            layout: Optional[Dict] = {},
            overwrite=False, trace_defaults={}):
        """
        Creates a figure with the given name and containing the specified traces.

        :param sources: Data source specification. Same syntax as :class:`~ploteries.figure_handler.FigureHandler`
        :param traces: Trace specification as dictionary. Fields in this dictionary that are :class:`SSQ` instances will be linked to the data source.

        Example:
        ```
        add_data('uniform', 'msft_stock', 0, np.array([0.0, 1.1, 2.2]))
        add_figure(
            traces = [
                {'type': 'scatter', 'name': 'plot 1',
                 'x': SSQ()['msft_stock']['data'][::2], 'y': SSQ()[1, ::2]},
                {'type': 'bar', 'name': 'plot 2', 'x': SSQ()['f0'], 'y': SSQ()['f1']}])
        ```
        """

        trace_defaults = {'type': 'scatter', **trace_defaults}

        # Build data mappings, remove SSQ objects from traces.
        mappings = []

        for k, _trace in enumerate(traces):
            mappings.extend([
                {'figure_keys': ('data', k, key), 'data_keys': ssq}
                for key, ssq in _trace.items() if isinstance(ssq, SSQ)])
            traces[k] = {**trace_defaults, **{
                key: val for key, val in _trace.items() if not isinstance(val, SSQ)}}

        # Build figure and traces, ensure type checking.
        figure = go.Figure()
        figure.update_layout(**layout)
        for trace in traces:
            figure.add_trace(getattr(go, trace.pop('type').capitalize())(**trace))

        # Add the figure definition to the data store.
        fh = FigureHandler(self.data_store, name, sources, mappings, figure)
        fh.write_def()
