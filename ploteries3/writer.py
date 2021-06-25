"""
This module aims to simulate the tensorboard API so that ploteries can easily used in place of tensorboard in existing or new projects.
"""
import numpy as np
from numbers import Number
from sqlalchemy import exc
from .data_store import DataStore
from .ndarray_data_handlers import UniformNDArrayDataHandler
from .figure_handlers import FigureHandler
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from typing import Optional, List, Dict
from pglib.py import SSQ


class Writer:
    _figures = {}

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
