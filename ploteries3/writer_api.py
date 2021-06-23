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
from pglib.py import SSQ


class Writer:
    _figures = {}

    def __init__(self, path):
        self.data_store = path if isinstance(path, DataStore) else DataStore(path)

    def add_scalars(self, tag: str, values: ArrayLike, global_step: int, names=None,
                    overwrite=False):
        """
        :param tag: The figure name. If specified in format '<tab>/<group>/...' , the tab and group entries will determine the position of the figure in the page.
        :param values: The values for each scalar trace as an array-like.
        :param names: The legend name to use for each trace.
        :param overwrite: Overwrite the previously-created figure definition (e.g., to replace legend names).
        """

        with self.data_store.begin_connection() as connection:

            # Check input
            values = np.require(values)
            if values.ndim != 1 or not isinstance(values[0], Number):
                raise ValueError('Expected a 1-dim array-like object.')

            # Add data.
            data_handler = UniformNDArrayDataHandler(self.data_store, name=tag)
            data_handler.add_data(global_step, values)

            # Check if the figure exists.
            if not overwrite:
                try:
                    FigureHandler.from_name(self.data_store, tag)
                    figure_exists = True
                except exc.NoResultFound:
                    figure_exists = False

            # Write the figure
            if overwrite or not figure_exists:
                # Create figure template
                figure = go.Figure()
                for k in range(len(values)):
                    figure.add_trace(go.Line(
                        x=[], y=[], name=(None if not names else names[k])))

                # Build data mappings
                data_mappings = {}
                for k in range(len(values)):
                    data_mappings.update({
                        # ('data', k, 'x'): (tag, 'meta', 'index'),
                        SSQ()['data'][k]['x']: SSQ()[tag]['meta']['index'],
                        # ('data', k, 'y'): (tag, 'data', (slice(None), k))
                        SSQ()['data'][k]['y']: SSQ()[tag]['data'][:, k]
                    })

                # Save figure.
                fig_handler = FigureHandler(
                    self.data_store,
                    name=tag,
                    sources={tag: tag},
                    data_mappings=data_mappings,
                    figure=figure)
                fig_handler._write_def()
