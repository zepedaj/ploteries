from .reader import Reader
import numbers
from dateutil.tz import tzlocal
import re
from sqlalchemy import Table, Column, Integer, String, \
    ForeignKey, types, insert, UniqueConstraint, func, exc
from sqlalchemy.ext.serializer import loads, dumps
from pglib.sqlalchemy import NumpyType, begin_connection
from datetime import datetime as _dt
import pytz
import os.path as osp
import time
# import json
# import plotly
# import plotly.graph_objects as go
# from . import types as pst
from pglib.nnets import numtor
from pglib.py import SliceSequence
import numpy as np
# TODO: Why does removing this generate numpy warning?
from plotly import express as px
from . import figure_managers
from inspect import isclass


def utc_now():
    return _dt.now(pytz.utc)


class Writer(Reader):
    #
    def __init__(self, path, flush_sec=10):
        if osp.isdir(path):
            path = osp.join(path, utc_now().localize(tzlocal).strftime(
                '%Y-%m-%d_%Hh%Mm%S.%f.sql'))
        self.path = path
        super().__init__(path, check_exists=False)

        # self.engine = create_engine('sqlite:///' + path)
        # self._metadata = MetaData(bind=self.engine)

        self._cache = {}
        self._last_flush = time.time()
        self.flush_sec = flush_sec

    @classmethod
    def register_add_method(cls, name, method):
        setattr(cls, name,
                (lambda *args, method=method, **kwargs: method(*args, **kwargs)))

    @classmethod
    def register_module_add_classmethods(cls, module):
        """
        Add all add_* class methods in all .figure_managers.FigureManager-derived classes in the module.
        """
        for manager in [
                val for key, val in vars(module).items() if
                isclass(val) and issubclass(val, figure_managers.FigureManager)]:
            for name, method in [
                (name, method) for name, method in
                    vars(manager).items() if re.match('add_.*', name)]:
                cls.register_add_method(name, getattr(manager, name))

    def _init_headers(self):
        super()._init_headers()
        self._figures.create()
        self._data_templates.create()
        self._content_types.create()

    def _create_data_table(self, name,  content_type, connection=None):
        #
        if name in self.RESERVED_TABLE_NAMES:
            raise Exception(f"The specified name '{name}' is reserved!")

        #
        # content_type_mappings = [(numbers.Real, types.Float),
        #                          (np.ndarray, NumpyType)]
        # content_type = next(
        #     filter(lambda x: issubclass(content_type, x[0]), content_type_mappings))[1]
        content_type = {np.ndarray: NumpyType}[content_type]

        # Create table
        table = Table(name, self._metadata,
                      Column('id', Integer, primary_key=True),
                      Column('global_step', Integer, nullable=False),
                      Column('write_time', types.DateTime, nullable=False),
                      Column('content', content_type, nullable=True))  # Need nullable for NaN values

        with begin_connection(self.engine, connection) as conn:
            table.create(conn, checkfirst=False)
            conn.execute(self._content_types.insert(
                {'table_name': name, 'content_type': content_type}))

    def figure_exists(self, tag):
        """ 
        Checks if a figure with the given tag has been registered
        """
        num_fig_recs = len(self.execute(
            self._figures.select().where(self._figures.c.tag == tag)))
        if num_fig_recs == 0:
            return False
        elif num_fig_recs == 1:
            return True
        else:
            raise Excepction('Unexpected case!')

    def get_data_table(self, name, content_type=None):
        table = self._metadata.tables[name]
        return table

    # Add content
    def add_data(self, name, content, global_step, write_time=None, connection=None):
        # content_type = {np.ndarray: LargeBinary, # CB2
        # Cache results
        content_type = type(content)
        write_time = write_time or utc_now()

        new_data = dict(content=content, global_step=global_step,
                        write_time=write_time)
        if not name in self._metadata.tables.keys():
            # Table does not exist, create it and add data (atomically as part of potential parent transaction)
            with begin_connection(self.engine, connection) as conn:
                self._create_data_table(name, content_type, connection=conn)
                table = self.get_data_table(name)
                conn.execute(table.insert(new_data))
        else:
            # Table exists, cache data
            self._cache.setdefault(name, []).append(new_data)
            if self.flush_sec is None or time.time()-self._last_flush > self.flush_sec:
                self.flush()

    # Register display
    def register_display(self, tag, figure, manager, *sources, connection=None):
        """
        Registers a plotly figure for display by storing the json representation of the figure in the table.

        tag: Figure name used in ploteries board.
        figure: Plotly figure object, with empty data placeholders.
        manager: Manager used to load figure.
        sources: Each source is a (sql query, [(figure slice sequence, sql query slice sequence), ...] ) tuple.
        The sql query is an sql alchemy query object. Each slice sequence applies to a figure object, and the sql output,
        respectively
        """

        # Insert figure
        with begin_connection(self.engine, connection) as conn:
            # TODO: Inconsistent state if failure after "insert figure".

            # Insert figure
            figure = conn.execute(
                self._figures.insert(), {'tag': tag, 'figure': figure, 'manager': manager})

            # Insert data mapper
            for sql, data_mapper in sources:
                conn.execute(
                    self._data_templates.insert(),
                    {'figure_id': figure.inserted_primary_key[0], 'sql': (sql,), 'data_mapper': data_mapper})

    def __del__(self): self.flush()

    def flush(self):
        # CB2: Flush in background process or separate thread.
        _cache = self._cache
        self._cache = {}
        for table_name, records in _cache.items():
            table = self._metadata.tables[table_name]
            with self.engine.begin() as conn:
                conn.execute(table.insert(), records)
            # Get table
        self._last_flush = time.time()

    # # Base add methods.
    # def add_figure(self, tag, figure, global_step, write_time=None):
    #     self._add_generic(tag, pst.FigureType, figure, global_step, write_time)

    # def add_histogram(self, tag, dat, global_step, write_time=None, **kwargs):

    #     # Compute histogram.
    #     bin_centers, hist = self._compute_histogram(dat, **kwargs)
    #     self._plot_histograms(tag, (bin_centers, hist),
    #                           dat, global_step, write_time=write_time)

    # @ staticmethod
    # def _compute_histogram(dat, bins=10, bin_centers=None, normalize=True):
    #     """
    #     'bins': Num bins or bin edges (passed to numpy.histogram to create a histogram).
    #     'bin_centers': Overrides 'bins' and specifies the bin centers instead of the edges.
    #         The first and last bin centers are assumed to extend to +/- infinity.
    #     """
    #     dat = numtor.asnumpy(dat)
    #     dat = dat.reshape(-1)

    #     # Infer bin edges
    #     if bin_centers is not None:
    #         bins = np.block(
    #             [-np.inf, np.convolve(bin_centers, [0.5, 0.5], mode='valid'), np.inf])

    #     # Build histogram
    #     hist, edges = np.histogram(dat, bins=bins)

    #     # Infer bin centers
    #     if bin_centers is None:
    #         bin_centers = np.convolve(edges, [0.5, 0.5], mode='valid')
    #         for k in [0, -1]:
    #             if not np.isfinite(bin_centers[k]):
    #                 bin_centers[k] = edges[k]

    #     # Normalize histogram
    #     if normalize:
    #         hist = hist/dat.size

    #     return bin_centers, hist

    # def _plot_histogram(self, tag, dat, global_step, write_time=None):
    #     # Build figure
    #     bin_centers, hist = dat
    #     fig = px.line(x=bin_centers, y=hist)
    #     self._add_generic(tag, pst.HistogramType, fig, global_step, write_time)


# REGISTER ALL add_* methods in all FigureManager-derived classes in module figure_managers.
Writer.register_module_add_classmethods(figure_managers)
