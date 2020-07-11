from sqlalchemy.engine import Engine
from sqlalchemy import event
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, \
    ForeignKey, types, insert, UniqueConstraint, func
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.serializer import loads, dumps
from sqlalchemy import exc
from pglib.sqlalchemy import NumpyType, PlotlyFigureType
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
import sqlite3
from ._sql_data_types import DataMapperType, sql_query_type_builder


RESERVED_TABLE_NAMES = ['__figures__', '__data_templates__']


def utc_now():
    return _dt.now(pytz.utc)


def sqlite3_concurrent_engine(path):
    # if ro: path+='?mode=ro'
    # connection = sqlite3.connect('file:' + path, uri=True, isolation_level=None)
    # sqlite3.connect('/tmp/wal.db', isolation_level=None)
    def creator():
        connection = sqlite3.connect(path, isolation_level=None)
        connection.execute('pragma journal_mode=wal;')
        return connection
    engine = create_engine('sqlite:///', creator=creator)
    return engine


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    #cursor.execute('PRAGMA synchronous=OFF')
    cursor.close()


class Writer(object):
    #
    def __init__(self, path, flush_sec=10):
        if osp.isdir(path):
            path = osp.join(path, utc_now().strftime(
                '%Y-%m-%d_%Hh%Mm%S.%f.sql'))
        self.path = path
        self.engine = create_engine('sqlite:///' + path)
        self._metadata = MetaData(bind=self.engine)
        self.SQLQueryType = sql_query_type_builder(
            scoped_session(sessionmaker(bind=self.engine)), self._metadata)
        self._init_tables()
        self._cache = {}
        self._last_flush = time.time()
        self.flush_sec = flush_sec

    # sqlalchemy helper functions
    # def _execute(self, cmd, *args, **
    #             kwargs): return self._connection.execute(cmd, *args, **kwargs)

    def execute(self, *args, **kwargs):
        with self.engine.begin() as conn:
            result = conn.execute(*args, **kwargs).fetchall()
        return result

    def _init_tables(self):

        # Create figures table
        self._figures = Table('__figures__', self._metadata,
                              Column('id', Integer, primary_key=True),
                              Column('tag', types.String,
                                     unique=True, nullable=False),
                              Column('figure', PlotlyFigureType, nullable=False))
        self._figures.create()

        # Create data templates table. Specifies replacements of the following type:
        # sql: is a query object with labeled columns (e.g., table.c.colname.label('field1')
        # data_mapper: [(<figure object slice sequence>, <sql output slice_sequence>)]
        self._data_templates = Table('__data_templates__', self._metadata,
                                     Column('id', Integer, primary_key=True),
                                     Column('figure_id', Integer, ForeignKey(
                                         '__figures__.id'), nullable=False),
                                     Column('sql', self.SQLQueryType,
                                            nullable=False),
                                     Column('data_mapper', DataMapperType,
                                            nullable=False))
        self._data_templates.create()

    def _create_table(self, name,  content_type, **kwargs):
        #
        if name in RESERVED_TABLE_NAMES:
            raise Exception(f"The specified name '{name}' is reserved!")
        #
        # content_type = {float: types.Float,
        #                np.ndarray: NumpyType}[content_type]
        content_type = {np.ndarray: NumpyType}[content_type]

        # Create table
        try:
            table = Table(name, self._metadata,
                          Column('id', Integer, primary_key=True),
                          Column('global_step', Integer, nullable=False),
                          Column('write_time', types.DateTime, nullable=False),
                          Column('content', content_type, nullable=True))  # Need nullable for NaN values
            table.create()
        except:  # TODO: Check if error is "table exists"
            raise

    def get_table(self, tag, content_type=None):
        try:
            table = self._metadata.tables[tag]
        except KeyError:
            if content_type is None:
                # Creation not requested
                raise
            self._create_table(tag, content_type)
            table = self._metadata.tables[tag]
        return table

    # Add content
    def add_data(self, name, content, global_step, write_time=None):
        # content_type = {np.ndarray: LargeBinary, # CB2
        # Cache results
        content_type = type(content)
        write_time = write_time or utc_now()
        self.get_table(name, content_type)  # Creates table if does not exist.
        self._cache.setdefault(name, []).append(
            dict(content=content, global_step=global_step, write_time=write_time))
        #
        if self.flush_sec is None or time.time()-self._last_flush > self.flush_sec:
            self.flush()

    # Register display
    def register_display(self, tag, figure, *sources):
        """
        Registers a plotly figure for display by storing the json representation of the figure in the table.

        tag: Figure name used in ploteries board.
        figure: Plotly figure object, with empty data placeholders.
        sources: Each source is a (sql query, [(figure slice sequence, sql query slice sequence), ...] ) tuple. 
        The sql query is an sql alchemy query object. Each slice sequence applies to a figure object, and the sql output, 
        respectively
        """

        # Insert figure
        with self.engine.begin() as conn:
            # TODO: Inconsistent state if failure after "insert figure".

            # Insert figure
            figure = conn.execute(
                self._figures.insert(), {'tag': tag, 'figure': figure})

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

    # All add methods
    # TODO

    # Base add methods.
    def add_figure(self, tag, figure, global_step, write_time=None):
        self._add_generic(tag, pst.FigureType, figure, global_step, write_time)

    def add_histogram(self, tag, dat, global_step, write_time=None, **kwargs):

        # Compute histogram.
        bin_centers, hist = self._compute_histogram(dat, **kwargs)
        self._plot_histograms(tag, (bin_centers, hist),
                              dat, global_step, write_time=write_time)

    @staticmethod
    def _compute_histogram(dat, bins=10, bin_centers=None, normalize=True):
        """
        'bins': Num bins or bin edges (passed to numpy.histogram to create a histogram).
        'bin_centers': Overrides 'bins' and specifies the bin centers instead of the edges.
            The first and last bin centers are assumed to extend to +/- infinity.
        """
        dat = numtor.asnumpy(dat)
        dat = dat.reshape(-1)

        # Infer bin edges
        if bin_centers is not None:
            bins = np.block(
                [-np.inf, np.convolve(bin_centers, [0.5, 0.5], mode='valid'), np.inf])

        # Build histogram
        hist, edges = np.histogram(dat, bins=bins)

        # Infer bin centers
        if bin_centers is None:
            bin_centers = np.convolve(edges, [0.5, 0.5], mode='valid')
            for k in [0, -1]:
                if not np.isfinite(bin_centers[k]):
                    bin_centers[k] = edges[k]

        # Normalize histogram
        if normalize:
            hist = hist/dat.size

        return bin_centers, hist

    def _plot_histogram(self, tag, dat, global_step, write_time=None):
        # Build figure
        bin_centers, hist = dat
        fig = px.line(x=bin_centers, y=hist)
        self._add_generic(tag, pst.HistogramType, fig, global_step, write_time)
