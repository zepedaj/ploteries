from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, \
    ForeignKey, types, insert, UniqueConstraint
from sqlalchemy import exc
from datetime import datetime as dt
import os.path as osp, time
import json, plotly, plotly.graph_objects as go
from . import types as pst
from pglib.nnets import numtor
import numpy as np
from plotly import express as px
import sqlite3

def sqlite3_concurrent_engine(path):
    #if ro: path+='?mode=ro'
    #connection = sqlite3.connect('file:' + path, uri=True, isolation_level=None)
    #sqlite3.connect('/tmp/wal.db', isolation_level=None)
    def creator():
        connection = sqlite3.connect(path, isolation_level=None)
        connection.execute('pragma journal_mode=wal;')
        return connection
    engine = create_engine('sqlite:///', creator=creator)
    return engine

######
# Plot series are stored in tables.
# CB2: What is tensorboard's wall time for? Same as write time?

########
class Writer(object):
    #
    def __init__(self, path, flush_sec=10):
        if osp.isdir(path):
            path = osp.join(path, dt.now().strftime('%Y-%m-%d_%Hh%Mm%S.%f.sql'))
        self.path=path
        #self._engine = create_engine('sqlite:///' + path)
        self._engine = sqlite3_concurrent_engine(path)
        self._connection = self._engine.connect()
        self._metadata = MetaData(bind=self._engine)
        self._init_table_meta()
        self._cache={}
        self._last_flush=time.time()
        self.flush_sec=flush_sec

    # sqlalchemy helper functions
    def _execute(self, cmd, *args, **kwargs): return self._connection.execute(cmd, *args, **kwargs)
    def _init_table_meta(self):
        self._table_meta = Table('__ps__table_meta', self._metadata,
                                 Column('id', Integer, primary_key=True),
                                 Column('name', types.String, unique=True, nullable=False),
                                 Column('content_type', types.String, nullable=False),
                                 UniqueConstraint('name', 'content_type', name='uix_1'))
                                 #Column('display_vars', types.LargeBinary))        
        self._table_meta.create()

    def _create_table(self, name,  content_type, **kwargs):
        # Register table (might have been done previously, as oper is not atomic...)
        try:
            self._execute(self._table_meta.insert().values(name=name, content_type=content_type.__name__))
        except exc.IntegrityError as err:
            if err.args[0] != '(sqlite3.IntegrityError) UNIQUE constraint failed: __ps__table_meta.name':
                raise

        # Create table
        table = Table(name, self._metadata,
                  Column('id', Integer, primary_key=True),
                  Column('global_step', Integer, nullable=False),
                  Column('write_time', types.DateTime, nullable=False),
                  Column('content', content_type, nullable=True)) #Need nullable for NaN values
        table.create()

    def get_table(self, tag, content_type):
        try: 
            table =self._metadata.tables[tag]
        except KeyError:
            self._create_table(tag, content_type)
            table =self._metadata.tables[tag]
        return table

    # Add content
    def _add_generic(self, tag, content_type, content, global_step, write_time=None):
        # Cache results
        write_time = write_time or dt.now()
        self.get_table(tag, content_type) #creates table
        self._cache.setdefault(tag, []).append(
            dict(content=content, global_step=global_step, write_time=write_time))
        
        if self.flush_sec is None or time.time()-self._last_flush > self.flush_sec: self.flush()
        # Get table
        #table=self.get_table(tag, content_type)        
        # Insert record        
        #ins = table.insert().values(content=content, global_step=global_step, write_time=write_time)
        #self._execute(ins)

    def __del__(self): self.flush()

    def flush(self):
        #CB2: Flush in background process or separate thread.
        _cache = self._cache
        self._cache={}
        for table_name, records in _cache.items():
            table=self._metadata.tables[table_name]
            self._execute(table.insert(), records)
            # Get table        
        self._last_flush=time.time()

    def add_figure(self, tag, figure, global_step, write_time=None):
        self._add_generic(tag, pst.FigureType, figure, global_step, write_time)

    def add_scalar(self, tag, value, global_step, write_time=None):
        self._add_generic(tag, types.Float, value, global_step, write_time)

    def add_histogram(self, tag, dat, global_step, write_time=None, **kwargs):
        
        # Compute histogram.
        bin_centers, hist = self._compute_histogram(dat, **kwargs)
        self._plot_histograms(tag, (bin_centers, hist), dat, global_step, write_time=write_time)

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
            bins = np.block([-np.inf, np.convolve(bin_centers, [0.5,0.5], mode='valid'), np.inf])
            
        # Build histogram
        hist,edges=np.histogram(dat, bins=bins)
   
        # Infer bin centers
        if bin_centers is None:
            bin_centers = np.convolve(edges, [0.5,0.5], mode='valid')
            for k in [0,-1]: 
                if not np.isfinite(bin_centers[k]): bin_centers[k] = edges[k]

        # Normalize histogram
        if normalize:
            hist=hist/dat.size

        return bin_centers, hist
        
    def _plot_histogram(self, tag, dat, global_step, write_time=None):
        # Build figure
        bin_centers, hist = dat
        fig = px.line(x=bin_centers,y=hist)
        self._add_generic(tag, pst.HistogramType, fig, global_step, write_time)
