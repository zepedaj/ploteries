import sqlite3
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine, MetaData, event, Table, Column, Integer, String, \
    ForeignKey, types, insert, UniqueConstraint, func, exc, column, text, select
from sqlalchemy.engine import Engine
from .figure_managers import load_figure
from pglib.sqlalchemy import PlotlyFigureType, ClassType, sql_query_type_builder
from ._sql_data_types import DataMapperType
import re


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
    # cursor.execute('PRAGMA synchronous=OFF')
    cursor.close()


class Reader(object):
    #
    def __init__(self, path, check_exists=True):
        #
        if check_exists:
            with open(path, 'r'):
                pass
        #
        self.path = path
        self.engine = create_engine('sqlite:///' + path)
        self._metadata = MetaData(bind=self.engine)
        self.SQLQueryType = sql_query_type_builder(
            scoped_session(sessionmaker(bind=self.engine)), self._metadata)
        self._init_headers()

    def load_figure(self, *args, **kwargs):
        return load_figure(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        with self.engine.begin() as conn:
            result = conn.execute(*args, **kwargs).fetchall()
        return result

    def tabs(self):
        self.execute()

    def _init_headers(self):
        self._figures = Table('__figures__', self._metadata,
                              Column('id', Integer, primary_key=True),
                              Column('tag', types.String,
                                     unique=True, nullable=False),
                              Column('manager', ClassType, nullable=False),
                              Column('figure', PlotlyFigureType, nullable=False))

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

    def get_figure_recs(self, tag=None, id=None, manager=None, name=None):
        # Add derived fields.
        query = select(
            [self._figures, ('fig_' + self._figures.c.id.cast(String)).label('name')])

        # Map name to id
        if name is not None:
            if id is not None:
                raise Exception('Invalid combimation of arguments.')
            id = int(re.match('^fig_(\d+)$', name).groups()[0])

        # Build query
        if tag is not None:
            query = query.where(self._figures.c.tag == tag)
        if id is not None:
            query = query.where(self._figures.c.id == id)
        if manager is not None:
            query = query.where(self._figures.c.manager == manager)
        return self.execute(query)
