import sqlite3
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine, MetaData, event, Table, Column, Integer, String, \
    ForeignKey, types, insert, UniqueConstraint, func, exc, column, text, select
from sqlalchemy.engine import Engine
from .figure_managers import load_figure as figure_manager_load_figure
from pglib.sqlalchemy import PlotlyFigureType, ClassType, sql_query_type_builder
from ._sql_data_types import DataMapperType
import re
from sqlalchemy.engine.result import RowProxy


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


#
#event.listen(Table, "column_reflect", do_this_on_column_reflect)
# @event.listens_for(Table, "column_reflect")
def _reflect_custom_types(inspector, table, column_info, _content_types_table):

    #
    if table.name in Reader.RESERVED_TABLE_NAMES or column_info['name'] != 'content':
        return

    with inspector.engine.begin() as conn:
        content_type_recs = conn.execute(_content_types_table.select().where(
            _content_types_table.c.table_name == table.name)).fetchall()
        assert len(content_type_recs) == 1
        column_info['type'] = content_type_recs[0].content_type


class Reader(object):
    RESERVED_TABLE_NAMES = ['__figures__',
                            '__data_templates__', '__content_types__']
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
        #
        self._init_headers()
        event.listen(Table, "column_reflect",
                     lambda inspector, table, column_info, _content_types_table=self._content_types:
                     _reflect_custom_types(inspector, table, column_info, _content_types_table))
        self._metadata.reflect()

    def load_figure(self, *args, **kwargs):
        return figure_manager_load_figure(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        with self.engine.begin() as conn:
            result = conn.execute(*args, **kwargs).fetchall()
        return result

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

        # Specifies the content type for each data table
        self._content_types = Table('__content_types__', self._metadata,
                                    Column('id', Integer, primary_key=True),
                                    Column('table_name', String,
                                           nullable=False),
                                    Column('content_type', ClassType,
                                           nullable=False))

    def load_figure_recs(self, *args, **kwargs):
        """
        load_figure_recs(row_proxy) : Returns the input arg.
        load_figure_recs([tag=tag | id=id | manager=manager | name=name]) : Gets matching rows.
        """

        # Return input RowProxy
        if len(args) > 0:
            if len(args) == 1 and isinstance(args[0], RowProxy) and len(kwargs) == 0:
                return args
            else:
                raise Exception('Invalid input args.')

        # Get id from name
        if 'name' in kwargs:
            if 'id' in kwargs:
                Exception('Invalid input args.')
            kwargs['id'] = re.match(
                '^fig_(\d+)$', kwargs.pop('name')).groups()[0]

        # Add derived fields.
        query = select(
            [self._figures, ('fig_' + self._figures.c.id.cast(String)).label('name')])

        # Add query constraints.
        for key, val in kwargs.items():
            query = query.where(getattr(self._figures.c, key) == val)

        return self.execute(query)

        # # Map name to id
        # if name is not None:
        #     if id is not None:
        #         raise Exception('Invalid combimation of arguments.')
        #     id = int(re.match('^fig_(\d+)$', name).groups()[0])

        # # Build query
        # if tag is not None:
        #     query = query.where(self._figures.c.tag == tag)
        # if id is not None:
        #     query = query.where(self._figures.c.id == id)
        # if manager is not None:
        #     query = query.where(self._figures.c.manager == manager)
        # return self.execute(query)
