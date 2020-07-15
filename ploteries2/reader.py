import sqlite3
import warnings
from collections import deque
import sqlalchemy as sqa
from sqlalchemy.sql.expression import alias
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine, MetaData, event, Table, Column, Integer, String, \
    ForeignKey, types, insert, UniqueConstraint, func, exc, column, text, select
from sqlalchemy.engine import Engine
from .figure_managers import load_figure as figure_manager_load_figure, global_steps as figure_manager_global_steps
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
# event.listen(Table, "column_reflect", do_this_on_column_reflect)
# @event.listens_for(Table, "column_reflect")
def _reflect_custom_types(inspector, table, column_info, _content_types_table):

    #
    if table.name in Reader.RESERVED_TABLE_NAMES or column_info['name'] != 'content':
        return

    with inspector.engine.begin() as conn:
        content_type_recs = conn.execute(_content_types_table.select().where(
            _content_types_table.c.table_name == table.name)).fetchall()
        if len(content_type_recs) == 1:
            column_info['type'] = content_type_recs[0].content_type
        elif len(content_type_recs) > 1:
            raise Exception('Unexpected case.')
        else:
            warnings.warn(
                f'Found orphan table `{table.name}` without entry in content types table. '
                'Possibly due to sqlite\'s non-transactional table creation that prevents atomicity '
                'of `register_figure` functions.')


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

    def get_data_table(self, name, content_type=None):
        table = self._metadata.tables[name]
        return table

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

    def global_steps(self, *args, **kwargs):
        """
        Returns all global steps associted to a given figure.
        """
        return figure_manager_global_steps(self, *args, **kwargs)

    def figure_exists(self, tag, expected={}):
        """
        Checks if a figure with the given tag has been registered and returns the 
        RowProxy record (casts to True) or None (casts to False).
        tag: Figure tag string.
        expected: If a figure record is found, these values will be checked against record values and 
            an error will be raised if these do not match. Can be used, e.g., to check an expected
            manager class.
        """
        fig_recs = self.execute(
            self._figures.select().where(self._figures.c.tag == tag))
        if len(fig_recs) == 0:
            out = None
        elif len(fig_recs) == 1:
            out = fig_recs[0]
            if not all([getattr(out, key) == expected[key] for key in expected.keys()]):
                raise Exception(
                    f'Retrieved figure record {out} does not match expected values {expected}.')
        else:
            raise Exception('Unexpected case!')

        return out

    def load_figure_recs(self, *args, check_single=False, **kwargs):
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

        # Execute query
        figure_recs = self.execute(query)

        # Check single
        if check_single and len(figure_recs) != 1:
            raise Exception(
                'Did not find exactly figure record matched the query specification.')

        return figure_recs

    @classmethod
    def join_data_tables(cls, tables, **join_kwargs):
        """
        Builds an sql query to create a (inner by default) join (on global_step) of the input tables.

        Use isouter=True for left-outer join, and cls.outer_join_data_tables for full outer join.

        tables: list of table or list of (table, content_field_name)-tuples.
        """

        if isinstance(tables[0], (tuple, list)):
            content_fields = [_fld for _tbl, _fld in tables]
            tables = [_tbl for _tbl, _fld in tables]
        else:
            content_fields = [f'content{_k}' for _k in range(len(tables))]

        # Build sql outer joins across all tables, keep all global_step values, even when not present in all tables.
        qry = sqa.select(
            [tables[0].c.global_step, tables[0].c.content.label(content_fields[0])])
        for k, curr_table in zip(range(1, len(tables)), tables[1:]):
            qry = alias(qry)
            qry = sqa.select(
                # Always is left outer or inner join, so global_step always non-null
                [qry.c.global_step.label('global_step')] +
                # [func.ifnull(qry.c.global_step, curr_table.c.global_step).label('global_step')] +
                [getattr(qry.c, content_fields[_l]) for _l in range(k)] +
                [curr_table.c.content.label(content_fields[k])]).select_from(
                    qry.join(curr_table, curr_table.c.global_step == qry.c.global_step, **join_kwargs))

        return qry

    @classmethod
    def outer_join_data_tables(cls, tables):
        """
        Builds an sql query to create an full outer join (on global_step) of the input tables. 
        Each row in each table is guaranteed ot appear once in the output.

        tables: list of table or list of (table, content_field_name)-tuples.
        """

        # Add content fields
        if not isinstance(tables[0], (tuple, list)):
            tables = [(_tbl, f'content{_k}') for _k, _tbl in enumerate(tables)]

        # For each table, do left join with all tables
        dq_tables = deque(tables)
        queries = []
        for k in range(len(tables)):
            queries.append(cls.join_data_tables(list(dq_tables), isouter=True))
            dq_tables.rotate(1)

        # Order fields to same order.
        field_order = queries[0].c.keys()
        queries = [
            sqa.select([getattr(_qry.c, _fld) for _fld in field_order])
            for _qry in queries]
        qry = sqa.union(*queries)

        return qry
