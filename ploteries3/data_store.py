from sqlalchemy import (func, Table, Column, Integer, String, DateTime, select,
                        ForeignKey, LargeBinary, create_engine, MetaData)
from pglib.sqlalchemy import ClassType, JSONEncodedType
from contextlib import contextmanager
from pglib.sqlalchemy import begin_connection
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql import column as _c


class DataStore:
    def __init__(self, path, read_only=False):
        #
        if read_only:
            with open(path, 'r'):
                pass
            self.writer_id = None
        #
        self.path = path
        self.engine = create_engine(f'sqlite:///{path}')
        self._metadata = MetaData(bind=self.engine)

        #
        self._metadata.reflect()
        self._create_tables()

        # Set writer instance
        if not read_only:
            with self.engine.connect() as conn:
                self.writer_id = conn.execute(
                    self._metadata.tables['writers'].insert()).inserted_primary_key.id

    @contextmanager
    def begin_connection(self, connection=None):
        with begin_connection(self.engine, connection) as connection:
            yield connection

    def get_data_handlers(self, *column_constraints: BinaryExpression, connection=None):
        """
        Gets the data handlers satisfying the specified equality constraints. E.g., 

        ```
        from ploteries3.data_store import _c
        ```

        * ``get_data_handlers()`` returns all handlers,
        * ``get_data_handlers(_c('name')=='arr1')`` returns the data handler of name 'arr1',
        * ``get_data_handlers(data_store.data_defs_table.c.name=='arr1')`` returns the data handler of name 'arr1',
        * ``get_data_handlers(_c('type')==UniformNDArrayDataHandler)`` returns all data handlers of that type. (NOT WORKING!)

        """
        with self.begin_connection(connection) as connection:
            handlers = list((
                _rec.handler.from_def_record(self, _rec) for _rec in
                connection.execute(select(self.data_defs_table).where(*column_constraints))))
        return handlers

    def _create_tables(self):
        """
        Creates new tables or sets their column type.
        """

        self.data_records_table = Table(
            'data_records', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('index', Integer, nullable=False),
            Column('created', DateTime, server_default=func.now(), nullable=False),
            Column('writer_id', ForeignKey('writers.id'), nullable=False),
            Column('data_def_id', ForeignKey('data_defs.id'), nullable=False),
            Column('bytes', LargeBinary))

        # Distinguishes between writing form different DataStore instances.
        self.writers_table = Table(
            'writers', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('created', DateTime, server_default=func.now(), nullable=False))

        # Specifies how to retrieve and decode data bytes from the data_records table
        self.data_defs_table = Table(
            'data_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', JSONEncodedType, nullable=True))

        # Specifies figure creation from stored data.
        self.figure_defs_table = Table(
            'figure_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', JSONEncodedType))

        self._metadata.create_all()
