from sqlalchemy import (func, Table, Column, Integer, String, DateTime,
                        ForeignKey, LargeBinary, create_engine, MetaData)
from pglib.sqlalchemy import ClassType, JSONEncodedType


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

    def _create_tables(self):
        """
        Creates new tables or sets their column type.
        """

        data_records = Table(
            'data_records', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('index', Integer, nullable=False),
            Column('created', DateTime, server_default=func.now(), nullable=False),  # AUTO? UTC?
            Column('writer_id', ForeignKey('writers.id'), nullable=False),
            Column('data_def_id', ForeignKey('data_defs.id'), nullable=False),
            Column('bytes', LargeBinary))

        # Distinguishes between writing form different DataStore instances.
        writers = Table(
            'writers', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('created', DateTime, server_default=func.now(), nullable=False))   # AUTO? UTC?

        # Specifies how to retrieve and decode data bytes from the data_records table
        data_defs = Table(
            'data_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', JSONEncodedType, nullable=True))

        # Specifies figure creation from stored data.
        figure_defs = Table(
            'figure_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', JSONEncodedType))

        self._metadata.create_all()
