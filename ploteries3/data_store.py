from sqlalchemy import (
    Table, Column, Integer, String, DateTime, ForeignKey, LargeBinary, create_engine, MetaData)
from pglib.sqlalchemy import ClassType, JSONEncodedType


class DataStore:
    def __init__(self, path, writer_id=None, read_only=False):
        #
        if read_only:
            with open(path, 'r'):
                pass
        elif writer_id is None:
            raise ValueError('writer_id is None')
        #
        self.path = path
        self.writer_id = writer_id
        self.engine = create_engine(f'sqlite:///{path}')
        self._metadata = MetaData(bind=self.engine)

        #
        self._metadata.reflect()
        self._create_tables()

    def _create_tables(self):
        """
        Creates new tables or sets their column type.
        """

        data_records = Table(
            'data_records', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('index', Integer, nullable=False),
            Column('created', DateTime, nullable=False),  # AUTO? UTC?
            Column('writer_id', ForeignKey('writers.id'), nullable=False),
            Column('data_header_id', ForeignKey('data_headers.id'), nullable=False),
            Column('bytes', LargeBinary))

        # Distinguishes between writing form different DataStore instances.
        writers = Table(
            'writers', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('created', DateTime, nullable=False))   # AUTO? UTC?

        # Specifies how to retrieve and decode data bytes from the data_records table
        data_defs = Table(
            'data_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', JSONEncodedType))

        # Specifies figure creation from stored data.
        figure_defs = Table(
            'figure_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', JSONEncodedType))
