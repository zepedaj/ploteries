import abc
from contextlib import contextmanager
from pglib.sqlalchemy import begin_connection
import re
from sqlalchemy.engine.result import Row
from sqlalchemy import insert, func, select, exc
from typing import Union
import numpy as np


class DataHandler(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def from_record(self, record):
        """
        Initializes a data handler object from a record from the data store's :attr:`data_defs` table.
        """

    @property
    @abc.abstractmethod
    def data_store(self):
        """
        Contains a :class:`~ploteries3.data_store.DataStore` object.
        """

    @property
    @abc.abstractmethod
    def _data_def(self):
        """
        Contains the Row retrieved from the data_defs table
        """

    @property
    def data_defs_table(self):
        return self.data_store._metadata.tables['data_defs']

    @property
    def data_records_table(self):
        return self.data_store._metadata.tables['data_records']

    @contextmanager
    def begin_connection(self, connection=None):
        with begin_connection(self.data_store.engine, connection) as connection:
            yield connection

    @classmethod
    def decode_params(self, params):
        """
        In-place decoding of the the params field of the data_defs record.
        """

    def encode_params(self):
        """
        Produces the params field to place in the data_defs record.
        """
        return None

    def _load_def(self, connection=None, check=True) -> Union[bool, Row]:
        """
        Loads and decodes the definition from the the data defs table, if it exists, returning
        the decoded data def row if successful.

        Returns False if no data def is found in the table, or the data def row if it is found.
        """

        with self.begin_connection(connection) as conn:
            # Build query
            qry = self.data_defs_table.select().where(self.data_defs_table.c.name == self.name)

            # Check that the store specs match the input.
            data_def = conn.execute(qry).fetchall()

        if len(data_def) == 0:
            return False
        elif len(data_def) == 1:
            data_def = data_def[0]
            if check:
                if not isinstance(self, data_def.handler):
                    raise TypeError(
                        f'The data definition handler {data_def.handler} '
                        f'does not match the current handler {type(self)}.')
            self.decode_params(data_def.params)
            return data_def
        else:
            raise Exception('Unexpected case.')

    def _write_def(self, connection=None):
        """
        Adds an entry to the data defs table and returns True if successful. If an entry already exists, returns False.
        """

        record_dict = {
            'name': self.name,
            'handler': type(self),
            'params': self.encode_params()
        }

        with self.begin_connection(connection) as connection:
            try:
                connection.execute(
                    insert(self.data_defs_table), record_dict)
            except exc.IntegrityError as err:
                if re.match(
                        f'\\(sqlite3.IntegrityError\\) UNIQUE constraint failed\\: {self.data_defs_table.name}\\.name',
                        str(err)):
                    return False
                else:
                    raise

            else:
                return True

    @abc.abstractmethod
    def encode_record_bytes(self, record_data) -> bytes:
        """
        Encodes the record's data to bytes to be added to the ``'bytes'`` field of the :attr:`data_records` table.
        """

    @abc.abstractmethod
    def decode_record_bytes(self, record_bytes: bytes):
        """
        Decodes the record's ``'bytes'`` field to produce the record's data.
        """

    def add(self, index, record_data, connection=None):
        """
        Add new data row.
        """

        # Convert data, build records
        record = {'index': index,
                  'writer_id': self.data_store.writer_id,
                  'data_def_id': self._data_def.id,
                  'bytes': self.encode_record_bytes(record_data)}
        # records = [{'row_bytes': np.ascontiguousarray(recfns.repack_fields(arr_row)).tobytes()} for arr_row in arr]

        # Write to database.
        with self.begin_connection(connection) as connection:
            connection.execute(insert(self.data_records_table), record)

    def load(self, *criterion, connection=None):
        """
        By default, loads all the records owned by this handler.

        :param criterion: Passed as extra args to a where clause to further restrict the number of records
        """
        with self.begin_connection(connection) as connection:
            # Copy records to pre-assigned memory space.
            records = list(connection.execute(select(self.data_records_table).where(
                self.data_records_table.c.data_def_id == self._data_def.id, *criterion).order_by(
                    self.data_records_table.c.index)))

        return self._format_records(records)

    def _format_records(self, records):
        meta = np.empty(len(records), dtype=[('index', 'i'),
                                             ('created', 'datetime64[us]'),
                                             ('writer_id', 'i')])
        meta['index'] = [_rec.index for _rec in records]
        meta['created'] = [_rec.created for _rec in records]
        meta['writer_id'] = [_rec.writer_id for _rec in records]

        return {'meta': meta,
                'data': self.merge_records_data(
                    [self.decode_record_bytes(_rec.bytes) for _rec in records])}

    @abc.abstractmethod
    def merge_records_data(self, records_data):
        """
        Merges the list of decoded record bytes to create the ``'data'`` field of the dictionary output by the load function.
        """

    def __len__(self, connection=None):
        """
        Returns the number of records in the array (i.e., the first dimension of the array). The shape of the entire stored array is hence (len(self), *self.recdims).
        """
        with self.lock:
            if self._data_def is None:
                return 0
            else:
                with self.begin_connection(connection) as connection:
                    return connection.execute(
                        select(func.count(self.data_records_table.c.id)).where(
                            self.data_records_table.c.data_def_id == self._data_def['id'])
                    ).scalar()
