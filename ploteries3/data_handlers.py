from threading import RLock
import json
import abc
from numpy.lib.format import dtype_to_descr, descr_to_dtype
from contextlib import contextmanager
import numpy as np
from collections import namedtuple
from numpy.lib import recfunctions as recfns
from pglib.sqlalchemy import begin_connection
import re
from sqlalchemy.engine.result import Row
from typing import Union, Optional
from sqlalchemy import insert, func, select, exc


class NDArraySpec(namedtuple('_NDArraySpec', ('dtype', 'shape'))):
    def __new__(cls, dtype, shape):
        dtype = recfns.repack_fields(np.empty(0, dtype=dtype)).dtype
        shape = tuple(shape)
        return super().__new__(cls, dtype, shape)

    @classmethod
    def produce(cls, val):
        if val is None:
            return None
        elif isinstance(val, (tuple, list)):
            return cls(*val)
        elif isinstance(val, dict):
            return cls(**val)
        elif isinstance(val, cls):
            return val
        else:
            raise TypeError(f'Cannot produce NDArraySpec from {type(val)}.')

    def as_serializable(self):
        return {'dtype': dtype_to_descr(self.dtype),
                'shape': self.shape}

    @classmethod
    def from_serializable(cls, val):
        return cls(dtype=descr_to_dtype(val['dtype']),
                   shape=tuple(val['shape']))


class DataHandler(abc.ABC):

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
            return data_def
        else:
            raise Exception('Unexpected case.')

    def _write_def(self, record_dict, connection=None):
        """
        Adds an entry to the data defs table and returns True if successful. If an entry already exists, returns False.
        """

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


class NDArrayDataHandler(DataHandler):
    """
    Writes numpy arrays efficiently enforcing the same shape and dtype, and loads them as a large numpy array, with each data_records table row corresponding to an (possibly multi-dimensional) entry in the numpy array.

    Supports multi-dimensional arrays of arbitary dtype (except Object) including (possibly nested) structured arrays.
    """
    _ndarray_spec = None
    data_store = None
    _data_def = None

    def __init__(self, data_store, name, ndarray_spec: Optional[NDArraySpec] = None):
        """
        :param data_store: :class:`ploteries3.data_store.DataStore` object.
        :param name: Data name.
        :param ndarray_spec: An :class:`NDArraySpec` producible.
        """
        self.lock = RLock()
        self.data_store = data_store
        self.name = name
        self.ndarray_spec = NDArraySpec.produce(ndarray_spec)

    def _load_def(self, connection=None) -> Union[bool, Row]:
        with self.begin_connection(connection) as conn:
            if (data_def := super()._load_def(connection=conn)):
                data_def.params['ndarray_spec'] = NDArraySpec.from_serializable(
                    data_def.params['ndarray_spec'])
            return data_def

    def _write_def(self, connection=None):
        with self.begin_connection(connection) as connection:
            return super()._write_def({
                'name': self.name,
                'handler': type(self),
                'params': {'ndarray_spec': self.ndarray_spec.as_serializable()}
            }, connection=connection)

    @ property
    def ndarray_spec(self):
        """
        The dtype of scalars in the array.
        """
        return self._ndarray_spec

    @ ndarray_spec.setter
    def ndarray_spec(self, ndarray_spec):

        ndarray_spec = NDArraySpec.produce(ndarray_spec)

        with self.lock:
            if self._ndarray_spec is not None:
                # Spec previously set, check match.
                if self._ndarray_spec != ndarray_spec:
                    raise ValueError(
                        f'Non-matching ndarray_spec {self.ndarray_spec} and {ndarray_spec}.')
            else:
                with self.begin_connection() as connection:
                    if loaded_def_row := self._load_def(connection=connection):
                        # Loaded an existing def, set and check match.
                        self._ndarray_spec = loaded_def_row.params['ndarray_spec']
                        self._data_def = loaded_def_row
                        if ndarray_spec is not None:
                            self.ndarray_spec = ndarray_spec
                    elif ndarray_spec is not None:
                        # Def does not exist, write, load and check match.
                        self._ndarray_spec = ndarray_spec
                        self._write_def(connection=connection)
                        loaded_def_row = self._load_def(connection=connection)
                        self._data_def = loaded_def_row
                        self.ndarray_spec = loaded_def_row.params['ndarray_spec']

    @ property
    def row_shape(self):
        """
        The dimensions of each record in the array. Will be an empty tuple for scalar records.
        """
        return None if self.ndarray_spec is None else self.ndarray_spec.shape

    @ property
    def rowsize(self):
        """
        Number of bytes in one row.
        """
        return int(np.prod(self.row_shape)*self.ndarray_spec.dtype.itemsize)

    def add(self, index, arr, connection=None):
        """
        Add new data row.
        """

        # Check data.
        self.ndarray_spec = NDArraySpec(arr.dtype, arr.shape)

        # Convert data, build records
        packed_arr = np.require(arr, dtype=self.ndarray_spec.dtype, requirements='C').view('u1')
        packed_arr.shape = packed_arr.size
        packed_arr = packed_arr.data
        record = {'index': index,
                  'writer_id': self.data_store.writer_id,
                  'data_def_id': self._data_def.id,
                  'bytes': packed_arr}
        # records = [{'row_bytes': np.ascontiguousarray(recfns.repack_fields(arr_row)).tobytes()} for arr_row in arr]

        # Write to database.
        with self.begin_connection(connection) as connection:
            connection.execute(insert(self.data_records_table), record)

    def load(self, connection=None):
        """
        Loads all data corresponding to this data handler and returns it as a numpy array.
        """
        if self._data_def is None:
            return None
        else:
            with self.begin_connection(connection) as connection:

                # Pre alloc output
                out_ndarray = np.empty(
                    (expected_len := self.__len__(connection), *self.ndarray_spec.shape),
                    dtype=self.ndarray_spec.dtype)
                out_buffer = out_ndarray.view(dtype='u1')
                out_buffer.shape = np.prod(out_buffer.shape)
                row_itemsize = self.rowsize

                # Copy records to pre-assigned memory space.
                for k, row in enumerate(
                        connection.execute(select(self.data_records_table).where(
                            self.data_records_table.c.data_def_id == self._data_def.id).order_by(
                                self.data_records_table.c.index))):
                    out_buffer[k*row_itemsize:(k+1)*row_itemsize] = bytearray(row.bytes)

                # Sanity checks.
                if k != expected_len-1:
                    raise Exception('Could not load the expected number of rows!')
                if (k+1)*row_itemsize != out_buffer.size:
                    raise Exception('Did not load the expected number of bytes!')

                return out_ndarray


class RaggedNDArrayDataHandler(DataHandler):
    """
    Writes rows in data_records table that each contain one numpy array with arbitrary shape and dtype. Supports multi-dimensional arrays of arbitary dtype (except Object) including (possibly nested) structured arrays.
    """

    data_store = None
    _data_def = None

    def __init__(self, data_store, name):
        """
        :param data_store: :class:`ploteries3.data_store.DataStore` object.
        :param name: Data name.
        """
        self.data_store = data_store
        self.name = name
        with self.begin_connection() as connection:
            if not (loaded_def := self._load_def(connection=connection)):
                self._write_def({'name': self.name, 'handler': type(self)})
                loaded_def = self._load_def(connection=connection)
            if not loaded_def:
                raise Exception('Unexpected error.')
        self._data_def = loaded_def

    @staticmethod
    def encode(arr):
        ndarray_spec = NDArraySpec(arr.dtype, arr.shape)
        packed_arr = np.require(arr, dtype=ndarray_spec.dtype, requirements='C').view('u1')
        packed_arr.shape = packed_arr.size
        packed_arr = packed_arr.data

        # Add header as bytes
        ndarray_specs_as_bytes = json.dumps(ndarray_spec.as_serializable()).encode('utf-8')
        ndarray_specs_len_as_bytes = (len(ndarray_specs_as_bytes)).to_bytes(8, 'big')
        return ndarray_specs_len_as_bytes + ndarray_specs_as_bytes + packed_arr

    @staticmethod
    def decode(arr_bytes):
        ndarray_specs_len_as_bytes = int.from_bytes(arr_bytes[:8], 'big')
        ndarray_spec = NDArraySpec.from_serializable(json.loads(
            arr_bytes[8: (data_start := (8 + ndarray_specs_len_as_bytes))].decode('utf-8')))
        out_arr = np.empty(shape=ndarray_spec.shape, dtype=ndarray_spec.dtype)

        out_buffer = out_arr.view(dtype='u1')
        out_buffer.shape = np.prod(out_buffer.shape)

        out_buffer[:] = bytearray(arr_bytes[data_start:])

        return out_arr

    def add(self, index, arr, connection=None):
        """
        Add new data row.
        """

        # Check data.

        # Convert data, build records
        record = {'index': index,
                  'writer_id': self.data_store.writer_id,
                  'data_def_id': self._data_def.id,
                  'bytes': self.encode(arr)}
        # records = [{'row_bytes': np.ascontiguousarray(recfns.repack_fields(arr_row)).tobytes()} for arr_row in arr]

        # Write to database.
        with self.begin_connection(connection) as connection:
            connection.execute(insert(self.data_records_table), record)

    def load(self, connection=None):
        """
        Loads all data corresponding to this data handler and returns it as a numpy array.
        """
        if self._data_def is None:
            return None
        else:
            with self.begin_connection(connection) as connection:

                # Pre alloc output
                out_arrays = []

                # Copy records to pre-assigned memory space.
                for k, row in enumerate(
                        connection.execute(select(self.data_records_table).where(
                            self.data_records_table.c.data_def_id == self._data_def.id).order_by(
                                self.data_records_table.c.index))):
                    out_arrays.append(self.decode(row.bytes))

                return out_arrays
