from threading import RLock
from numpy.lib.format import dtype_to_descr, descr_to_dtype
import sqlite3
from numba import njit, jit
from typing import Sequence, Union, Optional
import threading
import numpy as np
from jzf_io import abstract
from collections import namedtuple
from numpy.lib import recfunctions as recfns
import warnings
from collections import deque
import sqlalchemy as sqa
from sqlalchemy.sql.expression import alias
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine, MetaData, event, Table, Column, Integer, String, \
    ForeignKey, types, insert, UniqueConstraint, func, exc, column, text, select
from sqlalchemy.engine import Engine
# from .figure_managers import load_figure as figure_manager_load_figure, global_steps as figure_manager_global_steps
# from pglib.sqlalchemy import PlotlyFigureType, ClassType, sql_query_type_builder, JSONEncodedType
from pglib.sqlalchemy import JSONEncodedType, NumpyDtypeType, begin_connection
# from ._sql_data_types import DataMapperType
import re
from sqlalchemy.engine.result import Row


class NDArraySpec(namedtuple('_NDArraySpec', ('dtype', 'row_shape'))):
    def __new__(cls, dtype, row_shape):
        dtype = recfns.repack_fields(np.empty(0, dtype=dtype)).dtype
        row_shape = tuple(row_shape)
        return super().__new__(cls, dtype, row_shape)

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
                'row_shape': self.row_shape}

    @classmethod
    def from_serializable(cls, val):
        return cls(dtype=descr_to_dtype(val['dtype']),
                   row_shape=tuple(val['row_shape']))


class NumpyDataHandler:
    """
    Writes numpy arrays with the same row shape and dtype, and loads them as a large numpy array, with each data_records table row corresponding to an (possibly multi-dimensional) entry in the numpy array.

    Supports multi-dimensional arrays of arbitary dtype (except Object) including (possibly nested) structured arrays.
    """
    _ndarray_spec = None

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

    @property
    def data_defs_table(self):
        return self.data_store._metadata.tables['data_defs']

    @property
    def data_records_table(self):
        return self.data_store._metadata.tables['data_records']

    def _load_def(self, connection=None) -> Union[bool, Row]:
        """
        Loads and decodes the definition from the the data defs table, if it exists, returning
        the decoded data def row if successful.

        Returns False if no data def is found in the table, or the data def row if it is found.
        """

        with begin_connection(self.data_store.engine, connection) as conn:
            # Build query
            qry = self.data_defs_table.select().where(self.data_defs_table.c.name == self.name)

            # Check that the store specs match the input.
            data_def = conn.execute(qry).fetchall()

        if len(data_def) == 0:
            return False
        elif len(data_def) == 1:
            data_def = data_def[0]
            data_def.params['ndarray_spec'] = NDArraySpec.from_serializable(
                data_def.params['ndarray_spec'])
            return data_def
        else:
            raise Exception('Unexpected case.')

    def _write_def(self, connection=None):
        """
        Adds an entry to the data defs table and returns True if successful. If an entry already exists, returns False.
        """

        with begin_connection(self.data_store.engine, connection) as connection:
            try:
                connection.execute(
                    insert(self.data_defs_table),
                    [{'name': self.name,
                      'handler': type(self),
                      'params': {'ndarray_spec': self.ndarray_spec.as_serializable()}
                      }])
            except exc.IntegrityError as err:
                if re.match(
                        f'\\(sqlite3.IntegrityError\\) UNIQUE constraint failed\\: {self.data_defs_table.name}\\.name',
                        str(err)):
                    return False
                else:
                    raise

            else:
                return True

    @property
    def ndarray_spec(self):
        """
        The dtype of scalars in the array.
        """
        return self._ndarray_spec

    @ndarray_spec.setter
    def ndarray_spec(self, ndarray_spec):

        ndarray_spec = NDArraySpec.produce(ndarray_spec)

        with self.lock:
            if self._ndarray_spec is not None:
                # Spec previously set, check match.
                if self._ndarray_spec != ndarray_spec:
                    raise ValueError(
                        f'Non-matching ndarray_spec {self.ndarray_spec} and {ndarray_spec}.')
            else:
                if loaded_def_row := self._load_def():
                    # Loaded an existing def, set and check match.
                    self._ndarray_spec = loaded_def_row.params['ndarray_spec']
                    self._data_def = loaded_def_row
                    if ndarray_spec is not None:
                        self.ndarray_spec = ndarray_spec
                elif ndarray_spec is not None:
                    # Def does not exist, write, load and check match.
                    self._ndarray_spec = ndarray_spec
                    self._write_def()
                    loaded_def_row = self._load_def()
                    self._data_def = loaded_def_row
                    self.ndarray_spec = loaded_def_row.params['ndarray_spec']

    @property
    def row_shape(self):
        """
        The dimensions of each record in the array. Will be an empty tuple for scalar records.
        """
        return None if self.ndarray_spec is None else self.ndarray_spec.row_shape

    @property
    def rowsize(self):
        """
        Number of bytes in one row.
        """
        return int(np.prod(self.row_shape)*self.ndarray_spec.dtype.itemsize)

    def __len__(self, connection=None):
        """
        Returns the number of records in the array (i.e., the first dimension of the array). The shape of the entire stored array is hence (len(self), *self.recdims).

        TODO: Net ready?
        """
        with self.lock:
            if self._data_def is None:
                return 0
            else:
                with begin_connection(self.data_store.engine, connection) as connection:
                    return connection.execute(
                        select(func.count(self.data_records_table.c.id)).where(
                            self.data_records_table.c.data_def_id == self._data_def['id'])
                    ).scalar()

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
        with begin_connection(self.data_store.engine, connection) as connection:
            connection.execute(insert(self.data_records_table), record)

    def load(self, connection=None):
        if self._data_def is None:
            return None
        else:
            with begin_connection(self.data_store.engine, connection) as connection:

                # Pre alloc output
                out_ndarray = np.empty(
                    (expected_len := self.__len__(connection), *self.ndarray_spec.row_shape),
                    dtype=self.ndarray_spec.dtype)
                out_buffer = out_ndarray.view(dtype='u1')
                out_buffer.shape = np.prod(out_buffer.shape)
                row_itemsize = self.rowsize

                # Copy records to pre-assigned.
                for k, row in enumerate(
                        connection.execute(select(self.data_records_table).where(
                            self.data_records_table.c.data_def_id == self._data_def.id))):
                    out_buffer[k*row_itemsize:(k+1)*row_itemsize] = bytearray(row.bytes)

                # Sanity checks.
                if k != expected_len-1:
                    raise Exception('Could not load the expected number of rows!')
                if (k+1)*row_itemsize != out_buffer.size:
                    raise Exception('Did not load the expected number of bytes!')

                return out_ndarray

    def delete(self):
        """
        Delete the array from disk
        """

    def _get_ids(self, connection=None):
        with begin_connection(self.qladas.engine, connection) as connection:
            return np.array(
                [row.row_id
                 for row in connection.execute(select(self.table.c.row_id).order_by(
                     self.table.c.row_id))],
                dtype='i8')

    def _query_as_array(self,  qry, max_size, connection=None):

        # Pre-alloc output
        out = np.empty((max_size, *self.row_shape), dtype=self.dtype)
        out_buffer = out.view(dtype='u1')
        out_buffer.shape = np.prod(out_buffer.shape)
        out_row_ids = np.empty(len(out), dtype='i8')

        k = -1
        row_itemsize = self.rowsize
        with begin_connection(self.qladas.engine, connection) as connection:
            for k, row in enumerate(connection.execute(qry)):
                out_buffer[k*row_itemsize:(k+1)*row_itemsize] = bytearray(row.row_bytes)
                out_row_ids[k] = row.row_id

        return out[:(k+1)], out_row_ids[:(k+1)]

    def __getitem__(self, idx: Union[int, Sequence[int], slice], _num_recs=None) -> np.ndarray:

        num_recs = _num_recs if _num_recs is not None else len(self)

        if not isinstance(idx, (list, tuple, slice, int)):
            raise Exception('Invalid index type.')

        if isinstance(idx, int):
            # INTEGER INDEX
            return self[[idx]][0]

        elif isinstance(idx, (list, tuple)):
            # LIST/TUPLE INDEX

            # Check all indices before reading
            [self._check_valid_index(k, _num_recs=num_recs) for k in idx]

            # Prepare query.
            all_ids = self._get_ids()
            expected_ids, inverse_mapping = np.unique(
                [all_ids[_curr_idx] for _curr_idx in idx],
                return_inverse=True)
            qry = select(self.table.c.row_id, self.table.c.row_bytes).where(
                self.table.c.row_id.in_(list(map(int, expected_ids))))  # .order_by(self.table.c.row_id)

            # Retrieve bytes and write to out array
            retrieved_arr, retrieved_ids = self._query_as_array(qry, len(idx))
            out_arr = retrieved_arr[inverse_mapping]
            out_ids = retrieved_ids[inverse_mapping]

            # Verify row ids.
            assert np.array_equal(retrieved_ids, np.array(expected_ids, dtype='i8')
                                  ), 'Could not retrieved the requested rows.'

            return out_arr

        elif isinstance(idx, slice) and idx.step is not None:

            # SLICE INDEX, WITH STEP
            num_recs = _num_recs if _num_recs is not None else len(self)
            start, stop, step = self._get_slice_range(idx, num_recs)
            idx_list = list(range(start, stop, step))
            return self[idx_list]

        elif isinstance(idx, slice) and idx.step is None:
            # SLICE INDEX, NO STEP
            start, stop, _ = self._get_slice_range(idx, num_recs)
            num_to_read = max(stop-start, 0)

            # Prepare query.
            qry = select(
                self.table.c.row_id, self.table.c.row_bytes
            ).order_by(self.table.c.row_id).offset(start).limit(num_to_read)

            # Retrieve bytes and write to out array
            out_arr, retrieved_ids = self._query_as_array(qry, num_to_read)

            return out_arr

        else:
            raise Exception('Invalid input.')
