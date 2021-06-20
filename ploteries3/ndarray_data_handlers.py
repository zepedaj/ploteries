from threading import RLock
from numbers import Number
from numpy import typing as np_typing
import json
from numpy.lib.format import dtype_to_descr, descr_to_dtype
import numpy as np
from collections import namedtuple
from numpy.lib import recfunctions as recfns
from typing import Union, Optional
from .base_data_handler import DataHandler
from sqlalchemy import insert


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


class UniformNDArrayDataHandler(DataHandler):
    """
    Writes numpy arrays (and compataible, including scalars) efficiently enforcing the same shape and dtype, and loads them as a large numpy array, with each data_records table row corresponding to an (possibly multi-dimensional) entry in the numpy array. Supports lazy (upon first record add operation) definition of the ndarrary_spec (dtype and shape).

    Supports multi-dimensional arrays of arbitary dtype (except Object) including (possibly nested) structured arrays. Supports also scalar inputs that are number.Number sub-types (e.g., integers, floats) or can be converted to arrays using np.require (e.g., lists, tuples).
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

    @classmethod
    def from_record(self):
        pass

    @classmethod
    def decode_params(self, params):
        params['ndarray_spec'] = NDArraySpec.from_serializable(
            params['ndarray_spec'])
        return params

    def encode_params(self):
        return {'ndarray_spec': self.ndarray_spec.as_serializable()}

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

    @property
    def row_shape(self):
        """
        The dimensions of each record in the array. Will be an empty tuple for scalar records.
        """
        return None if self.ndarray_spec is None else self.ndarray_spec.shape

    @property
    def row_num_bytes(self):
        """
        Number of bytes in one row.
        """
        return int(np.prod(self.row_shape)*self.ndarray_spec.dtype.itemsize)

    def encode_record_bytes(self, arr):

        # Required for support of scalars and lists. Note that the data-type can be provided at object
        # instantiation time, or it can be inferred.
        arr = np.require(
            arr,
            dtype=(None if self.ndarray_spec is None else self.ndarray_spec.dtype))
        if not arr.shape:
            arr_shape = tuple()
            arr.shape = (1,)
        else:
            arr_shape = arr.shape

        # Check or set data.
        self.ndarray_spec = NDArraySpec(arr.dtype, arr_shape)

        # Convert data, build records
        packed_arr = np.require(arr, dtype=self.ndarray_spec.dtype, requirements='C').view('u1')
        packed_arr.shape = packed_arr.size

        return packed_arr.data

    def decode_record_bytes(self, data_bytes):
        return data_bytes

    def merge_records_data(self, records_data):

        # Pre alloc output
        out_ndarray = np.empty(
            (len(records_data), *self.ndarray_spec.shape),
            dtype=self.ndarray_spec.dtype)
        out_buffer = out_ndarray.view(dtype='u1')
        out_buffer.shape = np.prod(out_buffer.shape)
        row_itemsize = self.row_num_bytes

        for k, row_bytes in enumerate(records_data):
            out_buffer[k*row_itemsize:(k+1)*row_itemsize] = bytearray(row_bytes)

        # Sanity checks.
        if k != len(records_data)-1:
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
                self._write_def(connection=connection)
                loaded_def = self._load_def(connection=connection)
            if not loaded_def:
                raise Exception('Unexpected error.')
        self._data_def = loaded_def

    @classmethod
    def from_record(self):
        pass

    def encode_record_bytes(self, arr):

        ndarray_spec = NDArraySpec(arr.dtype, arr.shape)
        packed_arr = np.require(arr, dtype=ndarray_spec.dtype, requirements='C').view('u1')
        packed_arr.shape = packed_arr.size
        packed_arr = packed_arr.data

        # Add header as bytes
        ndarray_specs_as_bytes = json.dumps(ndarray_spec.as_serializable()).encode('utf-8')
        ndarray_specs_len_as_bytes = (len(ndarray_specs_as_bytes)).to_bytes(8, 'big')
        return ndarray_specs_len_as_bytes + ndarray_specs_as_bytes + packed_arr

    def decode_record_bytes(self, arr_bytes):
        ndarray_specs_len_as_bytes = int.from_bytes(arr_bytes[:8], 'big')
        ndarray_spec = NDArraySpec.from_serializable(json.loads(
            arr_bytes[8: (data_start := (8 + ndarray_specs_len_as_bytes))].decode('utf-8')))
        out_arr = np.empty(shape=ndarray_spec.shape, dtype=ndarray_spec.dtype)

        out_buffer = out_arr.view(dtype='u1')
        out_buffer.shape = np.prod(out_buffer.shape)

        out_buffer[:] = bytearray(arr_bytes[data_start:])

        return out_arr

    def merge_records_data(self, records_data):
        return records_data
