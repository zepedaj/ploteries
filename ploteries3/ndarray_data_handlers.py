from threading import RLock
import json
from numpy.lib.format import dtype_to_descr, descr_to_dtype
import numpy as np
from collections import namedtuple
from numpy.lib import recfunctions as recfns
from typing import Optional
from .base_handlers import DataHandler
from numpy.typing import ArrayLike
from .data_store import DataStore


class NDArraySpec(namedtuple('_NDArraySpec', ('dtype', 'shape'))):
    def __new__(cls, dtype, shape):
        dtype = recfns.repack_fields(np.empty(0, dtype=dtype)).dtype
        shape = tuple(shape)
        return super().__new__(cls, dtype, shape)

    @classmethod
    def produce(cls, val):
        if isinstance(val, (tuple, list)):
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
    Stores numpy arrays (and compatible, including scalars and lists thereof) efficiently enforcing the same shape and dtype, and loads them as a large numpy array, with each data_records table row corresponding to an entry (possibly multi-dimensional) in the numpy array. Supports lazy (upon first record add operation) definition of the ndarrary_spec (i.e., ndarray dtype and shape).

    Supports multi-dimensional arrays of arbitary dtype (except Object) including (possibly nested) structured arrays. Supports also scalar inputs (e.g., integers, floats) and array-like inputs that can be passed to :method:`np.require` (e.g., lists, tuples).

    Data store tables can be created in one of two ways:

    1. By explicitly calling :method:`write_def`:
    ```
    und_dh = UniformNDArrayDataHandler(data_store, 'name', {'dtype':dtype('f'), 'shape':(10,1)})
    und_dh.write_def()
    ```
    This method will fail if param :attr:`ndarray_spec` is not explicitly provided at initialization.

    2. By adding data to an implicitly initialized object:
    ```
    und_dh = UniformNDArrayDataHandler(data_store, 'name')
    und_dh.add_data(0, np.array([0.0, 1.0]))
    ```
    This method can also be used with explicitly initialized objects, and a type-check will be carried out in this case.

    3. Using :meth:`from_name`
    ```
    und_dh = UniformNDArrayDataHandler.from_name(data_store, 'data_name')
    ```

    """
    data_store = None
    decoded_data_def = None

    def __init__(self,
                 data_store: DataStore,
                 name: str,
                 ndarray_spec: Optional[NDArraySpec] = None,
                 _decoded_data_def=None):
        """
        :param data_store: :class:`ploteries3.data_store.DataStore` object.
        :param name: Data name.
        :param ndarray_spec: An :class:`NDArraySpec` producible.
        """
        self.lock = RLock()
        self.data_store = data_store
        self._init_name = name
        self._init_ndarray_spec = (
            None if ndarray_spec is None else
            NDArraySpec.produce(ndarray_spec))
        self.decoded_data_def = _decoded_data_def

    @property
    def name(self):
        return self.decoded_data_def.name if self.decoded_data_def else self._init_name

    @property
    def ndarray_spec(self):
        """
        The dtype and shape of data in each record.
        """
        if self.decoded_data_def is not None:
            return self.decoded_data_def.params['ndarray_spec']
        elif self._init_ndarray_spec is not None:
            return self._init_ndarray_spec
        else:
            raise ValueError(
                f'The type/shape of data records is not known yet for data handler {self.name}. '
                f'Add data to the table or specify the data type/shape at initialization.')

    @classmethod
    def from_def_record(cls, data_store, data_def_record):
        cls.decode_params(data_def_record.params)
        obj = cls(data_store, None, None, data_def_record)

        return obj

    @classmethod
    def decode_params(cls, params):
        params['ndarray_spec'] = NDArraySpec.from_serializable(
            params['ndarray_spec'])

    def encode_params(self, ndarray_spec=None):
        ndarray_spec = ndarray_spec or self.ndarray_spec
        return {'ndarray_spec': ndarray_spec.as_serializable()}

    @property
    def row_num_bytes(self):
        """
        Number of bytes in each record.
        """
        return int(np.prod(self.ndarray_spec.shape)*self.ndarray_spec.dtype.itemsize)

    def add_data(self, index: int, arr: ArrayLike, connection=None):
        #
        explicit_ndarray_spec = (
            None if (
                self._init_ndarray_spec is None and self.decoded_data_def is None) else
            self.ndarray_spec)

        # Required for support of scalars and lists. Note that the data-type can be provided at object
        # instantiation time, or it can be inferred.
        if not isinstance(arr, np.ndarray):
            arr = np.require(
                arr, dtype=None if not explicit_ndarray_spec else explicit_ndarray_spec.dtype)
            if not arr.shape:
                arr = arr.view()
                arr_shape = tuple()
                arr.shape = (1,)
        else:
            arr_shape = arr.shape

        # Check input ndarray spec against initialization ndarray spec.
        input_ndarray_spec = NDArraySpec(arr.dtype, arr_shape)
        if (self._init_ndarray_spec and input_ndarray_spec != self._init_ndarray_spec):
            raise TypeError(
                f"The input ndarray spec {input_ndarray_spec} does not match the value "
                f"{self._init_ndarray_spec} provided at initialization.")

        with self.lock, self.data_store.begin_connection(connection=connection) as connection:

            # Write data def if it does not exist.
            if self.decoded_data_def is None:
                self.write_def(connection=connection, ndarray_spec=input_ndarray_spec)
                self.decoded_data_def = self.load_decode_def(
                    self.data_store, self.name, connection=connection)

            #
            decoded_ndarray_spec = self.decoded_data_def['params']['ndarray_spec']
            # Check the input ndarray spec against the stored ndarray spec.
            if (self.decoded_data_def and input_ndarray_spec != decoded_ndarray_spec):
                raise TypeError(
                    f"The input ndarray spec {input_ndarray_spec} does not match the value "
                    f"{decoded_ndarray_spec} in data store table {self.name}.")

            # ADd data.
            super().add_data(index, arr, connection=connection)

    def encode_record_bytes(self, arr):

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
    decoded_data_def = None

    def __init__(self, data_store, name):
        """
        :param data_store: :class:`ploteries3.data_store.DataStore` object.
        :param name: Data name.
        """
        self.data_store = data_store
        self.name = name
        with self.data_store.begin_connection() as connection:
            if not (loaded_def := self.load_decode_def(
                    self.data_store, self.name, connection=connection)):
                self.write_def(connection=connection)
                loaded_def = self.load_decode_def(
                    self.data_store, self.name, connection=connection)
            if not loaded_def:
                raise Exception('Unexpected error.')
        self.decoded_data_def = loaded_def

    @ classmethod
    def from_def_record(cls, data_store, data_def_record):
        obj = object.__new__(cls)
        obj.data_store = data_store
        obj.name = data_def_record.name
        obj.decoded_data_def = data_def_record

        return obj

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
