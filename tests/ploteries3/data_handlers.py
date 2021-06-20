from unittest import TestCase
import numpy.testing as npt
import pglib.numpy as pgnp
import numpy as np
from ploteries3 import data_handlers as mdl
from .data_store import get_store
from pglib.sqlalchemy import begin_connection


def complex_dtype():
    all_types = ['f', 'f4', 'u1', 'i', 'L', 'datetime64[D]', 'datetime64[m]']
    dtype = [(f'f{k}', fld) for k, fld in enumerate(all_types*2)]
    return dtype


class TestUniformNDArrayDataHandler(TestCase):
    def test_create(self):
        with get_store() as store:

            ndarray_spec = mdl.NDArraySpec.produce(
                source_spec :=
                (complex_dtype(), (1, 3, 5)))
            wrong_ndarray_spec = mdl.NDArraySpec.produce(
                (complex_dtype(), (1, 3, 2)))

            # From None
            ndh1 = mdl.UniformNDArrayDataHandler(store, 'arr1')
            ndh1.ndarray_spec = source_spec
            self.assertEqual(ndh1.ndarray_spec, ndarray_spec)
            self.assertNotEqual(ndh1.ndarray_spec, wrong_ndarray_spec)
            with begin_connection(store.engine) as conn:
                results = list(conn.execute(ndh1.data_defs_table.select()).fetchall())
                self.assertEqual(len(results), 1)

            with self.assertRaises(ValueError):
                ndh1.ndarray_spec = wrong_ndarray_spec
            ndh1.ndarray_spec = ndarray_spec

            # Load existing
            ndh2 = mdl.UniformNDArrayDataHandler(store, 'arr1')
            self.assertEqual(ndh2.ndarray_spec, ndh1.ndarray_spec)
            self.assertNotEqual(ndh2.ndarray_spec, wrong_ndarray_spec)
            ndh2.ndarray_spec = ndarray_spec
            self.assertEqual(ndh2.ndarray_spec, ndh1.ndarray_spec)
            self.assertNotEqual(ndh2.ndarray_spec, wrong_ndarray_spec)

            with self.assertRaises(ValueError):
                ndh2.ndarray_spec = wrong_ndarray_spec
            ndh2.ndarray_spec = ndarray_spec

    def test_add(self):
        num_arrays = 10
        with get_store() as store:
            dh = mdl.UniformNDArrayDataHandler(store, 'arr1')

            arrs = [pgnp.random_array((10, 5, 7), dtype=complex_dtype())
                    for _ in range(num_arrays)]

            [dh.add(0, _arr) for _arr in arrs]

            dat = dh.load()
            npt.assert_array_equal(dat, np.array(arrs))


class TestRaggedNDArrayDataHandler(TestCase):
    def test_create(self):
        with get_store() as store:
            dh = mdl.RaggedNDArrayDataHandler(store, 'arr1')

    def test_encode_decode(self):
        arr = pgnp.random_array((10, 5, 7), dtype=complex_dtype())
        encoded_arr = mdl.RaggedNDArrayDataHandler.encode(arr)
        decoded_arr = mdl.RaggedNDArrayDataHandler.decode(encoded_arr)
        npt.assert_array_equal(arr, decoded_arr)

    def test_add(self):
        arrs = [pgnp.random_array((k+5, k*2, k+3), dtype=complex_dtype()) for k in range(1, 10)]
        with get_store() as store:
            dh = mdl.RaggedNDArrayDataHandler(store, 'arr1')
            for k, _arr in enumerate(arrs):
                dh.add(k, _arr)

            loaded_arrs = dh.load()

            for _arr, _loaded_arr in zip(arrs, loaded_arrs):
                npt.assert_array_equal(_arr, _loaded_arr)
