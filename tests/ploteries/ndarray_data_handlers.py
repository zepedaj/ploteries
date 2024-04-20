from unittest import TestCase

benchmark = lambda x: (lambda y: y)  # from marcabanca import benchmark
from sqlalchemy.sql import column as c
import numpy.testing as npt
import jztools.numpy as pgnp
import numpy as np
from ploteries import ndarray_data_handlers as mdl
from .data_store import get_store
from jztools.sqlalchemy import begin_connection


def complex_dtype():
    all_types = ["f", "f4", "u1", "i", "L", "datetime64[D]", "datetime64[m]"]
    dtype = [(f"f{k}", fld) for k, fld in enumerate(all_types * 2)]
    return dtype


class TestUniformNDArrayDataHandler(TestCase):
    def test_create(self):
        with get_store() as store:

            ndarray_spec = mdl.NDArraySpec.produce(
                source_spec := (complex_dtype(), (1, 3, 5))
            )

            # From explicit
            ndh1 = mdl.UniformNDArrayDataHandler(
                store, "arr1", ndarray_spec=source_spec
            )
            self.assertEqual(ndh1.ndarray_spec, ndarray_spec)
            ndh1.write_def()
            with begin_connection(store.engine) as conn:
                results = list(
                    conn.execute(ndh1.data_store.data_defs_table.select()).fetchall()
                )
                self.assertEqual(len(results), 1)

            # Implicit
            ndh2 = mdl.UniformNDArrayDataHandler(store, "arr2")
            ndh2.add_data(
                0, data2 := pgnp.random_array(ndarray_spec.shape, ndarray_spec.dtype)
            )
            store.flush()
            self.assertEqual(ndh2.ndarray_spec, ndarray_spec)
            npt.assert_array_equal(data2[None, ...], ndh2.load_data()["data"])

            # Load existing
            ndh3 = mdl.UniformNDArrayDataHandler.from_name(store, "arr1")
            self.assertEqual(ndh3.ndarray_spec, ndh1.ndarray_spec)

    def test_add(self):
        num_arrays = 10
        with get_store() as store:
            dh = mdl.UniformNDArrayDataHandler(store, "arr1")

            arrs = [
                pgnp.random_array((10, 5, 7), dtype=complex_dtype())
                for _ in range(num_arrays)
            ]

            [dh.add_data(k, _arr) for (k, _arr) in enumerate(arrs)]
            store.flush()

            dat = dh.load_data()
            npt.assert_array_equal(dat["data"], np.array(arrs))

            npt.assert_array_equal(
                store.get_data_handlers(c("name") == "arr1")[0].load_data()["data"],
                np.array(arrs),
            )

    def test_add_scalars(self):
        num_arrays = 10
        with get_store() as store:
            dh = mdl.UniformNDArrayDataHandler(store, "arr1")

            arrs = [_v for _v in range(num_arrays)]

            [dh.add_data(k, _arr) for (k, _arr) in enumerate(arrs)]
            store.flush()

            dat = dh.load_data()
            npt.assert_array_equal(dat["data"], np.array(arrs))

            npt.assert_array_equal(
                store.get_data_handlers(c("name") == "arr1")[0].load_data()["data"],
                np.array(arrs),
            )


class TestRaggedNDArrayDataHandler(TestCase):
    def test_create(self):
        with get_store() as store:
            dh = mdl.RaggedNDArrayDataHandler(store, "arr1")

    @benchmark(False)
    def test_encode_decode(self):
        arr = pgnp.random_array((10, 5, 7), dtype=complex_dtype())

        encoded_arr = mdl.RaggedNDArrayDataHandler.encode_record_bytes(None, arr)
        decoded_arr = mdl.RaggedNDArrayDataHandler.decode_record_bytes(
            None, encoded_arr
        )
        npt.assert_array_equal(arr, decoded_arr)

    def test_add(self):
        arrs = [
            pgnp.random_array((k + 5, k * 2, k + 3), dtype=complex_dtype())
            for k in range(1, 10)
        ]
        with get_store() as store:
            dh = mdl.RaggedNDArrayDataHandler(store, "arr1")
            for k, _arr in enumerate(arrs):
                dh.add_data(k, _arr)
            store.flush()

            loaded_arrs = dh.load_data()

            for _arr, _loaded_arr in zip(arrs, loaded_arrs["data"]):
                npt.assert_array_equal(_arr, _loaded_arr)

            for _arr, _loaded_arr in zip(
                arrs,
                store.get_data_handlers(c("name") == "arr1")[0].load_data()["data"],
            ):
                npt.assert_array_equal(_arr, _loaded_arr)
