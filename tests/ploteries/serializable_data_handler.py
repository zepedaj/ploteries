from unittest import TestCase

benchmark = lambda x: (lambda y: y)  # from marcabanca import benchmark
from xerializer import Serializer
from sqlalchemy.sql import column as c
import numpy.testing as npt
import jztools.numpy as pgnp
import numpy as np
from ploteries import serializable_data_handler as mdl
from .data_store import get_store
from jztools.sqlalchemy import begin_connection


from .ndarray_data_handlers import complex_dtype


class TestSerializableDataHandler(TestCase):
    def test_create(self):
        with get_store() as store:
            dh = mdl.SerializableDataHandler(store, "arr1")

    @benchmark(False)
    def test_encode_decode(self):
        class Mock:
            # Simulates SerializableDataHandler
            _serializer = Serializer()

        for val in [
            _np_arr := pgnp.random_array((10, 5, 7), dtype=complex_dtype()),
            {"np_arr": _np_arr, "list": [0, 1, 2], "slice": slice(0, 10, 20)},
        ]:
            encoded_val = mdl.SerializableDataHandler.encode_record_bytes(Mock(), val)
            decoded_val = mdl.SerializableDataHandler.decode_record_bytes(
                Mock(), encoded_val
            )
            npt.assert_equal(val, decoded_val)

    def test_add(self):
        arrs = [
            pgnp.random_array((k + 5, k * 2, k + 3), dtype=complex_dtype())
            for k in range(1, 10)
        ]
        with get_store() as store:
            dh = mdl.SerializableDataHandler(store, "arr1")
            for k, _arr in enumerate(arrs):
                dh.add_data(k, _arr)

            loaded_arrs = dh.load_data()

            for _arr, _loaded_arr in zip(arrs, loaded_arrs["data"]):
                npt.assert_array_equal(_arr, _loaded_arr)

            for _arr, _loaded_arr in zip(
                arrs,
                store.get_data_handlers(c("name") == "arr1")[0].load_data()["data"],
            ):
                npt.assert_array_equal(_arr, _loaded_arr)
