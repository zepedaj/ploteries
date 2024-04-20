from ploteries.figure_handler import table_handler as mdl
from ploteries.data_store import Ref_
import numpy.testing as npt
from jztools.slice_sequence import SSQ_
import plotly.graph_objects as go
from ploteries.serializable_data_handler import SerializableDataHandler
from unittest import TestCase
from ..data_store import get_store
import numpy as np
from jztools.profiling import time_and_print
from contextlib import contextmanager
import dash


@contextmanager
def get_store_with_table(transposed=False, sorting="ascending"):
    with get_store() as store:
        #
        data1_h = SerializableDataHandler(store, "table_data")
        for index in range(5):
            data1_h.add_data(
                index, {f"Column {index}": index, f"Column {index+1}": index * 2}
            )

        #
        tab1_h = mdl.TableHandler(
            store,
            "table1",
            ("table_data", SSQ_()),
            transposed=transposed,
            sorting=sorting,
        )

        store.flush()
        yield store, data1_h, tab1_h
        store.flush()


class TestTableHandler(TestCase):
    def test_create(self):
        self._test_create(False)

    def test_create_transposed(self):
        self._test_create(True)

    def _test_create(self, transposed):

        with get_store_with_table(transposed=transposed) as (store, data1_h, tab1_h):
            built_tbl = tab1_h.build_table()

            #
            def check_valid(_tbl):
                as_array = np.array(
                    [
                        [None if x == "" else x for x in _row.values()]
                        for _row in _tbl.data
                    ]
                ).T
                if transposed:
                    as_array = as_array[1:].T

                # Check indices.
                npt.assert_array_equal(as_array[0], [0, 1, 2, 3, 4])
                # Check entries
                npt.assert_array_equal(np.diag(as_array[1:]), [0, 1, 2, 3, 4])
                npt.assert_array_equal(np.diag(as_array[1:], -1), [0, 2, 4, 6, 8])

            #
            check_valid(built_tbl)

            # Write the definition to the store
            tab1_h.write_def()
            tab1_h_loaded = mdl.TableHandler.from_name(store, tab1_h.name)

            # Compare figures.
            built_tbl_loaded = tab1_h_loaded.build_table()
            self.assertEqual(
                built_tbl_json := built_tbl.to_plotly_json(),
                built_tbl_loaded.to_plotly_json(),
            )

    def test_encode_decode_params(self):
        with get_store_with_table() as (store, data1_h, tab1_h):
            orig_params = {key: getattr(tab1_h, key) for key in tab1_h._state_keys}
            mdl.TableHandler.decode_params(decoded_params := tab1_h.encode_params())
            self.assertDictEqual(orig_params, decoded_params)

    def test_update(self):
        with get_store_with_table() as (store, data1_h, tab1_h):
            # tab1_h.figure_dict['layout']['template']=None
            with self.assertRaisesRegex(
                Exception,
                r"Cannot update a definition that has not been retrieved from the data store.",
            ):
                tab1_h.write_def(mode="update")
            self.assertTrue(tab1_h.write_def())
            self.assertFalse(tab1_h.write_def())

            tab1_h = store.get_figure_handlers()[0]
            self.assertEqual(tab1_h.data_table_template["props"], {})
            tab1_h.data_table_template["props"] = None
            tab1_h.write_def(mode="update")
            tab1_h = store.get_figure_handlers()[0]
            self.assertIsNone(tab1_h.data_table_template["props"])
