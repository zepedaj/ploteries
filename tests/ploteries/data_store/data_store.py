from unittest import TestCase

benchmark = lambda x: (lambda y: y)  # from marcabanca import benchmark
import ploteries.data_store as mdl
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import jztools.numpy as pgnp
import numpy as np
import numpy.testing as npt
from sqlalchemy.sql import column as c
from ploteries.ndarray_data_handlers import (
    UniformNDArrayDataHandler,
    RaggedNDArrayDataHandler,
)
from jztools.sqlalchemy import ClassType
from sqlalchemy.sql.expression import bindparam


@contextmanager
def get_store():
    with NamedTemporaryFile() as tmpf:
        obj = mdl.DataStore(tmpf.name)
        yield obj
        obj.flush()


@contextmanager
def get_store_with_data(num_uniform=[3, 5, 2], num_ragged=[11, 6, 4, 9]):
    with get_store() as store:
        uniform = []
        for uniform_k, num_time_indices in enumerate(num_uniform):
            dh = UniformNDArrayDataHandler(store, f"uniform_{uniform_k}")
            arrs = [
                pgnp.random_array(
                    (10, 5, 7), dtype=[("f0", "datetime64[s]"), ("f1", "int")]
                )
                for _ in range(num_time_indices)
            ]
            [dh.add_data(k, _arr) for k, _arr in enumerate(arrs)]
            store.flush()
            uniform.append({"handler": dh, "arrays": arrs})

        ragged = []
        for ragged_k, num_time_indices in enumerate(num_ragged):
            dh = RaggedNDArrayDataHandler(store, f"ragged_{ragged_k}")
            arrs = [
                pgnp.random_array(
                    (10 + ragged_k, 5 + ragged_k, 7 + ragged_k),
                    dtype=[("f0", "datetime64[s]"), ("f1", "int")],
                )
                for _ in range(num_time_indices)
            ]
            [dh.add_data(k, _arr) for k, _arr in enumerate(arrs)]
            store.flush()
            ragged.append({"handler": dh, "arrays": arrs})

        yield store, uniform, ragged


class TestRef_(TestCase):
    # @benchmark(False)
    # def test_serialization(self):
    #     ref_ = mdl.Ref_('series1')['abc'][1]['def'][3]
    #     serialized = ref_.serialize()
    #     des_ref_ = mdl.Ref_.deserialize(serialized)
    #     self.assertEqual(ref_, des_ref_)

    def test_copy(self):
        ssq = mdl.Ref_("series1")["abc"][{"abc": 0, "def": 1}][0][::2][3:100][
            {"xyz": 2}
        ]
        self.assertEqual(ssq, ssq2 := ssq.copy())
        ssq2.slice_sequence[2]["abc"] = 1
        self.assertNotEqual(ssq, ssq2)

    # @benchmark(False)
    # def test_hash(self):
    #     ssq = mdl.Ref_('series1')['abc'][{'abc': 0, 'def': 1}][0][::2][3:100][{'xyz': 2}]
    #     self.assertEqual({ssq: 0, ssq: 1}, {ssq: 1})


class TestDataStore(TestCase):
    def test_create(self):
        with get_store() as obj:
            for tbl_name in ["data_records", "writers", "data_defs", "figure_defs"]:
                self.assertIn(tbl_name, obj._metadata.tables.keys())

    def test_get_data_handlers(self):
        num_arrays = 10
        time_index = -1
        with get_store() as store:
            dh = UniformNDArrayDataHandler(store, "arr1")

            arrs = [
                pgnp.random_array(
                    (10, 5, 7), dtype=[("f0", "datetime64"), ("f1", "int")]
                )
                for _ in range(num_arrays)
            ]

            [dh.add_data(time_index := time_index + 1, _arr) for _arr in arrs]
            store.flush()

            dat = dh.load_data()
            npt.assert_array_equal(dat["data"], np.array(arrs))

            for dh in [
                store.get_data_handlers(mdl.Col_("name") == "arr1")[0],
                # store.get_data_handlers(Col_('handler') == UniformNDArrayDataHandler)[0],# Not working.
            ]:
                npt.assert_array_equal(dh.load_data()["data"], np.array(arrs))

    def test_getitem__single_series(self):
        with get_store_with_data(
            num_uniform := [4, 3, 5, 2], num_ragged := [3, 8, 5]
        ) as (store, uniform, ragged):
            #
            for type_name, num, orig_data in [
                ("uniform", num_uniform, uniform),
                ("ragged", num_ragged, ragged),
            ]:
                for array_index in range(len(num)):
                    series_name = f"{type_name}_{array_index}"
                    #
                    stacked_arrays = orig_data[array_index]["arrays"]
                    npt.assert_array_equal(
                        stacked_arrays,
                        store.get_data_handlers(mdl.Col_("name") == series_name)[
                            0
                        ].load_data()["data"],
                    )
                    # Single array as string.
                    npt.assert_array_equal(
                        stacked_arrays, retrieved := store[series_name]["data"]
                    )
                    # Single array as tuple.
                    npt.assert_array_equal(
                        stacked_arrays,
                        retrieved := store[(series_name,)]["series"][series_name][
                            "data"
                        ],
                    )
                    # Single array as dictionary.
                    npt.assert_array_equal(
                        stacked_arrays,
                        retrieved := store[{"data": series_name}]["data"],
                    )

    def test_getitem__join(self):
        with get_store_with_data(
            num_uniform := [4, 3, 5, 2], num_ragged := [3, 8, 5]
        ) as (store, uniform, ragged):
            #
            for series_specs in [
                (("uniform", 2),),
                (("uniform", 0), ("ragged", 1)),
                (("uniform", 3), ("ragged", 0), ("uniform", 0), ("ragged", 2)),
                (("uniform", 2), ("ragged", 0), ("uniform", 1)),
            ]:
                series_names = [f"{type_name}_{k}" for type_name, k in series_specs]
                series_lengths = [
                    {"uniform": num_uniform, "ragged": num_ragged}[type_name][k]
                    for type_name, k in series_specs
                ]
                orig_data = [
                    {"uniform": uniform, "ragged": ragged}[type_name][k]
                    for type_name, k in series_specs
                ]
                joined_length = min(series_lengths)

                # Retrieve all
                retrieved = store[series_names]

                self.assertEqual(len(retrieved["series"]), len(series_names))
                self.assertEqual(len(retrieved["meta"]), joined_length)
                for _series_name, _orig_data in zip(series_names, orig_data):
                    npt.assert_array_equal(
                        _orig_data["arrays"][:joined_length],
                        retrieved["series"][_series_name]["data"],
                    )

                # Retrieve latest record.
                retrieved = store[{"data": series_names, "index": "latest"}]
                for _series_name, _orig_data in zip(series_names, orig_data):
                    npt.assert_array_equal(
                        _orig_data["arrays"][joined_length - 1][None, ...],
                        retrieved["series"][_series_name]["data"],
                    )

                # Retrieve k-th record.
                for index in range(joined_length):
                    retrieved = store[{"data": series_names, "index": index}]
                    for _series_name, _orig_data in zip(series_names, orig_data):
                        npt.assert_array_equal(
                            _orig_data["arrays"][index][None, ...],
                            retrieved["series"][_series_name]["data"],
                        )

                # Test call_multi
                ref0 = mdl.Ref_({"data": series_names, "index": index})
                (
                    source_data_pairs,
                    num_retrievals,
                    remainders,
                    output,
                ) = mdl.Ref_.call_multi(store, *[ref0] * 3, _test_output=True)
                self.assertEqual(num_retrievals, 1)
                ref0_data = ref0(store)
                [npt.assert_equal(_x, ref0_data) for _x in output]

                # # Retrieve non-existing record.
                # retrieved = store[{'data': series_names, 'index': joined_length}]
                # for _series_name, _orig_data in zip(series_names, orig_data):
                #     npt.assert_array_equal(
                #         _orig_data['arrays'][0][None, ...][:0],
                #         retrieved['series'][_series_name]['data'])

    @benchmark(False)
    def test_index_writer_order_by(self):
        pass

    @benchmark(False)
    def test_retrieve_non_existing(self):
        pass
