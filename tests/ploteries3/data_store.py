from unittest import TestCase
import ploteries3.data_store as mdl
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import pglib.numpy as pgnp
import numpy as np
import numpy.testing as npt
from sqlalchemy.sql import column as c
from ploteries3.ndarray_data_handlers import UniformNDArrayDataHandler, RaggedNDArrayDataHandler
from pglib.sqlalchemy import ClassType
from sqlalchemy.sql.expression import bindparam


@contextmanager
def get_store():
    with NamedTemporaryFile() as tmpf:
        obj = mdl.DataStore(tmpf.name)
        yield obj


@contextmanager
def get_store_with_data(num_uniform=[3, 5, 2], num_ragged=[11, 6, 4, 9]):
    with get_store() as store:
        uniform = []
        for uniform_k, num_time_indices in enumerate(num_uniform):
            dh = UniformNDArrayDataHandler(store, f'uniform_{uniform_k}')
            arrs = [
                pgnp.random_array((10, 5, 7), dtype=[('f0', 'datetime64[s]'), ('f1', 'int')])
                for _ in range(num_time_indices)]
            [dh.add_data(k, _arr) for k, _arr in enumerate(arrs)]
            uniform.append({'handler': dh, 'arrays': arrs})

        ragged = []
        for ragged_k, num_time_indices in enumerate(num_ragged):
            dh = RaggedNDArrayDataHandler(store, f'ragged_{ragged_k}')
            arrs = [
                pgnp.random_array(
                    (10 + ragged_k, 5 + ragged_k, 7 + ragged_k),
                    dtype=[('f0', 'datetime64[s]'),
                           ('f1', 'int')])
                for _ in range(num_time_indices)]
            [dh.add_data(k, _arr) for k, _arr in enumerate(arrs)]
            ragged.append({'handler': dh, 'arrays': arrs})

        yield store, uniform, ragged


class TestDataStore(TestCase):
    def test_create(self):
        with get_store() as obj:
            for tbl_name in ['data_records', 'writers', 'data_defs', 'figure_defs']:
                self.assertIn(tbl_name, obj._metadata.tables.keys())

    def test_get_data_handlers(self):
        num_arrays = 10
        with get_store() as store:
            dh = UniformNDArrayDataHandler(store, 'arr1')

            arrs = [pgnp.random_array((10, 5, 7), dtype=[('f0', 'datetime64'), ('f1', 'int')])
                    for _ in range(num_arrays)]

            [dh.add_data(0, _arr) for _arr in arrs]

            dat = dh.load_data()
            npt.assert_array_equal(dat['data'], np.array(arrs))

            for dh in [
                store.get_data_handlers(c('name') == 'arr1')[0],
                # store.get_data_handlers(c('handler') == UniformNDArrayDataHandler)[0],# Not working.
            ]:
                npt.assert_array_equal(
                    dh.load_data()['data'],
                    np.array(arrs))

    def test_getitem__single_series(self):

        with get_store_with_data(num_uniform := [4, 3, 5, 2], num_ragged := [3, 8, 5]) \
                as (store, uniform, ragged):

            #
            for type_name, num, orig_data in [
                    ('uniform', num_uniform, uniform),
                    ('ragged', num_ragged, ragged)]:
                for array_index in range(len(num_uniform)):

                    series_name = f'{type_name}_{array_index}'
                    #
                    stacked_arrays = orig_data[array_index]['arrays']
                    npt.assert_array_equal(
                        stacked_arrays,
                        store.get_data_handlers(
                            mdl.col('name') == series_name)[0].load_data()['data'])
                    npt.assert_array_equal(
                        stacked_arrays,
                        retrieved := store[series_name]['series'][series_name]['data'])

    def test_getitem__join(self):
        with get_store_with_data(num_uniform := [4, 3, 5, 2], num_ragged := [3, 8, 5]) \
                as (store, uniform, ragged):

            #
            for series_specs in [
                    (('uniform', 2),),
                    (('uniform', 0), ('ragged', 1)),
                    (('uniform', 3), ('ragged', 0), ('uniform', 0), ('ragged', 2)),
                    (('uniform', 2), ('ragged', 0), ('uniform', 1)),
            ]:
                series_names = [f'{type_name}_{k}' for type_name, k in series_specs]
                series_lengths = [
                    {'uniform': num_uniform, 'ragged': num_ragged}[type_name][k]
                    for type_name, k in series_specs]
                orig_data = [{'uniform': uniform, 'ragged': ragged}[type_name][k]
                             for type_name, k in series_specs]
                joined_length = min(series_lengths)

                retrieved = store[series_names]

                self.assertEqual(len(retrieved['series']), len(series_names))
                self.assertEqual(len(retrieved['meta']), joined_length)
                for _series_name, _orig_data in zip(series_names, orig_data):
                    npt.assert_array_equal(
                        _orig_data['arrays'][:joined_length],
                        retrieved['series'][_series_name]['data'])

                # for _series_name in series_names:
                #     #
                #     stacked_arrays = orig_data[array_index]['arrays']
                #     npt.assert_array_equal(
                #         stacked_arrays,
                #         store.get_data_handlers(
                #             mdl.col('name') == _series_name)[0].load_data()['data'])
                #     npt.assert_array_equal(
                #         stacked_arrays,
                #         retrieved)
