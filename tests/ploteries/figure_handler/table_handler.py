from ploteries.figure_handler import table_handler as mdl
from ploteries.data_store import Ref_
import numpy.testing as npt
from pglib.slice_sequence import SSQ_
import plotly.graph_objects as go
from ploteries.serializable_data_handler import SerializableDataHandler
from unittest import TestCase
from ..data_store import get_store
import numpy as np
from pglib.profiling import time_and_print
from contextlib import contextmanager
import dash


@contextmanager
def get_store_with_table(transposed=False):
    with get_store() as store:
        #
        data1_h = SerializableDataHandler(store, 'table_data')
        for index in range(5):
            data1_h.add_data(index, {f'Column {index}': index, f'Column {index+1}': index*2})

        #
        tab1_h = mdl.TableHandler(store, 'table1', 'table_data', transposed=transposed)

        store.flush()
        yield store, data1_h, tab1_h
        store.flush()


class TestTableHandler(TestCase):
    def test_create(self):

        with get_store_with_table() as (store, data1_h, tab1_h):
            built_tbl = tab1_h.build_table()

            #
            def check_valid(_tbl):
                as_array = np.array(
                    [
                        [None if x == '' else int(x) for x in _col]
                        for _col in _tbl['data'][0]['cells']['values']
                    ])
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
                built_tbl_json := built_tbl.to_json(),
                built_tbl_loaded.to_json())

    def test_encode_decode_params(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            orig_params = {
                'figure_dict': dict(fig_h.figure_dict)}
            mdl.FigureHandler.decode_params(decoded_params := fig_h.encode_params())
            self.assertDictEqual(orig_params, decoded_params)

    def test_from_traces(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            built_fig = fig_h.build_figure()
            fig_h_ft = mdl.FigureHandler.from_traces(
                store, 'from_traces', fig_h.figure_dict['data'])
            built_fig_ft = fig_h_ft.build_figure()

            self.assertEqual(
                built_fig.to_json(),
                built_fig_ft.to_json())

    def test_get_data_names(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig1_h):

            # Add another fig.
            arr3_h = UniformNDArrayDataHandler(store, 'arr3')
            arr3_h.add_data(
                0, np.array(
                    list(zip(range(20, 30), range(30, 40))),
                    dtype=[('f0', 'i'), ('f1', 'f')]))

            fig2_h = mdl.FigureHandler.from_traces(
                store, 'fig2',
                [{'x': Ref_('arr3')['data']['f0'],
                  'y':Ref_('arr3')['data']['f0']}])

            #
            self.assertEqual(
                set(fig1_h.get_data_names()),
                {'arr1', 'arr2'})
            self.assertEqual(
                set(fig2_h.get_data_names()),
                {'arr3'})

    def test_update(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            # fig_h.figure_dict['layout']['template']=None
            with self.assertRaisesRegex(Exception, r'Cannot update a definition that has not been retrieved from the data store.'):
                fig_h.write_def(mode='update')
            self.assertTrue(fig_h.write_def())
            self.assertFalse(fig_h.write_def())

            fig_h = store.get_figure_handlers()[0]
            self.assertIsInstance(fig_h.figure_dict['layout']['template'], dict)
            fig_h.figure_dict['layout']['template'] = None
            fig_h.write_def(mode='update')
            fig_h = store.get_figure_handlers()[0]
            self.assertIsNone(fig_h.figure_dict['layout']['template'])
