from ploteries3 import figure_handlers as mdl
import numpy.testing as npt
from pglib.py import SSQ
import plotly.graph_objects as go
from ploteries3.ndarray_data_handlers import UniformNDArrayDataHandler
from unittest import TestCase
from .data_store import get_store
import numpy as np
from pglib.profiling import time_and_print
from contextlib import contextmanager
from pglib.py import SSQ


@contextmanager
def get_store_with_fig():
    with get_store() as store:
        #
        arr1_h = UniformNDArrayDataHandler(store, 'arr1')
        arr1_h.add_data(
            0, np.array(
                list(zip(range(0, 10), range(10, 20))),
                dtype=[('f0', 'i'), ('f1', 'f')]))
        #
        arr2_h = UniformNDArrayDataHandler(store, 'arr2')
        arr2_h.add_data(
            0, np.array(
                list(zip(range(20, 30), range(30, 40))),
                dtype=[('f0', 'i'), ('f1', 'f')]))

        #
        fig = go.Figure()
        for k in range(2):
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'plot_{k}'))

        #
        fig_h = mdl.FigureHandler(
            store,
            'fig1',
            {'source1': 'arr1',
             'source2': 'arr2'},
            [{'figure_keys': ('data', 0, 'x'),
              'source_keys': ('source1', 'data', 0, 'f0')},
             {'figure_keys': SSQ()['data'][0]['y'],
              'source_keys': SSQ()['source1']['data'][0]['f1']},
             #
             (('data', 1, 'x'),
              ('source2', 'data', 0, 'f0')),
             (('data', 1, 'y'),
              ('source2', 'data', 0, 'f1'))],
            figure=fig)

        yield store, arr1_h, arr2_h, fig_h


class TestFigureHandler(TestCase):
    def test_create(self):

        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            built_fig = fig_h.build_figure()

            npt.assert_array_equal(
                SSQ()['data', 0, 'x'](built_fig),
                arr1_h.load_data()['data'][0]['f0'])

            npt.assert_array_equal(
                SSQ()['data', 0, 'y'](built_fig),
                arr1_h.load_data()['data'][0]['f1'])

            npt.assert_array_equal(
                SSQ()['data', 1, 'x'](built_fig),
                arr2_h.load_data()['data'][0]['f0'])

            npt.assert_array_equal(
                SSQ()['data', 1, 'y'](built_fig),
                arr2_h.load_data()['data'][0]['f1'])

            # Write the definition to the store
            fig_h.write_def()
            fig_h_loaded = mdl.FigureHandler.from_name(store, fig_h.name)

            # Compare figures.
            built_fig_loaded = fig_h_loaded.build_figure()
            self.assertEqual(
                built_fig.to_json(),
                built_fig_loaded.to_json())

            # Compare html
            self.assertEqual(
                str(fig_h.build_html()),
                str(fig_h_loaded.build_html()))

    def test_encode_decode_params(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            orig_params = {
                'sources': dict(fig_h.sources),
                'mappings': tuple(fig_h.mappings),
                'figure': dict(fig_h.figure)}
            mdl.FigureHandler.decode_params(decoded_params := fig_h.encode_params())
            self.assertDictEqual(orig_params, decoded_params)
